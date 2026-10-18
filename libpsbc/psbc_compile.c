/*
 * libpsbc — SPIR-V to PS4/PS5 Shader Binary Compiler Library
 *
 * Extracted from cmd/psbc/main.c. Contains the full compilation pipeline:
 *   SPIR-V → NIR (via radv_shader_spirv_to_nir)
 *   NIR optimization (radv_optimize_nir, radv_nir_lower_io)
 *   NIR postprocessing (radv_postprocess_nir)
 *   NIR → GCN ISA (via ACO: radv_shader_nir_to_asm)
 *   GCN ISA → GnmShaderFileHeader (buildshaderbinary)
 */

#include "psbc_compile.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#include <pssl_types.h>

#include "aco_interface.h"
#include "ac_gpu_info.h"
#include "ac_shader_util.h"
#include "ac_binary.h"
#include "ac_nir.h"
#include "amd_family.h"
#include "nir/radv_nir.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "radv_shader_info.h"
#include "radv_aco_shader_info.h"
#include "radv_pipeline.h"
#include "sid.h"

#include "crc32_sb.h"

/* === Mesa stage mapping === */

static mesa_shader_stage psbc_to_mesa_stage(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:     return MESA_SHADER_VERTEX;
    case PSBC_STAGE_TESS_CTRL:  return MESA_SHADER_TESS_CTRL;
    case PSBC_STAGE_TESS_EVAL:  return MESA_SHADER_TESS_EVAL;
    case PSBC_STAGE_GEOMETRY:   return MESA_SHADER_GEOMETRY;
    case PSBC_STAGE_FRAGMENT:   return MESA_SHADER_FRAGMENT;
    case PSBC_STAGE_COMPUTE:    return MESA_SHADER_COMPUTE;
    case PSBC_STAGE_TASK:       return MESA_SHADER_TASK;
    case PSBC_STAGE_EXPORT:     return MESA_SHADER_VERTEX;  /* ES is a VS variant */
    case PSBC_STAGE_LOCAL:      return MESA_SHADER_VERTEX;  /* LS is a VS variant */
    default:                    return MESA_SHADER_NONE;
    }
}

/* === Init/shutdown (refcounted) === */

static int g_init_refcount = 0;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

void psbc_init(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_init_refcount++ == 0) {
        glsl_type_singleton_init_or_ref();
    }
    pthread_mutex_unlock(&g_init_mutex);
}

void psbc_shutdown(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_init_refcount > 0 && --g_init_refcount == 0) {
        glsl_type_singleton_decref();
    }
    pthread_mutex_unlock(&g_init_mutex);
}

/* === PSSL/GNM type mapping === */

static PsslShaderType psbshtype(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return PSSL_SHADER_VS;
    case PSBC_STAGE_TESS_EVAL:   return PSSL_SHADER_VS;  /* DS outputs vertices */
    case PSBC_STAGE_EXPORT:      return PSSL_SHADER_VS;  /* ES is a VS variant */
    case PSBC_STAGE_LOCAL:       return PSSL_SHADER_VS;  /* LS is a VS variant */
    case PSBC_STAGE_FRAGMENT:    return PSSL_SHADER_FS;
    case PSBC_STAGE_COMPUTE:     return PSSL_SHADER_CS;
    case PSBC_STAGE_GEOMETRY:    return PSSL_SHADER_VS;  /* no PSSL GS type; use VS */
    case PSBC_STAGE_TESS_CTRL:   return PSSL_SHADER_VS;  /* no PSSL HS type; use VS */
    default:                      return PSSL_SHADER_VS;
    }
}

static GnmShaderType gnmshtype(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return GNM_SHADER_VERTEX;
    case PSBC_STAGE_TESS_EVAL:   return GNM_SHADER_VERTEX;  /* DS outputs vertices */
    case PSBC_STAGE_EXPORT:      return GNM_SHADER_VERTEX;  /* ES is a VS variant */
    case PSBC_STAGE_LOCAL:       return GNM_SHADER_VERTEX;  /* LS is a VS variant */
    case PSBC_STAGE_FRAGMENT:    return GNM_SHADER_PIXEL;
    case PSBC_STAGE_COMPUTE:     return GNM_SHADER_COMPUTE;
    case PSBC_STAGE_GEOMETRY:    return GNM_SHADER_GEOMETRY;
    case PSBC_STAGE_TESS_CTRL:   return GNM_SHADER_HULL;
    default:                      return GNM_SHADER_VERTEX;
    }
}

static GnmShaderBinaryType shbintype(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return GNM_SHB_VS_VS;
    case PSBC_STAGE_TESS_EVAL:   return GNM_SHB_DS_VS;
    case PSBC_STAGE_EXPORT:      return GNM_SHB_VS_ES;
    case PSBC_STAGE_LOCAL:       return GNM_SHB_VS_LS;
    case PSBC_STAGE_FRAGMENT:    return GNM_SHB_PS;
    case PSBC_STAGE_COMPUTE:     return GNM_SHB_CS;
    case PSBC_STAGE_GEOMETRY:    return GNM_SHB_GS;
    case PSBC_STAGE_TESS_CTRL:   return GNM_SHB_HS;
    default:                      return GNM_SHB_VS_VS;
    }
}

static uint32_t headershsize(PsbcStage s) {
    switch (s) {
    case PSBC_STAGE_VERTEX:      return sizeof(GnmVsShader);
    case PSBC_STAGE_TESS_EVAL:   return sizeof(GnmVsShader);  /* DS uses VS struct */
    case PSBC_STAGE_EXPORT:      return sizeof(GnmEsShader);
    case PSBC_STAGE_LOCAL:       return sizeof(GnmLsShader);
    case PSBC_STAGE_FRAGMENT:    return sizeof(GnmPsShader);
    case PSBC_STAGE_COMPUTE:     return sizeof(GnmCsShader);
    case PSBC_STAGE_GEOMETRY:    return sizeof(GnmGsShader);
    case PSBC_STAGE_TESS_CTRL:   return sizeof(GnmHsShader);
    default:                      return sizeof(GnmVsShader);
    }
}

static uint32_t hashsb(
    const void* code, uint32_t codesize, const GnmShaderBinaryInfo* sb
) {
    static const uint8_t padbytes[8] = {0};
    const uint32_t numpadbytes =
        (sb->length & 0x7) ? 8 - (sb->length & 0x7) : 0;
    assert(numpadbytes <= sizeof(padbytes));

    uint32_t hash = crc32_sb(code, codesize, crc32_sb_begin());
    hash = crc32_sb(padbytes, numpadbytes, hash);
    hash = crc32_sb(sb, sizeof(*sb) - sizeof(sb->crc32), hash);
    return crc32_sb_end(hash);
}

/*
 * Compute the number of position exports from shader info.
 * Ported from radv_get_num_pos_exports() which is static in radv_shader.c.
 */
static unsigned get_num_pos_exports(const struct radv_shader_info *info,
                                    unsigned *clip_dist_mask_out,
                                    unsigned *cull_dist_mask_out) {
    unsigned num = 1;

    if (info->outinfo.writes_pointsize || info->outinfo.writes_viewport_index ||
        info->outinfo.writes_layer || info->outinfo.writes_primitive_shading_rate)
        num++;

    unsigned num_clip_dist_comps = util_bitcount(info->outinfo.clip_dist_mask);
    unsigned num_cull_dist_comps = info->has_ngg_culling ? 0 : util_bitcount(info->outinfo.cull_dist_mask);
    unsigned clip_cull_mask = BITFIELD_MASK(num_clip_dist_comps + num_cull_dist_comps);

    if (clip_cull_mask & 0x0f)
        num++;
    if (clip_cull_mask & 0xf0)
        num++;

    if (clip_dist_mask_out)
        *clip_dist_mask_out = BITFIELD_MASK(num_clip_dist_comps);
    if (cull_dist_mask_out)
        *cull_dist_mask_out = BITFIELD_RANGE(num_clip_dist_comps, num_cull_dist_comps);
    return num;
}

/*
 * Compute db_shader_control register value for PS.
 * Ported from the inline computation in radv_precompute_registers_hw_ps().
 */
static uint32_t compute_db_shader_control(const struct radv_shader_info *info) {
    unsigned conservative_z_export = V_02880C_EXPORT_ANY_Z;
    if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_GREATER)
        conservative_z_export = V_02880C_EXPORT_GREATER_THAN_Z;
    else if (info->ps.depth_layout == FRAG_DEPTH_LAYOUT_LESS)
        conservative_z_export = V_02880C_EXPORT_LESS_THAN_Z;

    const unsigned z_order =
        info->ps.early_fragment_test || !info->ps.writes_memory
            ? V_02880C_EARLY_Z_THEN_LATE_Z : V_02880C_LATE_Z;

    return S_02880C_Z_EXPORT_ENABLE(info->ps.writes_z) |
           S_02880C_STENCIL_TEST_VAL_EXPORT_ENABLE(info->ps.writes_stencil) |
           S_02880C_KILL_ENABLE(info->ps.can_discard) |
           S_02880C_MASK_EXPORT_ENABLE(info->ps.writes_sample_mask) |
           S_02880C_CONSERVATIVE_Z_EXPORT(conservative_z_export) |
           S_02880C_Z_ORDER(z_order) |
           S_02880C_DEPTH_BEFORE_SHADER(info->ps.early_fragment_test) |
           S_02880C_PRE_SHADER_DEPTH_COVERAGE_ENABLE(info->ps.post_depth_coverage) |
           S_02880C_EXEC_ON_HIER_FAIL(info->ps.writes_memory) |
           S_02880C_EXEC_ON_NOOP(info->ps.writes_memory) |
           S_02880C_PRIMITIVE_ORDERED_PIXEL_SHADER(info->ps.pops);
}

/* === Shader binary builder === */

typedef struct {
    const struct nir_shader* nir;
    const struct radv_shader_info* rinfo;
    const struct radv_shader_args* rargs;
    const struct ac_shader_config* config;
    enum amd_gfx_level gfx_level;
    enum radeon_family family;
    mesa_shader_stage stage;
    PsbcStage psbc_stage;
    const uint32_t* spirv_data;
    size_t spirv_size;
    bool neo;
} BuildContext;

/* Build the shader binary into a memory buffer instead of a file. */
static PsbcResult buildshaderbinary(
    const BuildContext* ctx,
    const uint32_t* code, uint32_t code_dw,
    uint8_t** out_data, size_t* out_size
) {
    uint32_t codesize = code_dw * sizeof(uint32_t);
    if (ctx->nir->info.stage == MESA_SHADER_VERTEX &&
        ctx->rinfo->vs.has_prolog) {
        codesize += sizeof(uint32_t);
    }

    uint32_t* newcode = malloc(codesize);
    if (!newcode)
        return PSBC_RESULT_OUT_OF_MEMORY;
    uint32_t* nextcode = newcode;

    /* prepend vertex shaders that use vertex input with essentially a
     * jump instruction to a fetch "shader" */
    if (ctx->nir->info.stage == MESA_SHADER_VERTEX &&
        ctx->rinfo->vs.has_prolog) {
        /* s_swappc_b64 s[0:1], s[0:1] */
        newcode[0] = 0xbe802100;
        nextcode += 1;
    }

    memcpy(nextcode, code, code_dw * sizeof(uint32_t));

    /* Count input usage slots.
     * prolog_inputs (SUBPTR_FETCHSHADER) and vertex_buffers
     * (PTR_VERTEXBUFFERTABLE) are VS-specific — they belong to the VS
     * portion of a merged pipeline. For standalone HS/GS compilation,
     * these args are declared because previous_stage=VERTEX, but the
     * slots should not be emitted in the HS/GS shader binary.
     * The descriptor table (PTR_INDIRECTRESOURCETABLE) is shared by
     * all stages and should always be emitted when used. */
    const bool is_vs_family =
        ctx->stage == MESA_SHADER_VERTEX ||
        ctx->psbc_stage == PSBC_STAGE_EXPORT ||
        ctx->psbc_stage == PSBC_STAGE_LOCAL;

    uint32_t numinputslots = 0;
    if (is_vs_family && ctx->rargs->prolog_inputs.used) {
        numinputslots += 1;
    }
    if (is_vs_family && ctx->rargs->ac.vertex_buffers.used) {
        numinputslots += 1;
    }
    if (ctx->rargs->descriptors[0].used) {
        numinputslots += 1;
    }

    uint32_t shspecificsize = headershsize(ctx->psbc_stage);
    shspecificsize += numinputslots * sizeof(GnmInputUsageSlot);

    /* exclude position as it doesn't count as an export semantic in PSB */
    const uint32_t outputswritten =
        util_bitcount64(ctx->nir->info.outputs_written & ~VARYING_BIT_POS);

    switch (ctx->stage) {
    case MESA_SHADER_VERTEX:
    case MESA_SHADER_TESS_EVAL:  /* DS outputs vertices like VS */
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmVertexInputSemantic);
        shspecificsize +=
            outputswritten * sizeof(GnmVertexExportSemantic);
        break;
    case MESA_SHADER_FRAGMENT:
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmPixelInputSemantic);
        break;
    case MESA_SHADER_GEOMETRY:
        /* GS has input/export semantics like VS */
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmVertexInputSemantic);
        shspecificsize +=
            outputswritten * sizeof(GnmVertexExportSemantic);
        break;
    case MESA_SHADER_TESS_CTRL:
        /* HS has input semantics (from VS outputs) but no export semantics */
        shspecificsize += util_bitcount64(ctx->nir->info.inputs_read) *
                          sizeof(GnmVertexInputSemantic);
        break;
    default:
        break;
    }

    /* GCN bytecode that comes after the headers must be 4 byte aligned */
    const uint32_t alignbytes =
        ((shspecificsize + 3) & (-4)) - shspecificsize;
    shspecificsize += alignbytes;

    /* Calculate total size */
    const size_t total_size = sizeof(PsslBinaryHeader) +
                              sizeof(GnmShaderFileHeader) +
                              shspecificsize +
                              codesize +
                              sizeof(GnmShaderBinaryInfo) +
                              sizeof(PsslBinaryParamInfo);

    uint8_t* buf = malloc(total_size);
    if (!buf) {
        free(newcode);
        return PSBC_RESULT_OUT_OF_MEMORY;
    }
    size_t offset = 0;

    /* PSSL binary header */
    const PsslBinaryHeader psbh = {
        .vermajor = 0,
        .verminor = 4,
        .shadertype = psbshtype(ctx->psbc_stage),
        .codetype = PSSL_CODE_ISA,
        .compilertype = PSSL_COMPILER_UNSPECIFIED,
        .codesize = sizeof(GnmShaderFileHeader) + shspecificsize +
                    codesize + sizeof(GnmShaderBinaryInfo),
    };
    memcpy(buf + offset, &psbh, sizeof(psbh));
    offset += sizeof(psbh);

    /* GNM shader file header */
    const GnmShaderFileHeader gsfh = {
        .magic = GNM_SHADER_FILE_HEADER_ID,
        .vermajor = 7,
        .verminor = 2,
        .type = gnmshtype(ctx->psbc_stage),
        .headersizedwords = shspecificsize / 4,
        .targetgpumodes =
            (ctx->gfx_level >= GFX10_3) ? GNM_TARGETGPUMODE_NEO :
            (ctx->neo ? GNM_TARGETGPUMODE_NEO : GNM_TARGETGPUMODE_BASE),
    };
    memcpy(buf + offset, &gsfh, sizeof(gsfh));
    offset += sizeof(gsfh);

    /* Shader-specific header (VS/PS/GS/HS/ES/LS registers) */
    switch (ctx->psbc_stage) {
    case PSBC_STAGE_VERTEX:
    case PSBC_STAGE_TESS_EVAL: {
        const uint32_t nparams =
            MAX2(ctx->rinfo->outinfo.param_exports, 1);
        unsigned clip_dist_mask = 0, cull_dist_mask = 0;
        const unsigned num_pos_exports =
            get_num_pos_exports(ctx->rinfo, &clip_dist_mask, &cull_dist_mask);
        const uint32_t total_mask = clip_dist_mask | cull_dist_mask;
        const bool misc_vec_ena =
            ctx->rinfo->outinfo.writes_pointsize ||
            ctx->rinfo->outinfo.writes_layer ||
            ctx->rinfo->outinfo.writes_viewport_index ||
            ctx->rinfo->outinfo.writes_primitive_shading_rate;

        const GnmVsShader vsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlovs = shspecificsize,
                    .spishaderpgmhivs = 0xffffffff,
                    .spishaderpgmrsrc1vs = ctx->config->rsrc1,
                    .spishaderpgmrsrc2vs = ctx->config->rsrc2,
                    .spivsoutconfig =
                        S_0286C4_VS_EXPORT_COUNT(nparams - 1),
                    .spishaderposformat =
                        S_02870C_POS0_EXPORT_FORMAT(
                            V_02870C_SPI_SHADER_4COMP
                        ) |
                        S_02870C_POS1_EXPORT_FORMAT(
                            num_pos_exports > 1
                                ? V_02870C_SPI_SHADER_4COMP
                                : V_02870C_SPI_SHADER_NONE
                        ) |
                        S_02870C_POS2_EXPORT_FORMAT(
                            num_pos_exports > 2
                                ? V_02870C_SPI_SHADER_4COMP
                                : V_02870C_SPI_SHADER_NONE
                        ) |
                        S_02870C_POS3_EXPORT_FORMAT(
                            num_pos_exports > 3
                                ? V_02870C_SPI_SHADER_4COMP
                                : V_02870C_SPI_SHADER_NONE
                        ),
                    .paclvsoutcntl =
                        S_02881C_USE_VTX_POINT_SIZE(
                            ctx->rinfo->outinfo.writes_pointsize
                        ) |
                        S_02881C_USE_VTX_RENDER_TARGET_INDX(
                            ctx->rinfo->outinfo.writes_layer
                        ) |
                        S_02881C_USE_VTX_VIEWPORT_INDX(
                            ctx->rinfo->outinfo.writes_viewport_index
                        ) |
                        S_02881C_USE_VTX_VRS_RATE(
                            ctx->rinfo->outinfo
                                .writes_primitive_shading_rate
                        ) |
                        S_02881C_VS_OUT_MISC_VEC_ENA(misc_vec_ena) |
                        S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(
                            misc_vec_ena ||
                            (ctx->gfx_level >= GFX10_3 &&
                             num_pos_exports > 1)
                        ) |
                        S_02881C_VS_OUT_CCDIST0_VEC_ENA(
                            (total_mask & 0x0f) != 0
                        ) |
                        S_02881C_VS_OUT_CCDIST1_VEC_ENA(
                            (total_mask & 0xf0) != 0
                        ) |
                        total_mask << 8 |
                        clip_dist_mask,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &vsh, sizeof(vsh));
        offset += sizeof(vsh);
        break;
    }
    case PSBC_STAGE_FRAGMENT: {
        const bool param_gen = ctx->gfx_level >= GFX11 &&
                               !ctx->rinfo->ps.num_inputs &&
                               ctx->config->lds_size;

        const GnmPsShader psh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlops = shspecificsize,
                    .spishaderpgmhips = 0xffffffff,
                    .spishaderpgmrsrc1ps = ctx->config->rsrc1,
                    .spishaderpgmrsrc2ps = ctx->config->rsrc2,
                    .spishaderzformat = ac_get_spi_shader_z_format(
                        ctx->rinfo->ps.writes_z,
                        ctx->rinfo->ps.writes_stencil,
                        ctx->rinfo->ps.writes_sample_mask,
                        ctx->rinfo->ps.writes_mrt0_alpha
                    ),
                    .spishadercolformat =
                        ctx->rinfo->ps.spi_shader_col_format,
                    .spipsinputena = ctx->config->spi_ps_input_ena,
                    .spipsinputaddr = ctx->config->spi_ps_input_addr,
                    .spipsincontrol =
                        (ctx->gfx_level != GFX10_3
                             ? S_0286D8_NUM_INTERP(ctx->rinfo->ps.num_inputs)
                             : 0) |
                        S_0286D8_PS_W32_EN(
                            ctx->rinfo->wave_size == 32
                        ) |
                        S_0286D8_PARAM_GEN(param_gen),
                    .spibaryccntl = S_0286E0_FRONT_FACE_ALL_BITS(1),
                    .dbshadercontrol = compute_db_shader_control(ctx->rinfo),
                    .cbshadermask = ac_get_cb_shader_mask(
                        ctx->rinfo->ps.spi_shader_col_format
                    ),
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
        };
        memcpy(buf + offset, &psh, sizeof(psh));
        offset += sizeof(psh);
        break;
    }
    case PSBC_STAGE_COMPUTE: {
        const GnmCsShader csh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .computepgmlo = shspecificsize,
                    .computepgmhi = 0,
                    .computepgmrsrc1 = ctx->config->rsrc1,
                    .computepgmrsrc2 = ctx->config->rsrc2,
                    /* Thread group size from NIR compute shader info */
                    .computenumthreadx =
                        ctx->nir->info.workgroup_size[0] ?
                        ctx->nir->info.workgroup_size[0] : 1,
                    .computenumthready =
                        ctx->nir->info.workgroup_size[1] ?
                        ctx->nir->info.workgroup_size[1] : 1,
                    .computenumthreadz =
                        ctx->nir->info.workgroup_size[2] ?
                        ctx->nir->info.workgroup_size[2] : 1,
                },
        };
        memcpy(buf + offset, &csh, sizeof(csh));
        offset += sizeof(csh);
        break;
    }
    case PSBC_STAGE_GEOMETRY: {
        /* Map mesa_prim to VGT_GS_OUT_PRIM_TYPE values */
        uint32_t gs_out_prim;
        switch (ctx->nir->info.gs.output_primitive) {
        case MESA_PRIM_POINTS:       gs_out_prim = V_028A6C_POINTLIST; break;
        case MESA_PRIM_LINES:        gs_out_prim = V_028A6C_LINESTRIP; break;
        case MESA_PRIM_TRIANGLES:    gs_out_prim = V_028A6C_TRISTRIP;  break;
        default:                     gs_out_prim = V_028A6C_TRISTRIP;  break;
        }

        const GnmGsShader gsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlogs = shspecificsize,
                    .spishaderpgmhigs = 0xffffffff,
                    .spishaderpgmrsrc1gs = ctx->config->rsrc1,
                    .spishaderpgmrsrc2gs = ctx->config->rsrc2,
                    .vgtstrmoutconfig = 0,
                    .vgtgsoutprimtype = gs_out_prim,
                    .vgtgsinstancecnt =
                        S_028B90_CNT(MIN2(ctx->rinfo->gs.invocations, 127)) |
                        S_028B90_ENABLE(ctx->rinfo->gs.invocations > 0),
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &gsh, sizeof(gsh));
        offset += sizeof(gsh);
        break;
    }
    case PSBC_STAGE_TESS_CTRL: {
        /* Map tess primitive mode to VGT_TF_PARAM TYPE field */
        uint32_t tf_type;
        switch (ctx->nir->info.tess._primitive_mode) {
        case TESS_PRIMITIVE_ISOLINES:   tf_type = V_028B6C_TESS_ISOLINE;  break;
        case TESS_PRIMITIVE_TRIANGLES:  tf_type = V_028B6C_TESS_TRIANGLE; break;
        case TESS_PRIMITIVE_QUADS:      tf_type = V_028B6C_TESS_QUAD;     break;
        default:                        tf_type = V_028B6C_TESS_TRIANGLE; break;
        }

        /* Map tess spacing to VGT_TF_PARAM PARTITIONING field */
        uint32_t tf_partition;
        switch (ctx->nir->info.tess.spacing) {
        case TESS_SPACING_EQUAL:           tf_partition = V_028B6C_PART_INTEGER;    break;
        case TESS_SPACING_FRACTIONAL_ODD:  tf_partition = V_028B6C_PART_FRAC_ODD;   break;
        case TESS_SPACING_FRACTIONAL_EVEN: tf_partition = V_028B6C_PART_FRAC_EVEN;  break;
        default:                           tf_partition = V_028B6C_PART_INTEGER;    break;
        }

        /* Map CCW + point_mode to VGT_TF_PARAM TOPOLOGY field */
        uint32_t tf_topology;
        if (ctx->nir->info.tess.point_mode) {
            tf_topology = V_028B6C_OUTPUT_POINT;
        } else if (ctx->nir->info.tess.ccw) {
            tf_topology = V_028B6C_OUTPUT_TRIANGLE_CCW;
        } else {
            tf_topology = V_028B6C_OUTPUT_TRIANGLE_CW;
        }

        const uint32_t tf_param =
            S_028B6C_TYPE(tf_type) |
            S_028B6C_PARTITIONING(tf_partition) |
            S_028B6C_TOPOLOGY(tf_topology);

        const GnmHsShader hsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlohs = shspecificsize,
                    .spishaderpgmhihs = 0xffffffff,
                    .spishaderpgmrsrc1hs = ctx->config->rsrc1,
                    .spishaderpgmrsrc2hs = ctx->config->rsrc2,
                    .vgttfparam = tf_param,
                    /* Tessellation levels — use defaults if not available */
                    .vgthosmaxtesslevel = 64,
                    .vgthosmintesslevel = 1,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
        };
        memcpy(buf + offset, &hsh, sizeof(hsh));
        offset += sizeof(hsh);
        break;
    }
    case PSBC_STAGE_EXPORT: {
        /* ES (Export Shader) — VS variant with ES registers.
         * Runs before GS in a geometry pipeline. */
        const GnmEsShader esh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmloes = shspecificsize,
                    .spishaderpgmhies = 0xffffffff,
                    .spishaderpgmrsrc1es = ctx->config->rsrc1,
                    .spishaderpgmrsrc2es = ctx->config->rsrc2,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &esh, sizeof(esh));
        offset += sizeof(esh);
        break;
    }
    case PSBC_STAGE_LOCAL: {
        /* LS (Local Shader) — VS variant with LS registers.
         * Runs before HS in a tessellation pipeline. */
        const GnmLsShader lsh = {
            .common =
                {
                    .shadersize =
                        codesize + sizeof(GnmShaderBinaryInfo),
                    .numinputusageslots = numinputslots,
                },
            .registers =
                {
                    .spishaderpgmlols = shspecificsize,
                    .spishaderpgmhils = 0xffffffff,
                    .spishaderpgmrsrc1ls = ctx->config->rsrc1,
                    .spishaderpgmrsrc2ls = ctx->config->rsrc2,
                },
            .numinputsemantics =
                util_bitcount64(ctx->nir->info.inputs_read),
            .numexportsemantics = outputswritten,
        };
        memcpy(buf + offset, &lsh, sizeof(lsh));
        offset += sizeof(lsh);
        break;
    }
    default:
        free(newcode);
        free(buf);
        return PSBC_RESULT_UNSUPPORTED_STAGE;
    }

    /* write shader common data: input usage slots */
    if (is_vs_family && ctx->rargs->prolog_inputs.used) {
        const GnmInputUsageSlot s = {
            .usagetype = GNM_SHINPUTUSAGE_SUBPTR_FETCHSHADER,
            .startregister =
                ctx->rargs->ac.args[ctx->rargs->prolog_inputs.arg_index]
                    .offset,
        };
        memcpy(buf + offset, &s, sizeof(s));
        offset += sizeof(s);
    }
    if (is_vs_family && ctx->rargs->ac.vertex_buffers.used) {
        const GnmInputUsageSlot s = {
            .usagetype = GNM_SHINPUTUSAGE_PTR_VERTEXBUFFERTABLE,
            .startregister =
                ctx->rargs->ac
                    .args[ctx->rargs->ac.vertex_buffers.arg_index]
                    .offset,
        };
        memcpy(buf + offset, &s, sizeof(s));
        offset += sizeof(s);
    }
    if (ctx->rargs->descriptors[0].used) {
        const GnmInputUsageSlot s = {
            .usagetype = GNM_SHINPUTUSAGE_PTR_INDIRECTRESOURCETABLE,
            .startregister =
                ctx->rargs->ac
                    .args[ctx->rargs->descriptors[0].arg_index]
                    .offset,
        };
        memcpy(buf + offset, &s, sizeof(s));
        offset += sizeof(s);
    }

    /* write input/export semantics */
    switch (ctx->stage) {
    case MESA_SHADER_VERTEX:
    case MESA_SHADER_TESS_EVAL:  /* DS uses same semantics as VS */
        for (uint32_t i = 0;
             i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
            const GnmVertexInputSemantic input = {
                .semantic = i,
                .vgpr =
                    ctx->rargs->ac
                        .args[ctx->rargs->vs_inputs[i].arg_index]
                        .offset,
                .sizeinelements =
                    ctx->rargs->ac
                        .args[ctx->rargs->vs_inputs[i].arg_index]
                        .size,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        for (uint32_t i = 0; i < outputswritten; i += 1) {
            const GnmVertexExportSemantic out = {
                .semantic = 15 + i,
                .outindex = i,
                .exportf16 = 0,
            };
            memcpy(buf + offset, &out, sizeof(out));
            offset += sizeof(out);
        }
        break;
    case MESA_SHADER_FRAGMENT:
        for (uint32_t i = 0;
             i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
            const GnmPixelInputSemantic input = {
                .semantic = 15 + i,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        break;
    case MESA_SHADER_GEOMETRY:
        /* GS input/export semantics use the same vertex semantic format */
        for (uint32_t i = 0;
             i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
            const GnmVertexInputSemantic input = {
                .semantic = i,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        for (uint32_t i = 0; i < outputswritten; i += 1) {
            const GnmVertexExportSemantic out = {
                .semantic = 15 + i,
                .outindex = i,
                .exportf16 = 0,
            };
            memcpy(buf + offset, &out, sizeof(out));
            offset += sizeof(out);
        }
        break;
    case MESA_SHADER_TESS_CTRL:
        /* HS input semantics describe which VS outputs are read.
         * No export semantics — HS outputs go to LDS for DS. */
        for (uint32_t i = 0;
             i < util_bitcount64(ctx->nir->info.inputs_read); i += 1) {
            const GnmVertexInputSemantic input = {
                .semantic = i,
            };
            memcpy(buf + offset, &input, sizeof(input));
            offset += sizeof(input);
        }
        break;
    default:
        break;
    }

    /* alignment padding */
    memset(buf + offset, 0, alignbytes);
    offset += alignbytes;

    /* shader code */
    memcpy(buf + offset, newcode, codesize);
    offset += codesize;

    /* shader binary info */
    /* Compute chunkusagebaseoffsetdwords: offset from OrbShdr back to the
     * input usage slot table, in dwords.
     * Input usage slots are at: PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE + headershsize
     * OrbShdr is at: current offset (= PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE + shspecificsize + codesize)
     * Distance = offset - (PSSL_HEADER_SIZE + GNM_FILE_HEADER_SIZE + headershsize)
     * But the comment says "starts at ((uint32_t*)&ShaderBinaryInfo) - chunkusagebaseoffsetdwords"
     * so chunkusagebaseoffsetdwords = distance / 4
     * Note: if numinputslots == 0, the offset is 0 (no table to point to). */
    const size_t inputslots_offset =
        sizeof(PsslBinaryHeader) + sizeof(GnmShaderFileHeader) +
        headershsize(ctx->psbc_stage);
    const size_t orbshdr_offset = offset;
    const uint32_t chunkusageoffset =
        numinputslots > 0
            ? (uint32_t)((orbshdr_offset - inputslots_offset) / 4)
            : 0;

    /* Compute shader hash from SPIR-V data.
     * The PS4 uses this for shader cache identification.
     * We use a simple hash of the SPIR-V binary. */
    uint64_t shaderhash = 0;
    if (ctx->spirv_data && ctx->spirv_size > 0) {
        /* FNV-1a hash of the SPIR-V bytes */
        const uint8_t* sp = (const uint8_t*)ctx->spirv_data;
        shaderhash = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < ctx->spirv_size; i += 1) {
            shaderhash ^= sp[i];
            shaderhash *= 0x100000001b3ULL;
        }
    }

    GnmShaderBinaryInfo bininfo = {
        .signature = GNM_SHADER_BINARY_INFO_MAGIC,
        .version = 7,
        .ispsslcg = 1,
        .type = shbintype(ctx->psbc_stage),
        .length = codesize,
        .chunkusagebaseoffsetdwords = chunkusageoffset,
        .numinputusageslots = numinputslots,
        .shaderhash0 = (uint32_t)(shaderhash & 0xffffffff),
        .shaderhash1 = (uint32_t)(shaderhash >> 32),
    };
    bininfo.crc32 = hashsb(newcode, codesize, &bininfo);
    memcpy(buf + offset, &bininfo, sizeof(bininfo));
    offset += sizeof(bininfo);

    /* param info */
    const PsslBinaryParamInfo paraminfo = {0};
    memcpy(buf + offset, &paraminfo, sizeof(paraminfo));
    offset += sizeof(paraminfo);

    free(newcode);

    *out_data = buf;
    *out_size = offset;
    return PSBC_RESULT_OK;
}

/* === GPU target setup === */

static void setup_ac_info(struct ac_compiler_info* ac_info, enum amd_gfx_level gfxlevel) {
    ac_info->gfx_level = gfxlevel;

    if (gfxlevel >= GFX10_3) {
        ac_info->max_waves_per_simd = 16;
        ac_info->num_physical_sgprs_per_simd = 128 * 16;
        ac_info->num_physical_wave64_vgprs_per_simd = 256;
        ac_info->num_simd_per_compute_unit = 2;
        ac_info->min_sgpr_alloc = 1;
        ac_info->max_sgpr_alloc = 128;
        ac_info->sgpr_alloc_granularity = 1;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 256;
        ac_info->wave64_vgpr_alloc_granularity = 4;
        ac_info->has_packed_math_16bit = true;
        ac_info->has_fma_mix = true;
    } else if (gfxlevel == GFX10) {
        ac_info->max_waves_per_simd = 20;
        ac_info->num_physical_sgprs_per_simd = 128 * 20;
        ac_info->num_physical_wave64_vgprs_per_simd = 256;
        ac_info->num_simd_per_compute_unit = 2;
        ac_info->min_sgpr_alloc = 1;
        ac_info->max_sgpr_alloc = 128;
        ac_info->sgpr_alloc_granularity = 1;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 256;
        ac_info->wave64_vgpr_alloc_granularity = 4;
        ac_info->has_packed_math_16bit = true;
    } else if (gfxlevel == GFX8) {
        /* same limits as ac_fill_compiler_info() in Mesa for GFX8 */
        ac_info->max_waves_per_simd = 10;
        ac_info->num_physical_sgprs_per_simd = 800;
        ac_info->num_physical_wave64_vgprs_per_simd = 256;
        ac_info->num_simd_per_compute_unit = 4;
        ac_info->min_sgpr_alloc = 16;
        ac_info->max_sgpr_alloc = 102;
        ac_info->sgpr_alloc_granularity = 16;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 256;
        ac_info->wave64_vgpr_alloc_granularity = 4;
    } else {
        /* GFX7 */
        /* same limits as ac_fill_compiler_info() in Mesa for GFX7, the old
         * values (12 SGPRs, 64 VGPRs) made any shader with a texture fail
         * register allocation */
        ac_info->max_waves_per_simd = 10;
        ac_info->num_physical_sgprs_per_simd = 512;
        ac_info->num_physical_wave64_vgprs_per_simd = 256;
        ac_info->num_simd_per_compute_unit = 4;
        ac_info->min_sgpr_alloc = 8;
        ac_info->max_sgpr_alloc = 104;
        ac_info->sgpr_alloc_granularity = 8;
        ac_info->min_wave64_vgpr_alloc = 4;
        ac_info->max_vgpr_alloc = 256;
        ac_info->wave64_vgpr_alloc_granularity = 4;
    }
}

static void setup_target(PsbcTarget target,
                         enum amd_gfx_level* gfxlevel,
                         enum radeon_family* chipfamily,
                         bool* neo) {
    switch (target) {
    case PSBC_TARGET_PS5:
        *gfxlevel = GFX10_3;
        *chipfamily = CHIP_NAVI21;
        *neo = false;
        break;
    case PSBC_TARGET_PS4_NEO:
        *gfxlevel = GFX8;
        *chipfamily = CHIP_TONGA;
        *neo = true;
        break;
    case PSBC_TARGET_PS4_BASE:
    default:
        *gfxlevel = GFX7;
        *chipfamily = CHIP_KAVERI;
        *neo = false;
        break;
    }
}

/* === Main compilation function === */

PsbcResult psbc_compile_shader(
    const uint32_t*       spirv,
    size_t                spirv_size,
    const PsbcCompileOptions* opts,
    PsbcShaderOutput*     out
) {
    if (!spirv || !opts || !out)
        return PSBC_RESULT_INTERNAL_ERROR;

    memset(out, 0, sizeof(*out));

    /* Validate SPIR-V header */
    if (spirv_size < 4 || spirv[0] != 0x07230203u)
        return PSBC_RESULT_INVALID_SPIRV;

    mesa_shader_stage mesa_stage = psbc_to_mesa_stage(opts->stage);
    if (mesa_stage == MESA_SHADER_NONE)
        return PSBC_RESULT_UNSUPPORTED_STAGE;

    /* Ensure library is initialized */
    psbc_init();

    /* Select GFX level and family based on target */
    enum amd_gfx_level gfxlevel;
    enum radeon_family chipfamily;
    bool neo = false;
    setup_target(opts->target, &gfxlevel, &chipfamily, &neo);

    /* Construct ac_compiler_info for the target GPU */
    struct ac_compiler_info ac_info = {0};
    setup_ac_info(&ac_info, gfxlevel);

    /* Construct radv_compiler_info */
    struct radv_compiler_info compiler_info = {0};
    compiler_info.ac = &ac_info;
    compiler_info.hw.address32_hi = 0;
    compiler_info.key.ge_wave_size = 64;
    compiler_info.key.ps_wave_size = (gfxlevel >= GFX10_3) ? 32 : 64;
    compiler_info.key.cs_wave_size = (gfxlevel >= GFX10_3) ? 32 : 64;
    compiler_info.key.rt_wave_size = 64;
    compiler_info.key.family = chipfamily;
    compiler_info.key.load_grid_size_from_user_sgpr = (gfxlevel >= GFX10_3);
    /* ACO uses debug.family for disassembly and init_program assertion */
    compiler_info.debug.family = chipfamily;

    /* Initialize NIR options for all stages */
    radv_get_nir_options(&compiler_info);

    /* Construct radv_shader_stage */
    struct radv_shader_stage stage = {0};
    stage.stage = mesa_stage;
    /* Set next_stage based on the pipeline graph.
     * For standalone compilation we assume the simplest pipeline:
     *   VS → FS, VS → HS → DS → FS, VS → GS → FS
     */
    switch (mesa_stage) {
    case MESA_SHADER_VERTEX:    stage.next_stage = MESA_SHADER_FRAGMENT;     break;
    case MESA_SHADER_TESS_CTRL: stage.next_stage = MESA_SHADER_TESS_EVAL;    break;
    case MESA_SHADER_TESS_EVAL: stage.next_stage = MESA_SHADER_FRAGMENT;     break;
    case MESA_SHADER_GEOMETRY:  stage.next_stage = MESA_SHADER_FRAGMENT;     break;
    default:                    stage.next_stage = MESA_SHADER_NONE;         break;
    }
    stage.spirv.data = (const char*)spirv;
    stage.spirv.size = spirv_size;
    stage.entrypoint = opts->entrypoint ? opts->entrypoint : "main";
    stage.key.optimisations_disabled = !opts->optimise;

    /* SPIR-V to NIR */
    const struct radv_spirv_to_nir_options spirv_options = {
        .lower_view_index_to_zero = true,
        .lower_view_index_to_device_index = false,
    };

    nir_shader* nir = radv_shader_spirv_to_nir(
        &compiler_info, &stage, &spirv_options, false
    );

    if (!nir) {
        psbc_shutdown();
        return PSBC_RESULT_COMPILE_NIR;
    }

    /* Optimize NIR */
    radv_optimize_nir(nir, !opts->optimise);

    /* Gather info again — outputs_read can be out-of-date */
    nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
    radv_nir_lower_io(nir);

    /* Shader info + args + postprocess */
    radv_nir_shader_info_init(stage.stage, stage.next_stage, &stage.info);

    struct radv_shader_layout layout = {0};
    struct radv_graphics_state_key gfx_state = {0};
    gfx_state.ps.epilog.spi_shader_col_format = V_028714_SPI_SHADER_FP16_ABGR;
    gfx_state.ps.epilog.color_is_int8 = 0xff;
    gfx_state.ps.has_epilog = false;

    radv_nir_shader_info_pass(
        &compiler_info, nir, &layout, &stage.key, &gfx_state,
        RADV_PIPELINE_GRAPHICS, false, &stage.info
    );

    /* Determine previous stage for shader args declaration.
     * HS/GS need previous_stage=VERTEX so that the merged-pipeline args
     * (prolog_inputs, vertex_buffers, etc.) are declared for the NIR
     * lowering passes. The VS-specific input usage slots are filtered
     * out in buildshaderbinary for non-VS stages. */
    mesa_shader_stage previous_stage = MESA_SHADER_NONE;
    switch (mesa_stage) {
    case MESA_SHADER_TESS_CTRL: previous_stage = MESA_SHADER_VERTEX;    break;
    case MESA_SHADER_TESS_EVAL: previous_stage = MESA_SHADER_TESS_CTRL; break;
    case MESA_SHADER_GEOMETRY:  previous_stage = MESA_SHADER_VERTEX;    break;
    default:                    previous_stage = MESA_SHADER_NONE;      break;
    }

    radv_declare_shader_args(
        &compiler_info, &gfx_state, &stage, previous_stage, NULL
    );

    stage.nir = nir;
    stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
    stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;

    /* For geometry shaders, run the legacy GS lowering pass.
     * On GFX10.3 (PS5), NGG GS requires merging with the previous stage,
     * which is not possible for standalone compilation. We use the legacy
     * GS path instead, which ACO supports on all GFX levels. */
    if (mesa_stage == MESA_SHADER_GEOMETRY && !stage.info.is_ngg) {
        /* Convert emit_vertex to emit_vertex_with_counter */
        NIR_PASS(_, nir, nir_lower_gs_intrinsics,
                 nir_lower_gs_intrinsics_per_stream);
        NIR_PASS(_, nir, nir_lower_vars_to_ssa);

        ac_nir_lower_legacy_gs_options gs_options = {
            .gfx_level = gfxlevel,
            .export_clipdist_mask = 0xff,
            .write_pos_to_clipvertex = true,
            .has_param_exports = stage.info.outinfo.param_exports > 0,
            .disable_streamout = true,
        };
        nir_shader* gs_copy = NULL;
        ac_nir_legacy_gs_info gs_out_info = {0};
        NIR_PASS(_, nir, ac_nir_lower_legacy_gs, &gs_options, &gs_copy, &gs_out_info);
        /* The GS copy shader is discarded for standalone compilation.
         * The PS4 runtime generates the copy shader separately. */
        if (gs_copy)
            ralloc_free(gs_copy);
    }

    radv_postprocess_nir(
        &compiler_info, &gfx_state, &stage
    );

    /* Compile NIR to GCN ISA via ACO */
    struct radv_shader_binary* binary = radv_shader_nir_to_asm(
        &compiler_info, &stage, &nir, 1, &gfx_state
    );

    if (!binary) {
        psbc_shutdown();
        return PSBC_RESULT_COMPILE_ACO;
    }

    /* Extract code from radv_shader_binary_legacy */
    struct radv_shader_binary_legacy* legacy =
        (struct radv_shader_binary_legacy*)binary;

    /* The data layout in radv_shader_binary_legacy is:
     * [stats | code | ir | disasm | debug_info]
     * Code starts at offset stats_size. */
    const uint32_t* code = (const uint32_t*)(legacy->data + legacy->stats_size);
    uint32_t code_dw = legacy->code_size / sizeof(uint32_t);

    /* Build the PS4/PS5 shader binary */
    const BuildContext buildctx = {
        .rinfo = &binary->info,
        .rargs = &stage.args,
        .config = &binary->config,
        .nir = nir,
        .gfx_level = gfxlevel,
        .family = chipfamily,
        .stage = mesa_stage,
        .psbc_stage = opts->stage,
        .spirv_data = spirv,
        .spirv_size = spirv_size,
        .neo = neo,
    };

    uint8_t* output_data = NULL;
    size_t output_size = 0;
    PsbcResult result = buildshaderbinary(&buildctx, code, code_dw,
                                          &output_data, &output_size);
    out->data = output_data;
    out->size = output_size;

    free(binary);
    psbc_shutdown();

    return result;
}

void psbc_free_output(PsbcShaderOutput* out) {
    if (out && out->data) {
        free(out->data);
        out->data = NULL;
        out->size = 0;
    }
}

const char* psbc_result_string(PsbcResult result) {
    switch (result) {
    case PSBC_RESULT_OK:               return "success";
    case PSBC_RESULT_INVALID_SPIRV:    return "invalid SPIR-V bytecode";
    case PSBC_RESULT_UNSUPPORTED_STAGE: return "unsupported shader stage";
    case PSBC_RESULT_COMPILE_NIR:      return "SPIR-V to NIR compilation failed";
    case PSBC_RESULT_COMPILE_ACO:      return "ACO shader compilation failed";
    case PSBC_RESULT_OUT_OF_MEMORY:    return "out of memory";
    case PSBC_RESULT_INTERNAL_ERROR:   return "internal error";
    default:                           return "unknown error";
    }
}

PsbcStage psbc_stage_from_name(const char* name) {
    if (!name)
        return PSBC_STAGE_NONE;
    if (!strcmp(name, "vertex"))      return PSBC_STAGE_VERTEX;
    if (!strcmp(name, "tess-ctrl"))   return PSBC_STAGE_TESS_CTRL;
    if (!strcmp(name, "tess-eval"))   return PSBC_STAGE_TESS_EVAL;
    if (!strcmp(name, "geometry"))    return PSBC_STAGE_GEOMETRY;
    if (!strcmp(name, "fragment"))    return PSBC_STAGE_FRAGMENT;
    if (!strcmp(name, "compute"))     return PSBC_STAGE_COMPUTE;
    if (!strcmp(name, "task"))        return PSBC_STAGE_TASK;
    if (!strcmp(name, "export"))      return PSBC_STAGE_EXPORT;
    if (!strcmp(name, "local"))       return PSBC_STAGE_LOCAL;
    return PSBC_STAGE_NONE;
}
