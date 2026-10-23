# opengnm-psbc — SPIR-V to PS4 Shader Binary Compiler
#
# Builds the psbc compiler from vendored Mesa 26.2.0 sources.
# The compiler uses Mesa's NIR → ACO pipeline to translate SPIR-V
# into GCN shader binaries for the PS4 GPU (Liverpool).
#
# Build steps:
# 1. Generate Mesa codegen sources (NIR opcodes, ACO opcodes, SPIRV info)
# 2. Compile all Mesa vendored C/C++ sources into libpsbc.a
# 3. Compile CLI wrapper (cmd/psbc/main.c) and link against libpsbc.a
# 4. Link into the psbc binary

CONFIG ?= config.mak
include $(CONFIG)

PSBC ?= opengnm-psbc
LIBPSBC ?= libpsbc.a

.PHONY: all clean install generated libpsbc
.DEFAULT_GOAL := all

all: $(PSBC)

# === AMD register JSON files (for codegen) ===
# Include all GPU generations so all register fields are available
# The -rsrc.json files contain buffer/image resource constants (V_008F0C_*)
AMD_REGISTER_FILES = \
	src/amd/registers/gfx6.json \
	src/amd/registers/gfx7.json \
	src/amd/registers/gfx8.json \
	src/amd/registers/gfx81.json \
	src/amd/registers/gfx9.json \
	src/amd/registers/gfx940.json \
	src/amd/registers/gfx10.json \
	src/amd/registers/gfx10-rsrc.json \
	src/amd/registers/gfx103.json \
	src/amd/registers/gfx11.json \
	src/amd/registers/gfx11-rsrc.json \
	src/amd/registers/gfx115.json \
	src/amd/registers/gfx12.json \
	src/amd/registers/gfx12-rsrc.json \
	src/amd/registers/registers-manually-defined.json \
	src/amd/registers/pkt3.json

# === Generated sources (via Python codegen from Mesa) ===
GENERATED = \
	src/amd/common/amdgfxregs.h \
	src/amd/common/sid_tables.h \
	src/amd/compiler/aco_builder.h \
	src/amd/compiler/aco_opcodes.cpp \
	src/amd/compiler/aco_opcodes.h \
	src/compiler/nir/nir_builder_opcodes.h \
	src/compiler/nir/nir_constant_expressions.c \
	src/compiler/nir/nir_intrinsics.c \
	src/compiler/nir/nir_intrinsics.h \
	src/compiler/nir/nir_intrinsics_indices.h \
	src/compiler/nir/nir_opcodes.c \
	src/compiler/nir/nir_opcodes.h \
	src/compiler/nir/nir_opt_algebraic.c \
	src/compiler/spirv/spirv_info.c \
	src/compiler/spirv/spirv_info.h \
	src/compiler/spirv/vtn_gather_types.c \
	src/compiler/spirv/vtn_generator_ids.h

# === Codegen rules ===

# NIR opcodes
src/compiler/nir/nir_opcodes.h: src/compiler/nir/nir_opcodes_h.py src/compiler/nir/nir_opcodes.py
	$(PYTHON) $< > $@
src/compiler/nir/nir_opcodes.c: src/compiler/nir/nir_opcodes_c.py src/compiler/nir/nir_opcodes.py
	$(PYTHON) $< > $@

# NIR intrinsics
src/compiler/nir/nir_intrinsics.h: src/compiler/nir/nir_intrinsics_h.py src/compiler/nir/nir_intrinsics.py
	$(PYTHON) $< --out $@
src/compiler/nir/nir_intrinsics.c: src/compiler/nir/nir_intrinsics_c.py src/compiler/nir/nir_intrinsics.py
	$(PYTHON) $< --out $@
src/compiler/nir/nir_intrinsics_indices.h: src/compiler/nir/nir_intrinsics_indices_h.py src/compiler/nir/nir_intrinsics.py
	$(PYTHON) $< --out $@

# NIR constant expressions
src/compiler/nir/nir_constant_expressions.c: src/compiler/nir/nir_constant_expressions.py src/compiler/nir/nir_opcodes.py
	$(PYTHON) $< > $@

# NIR algebraic optimizations
src/compiler/nir/nir_opt_algebraic.c: src/compiler/nir/nir_opt_algebraic.py src/compiler/nir/nir_algebraic.py src/compiler/nir/nir_opcodes.py
	$(PYTHON) src/compiler/nir/nir_opt_algebraic.py --out $@

# NIR builder opcodes
src/compiler/nir/nir_builder_opcodes.h: src/compiler/nir/nir_builder_opcodes_h.py src/compiler/nir/nir_opcodes.py
	$(PYTHON) $< > $@

# ACO opcodes
src/amd/compiler/aco_opcodes.h: src/amd/compiler/aco_opcodes_h.py src/amd/compiler/aco_opcodes.py
	$(PYTHON) $< > $@
src/amd/compiler/aco_opcodes.cpp: src/amd/compiler/aco_opcodes_cpp.py src/amd/compiler/aco_opcodes.py
	$(PYTHON) $< > $@

# ACO builder
src/amd/compiler/aco_builder.h: src/amd/compiler/aco_builder_h.py src/amd/compiler/aco_opcodes.py
	$(PYTHON) $< > $@

# SPIRV info (generates spirv_info.c and spirv_info.h)
src/compiler/spirv/spirv_info.c src/compiler/spirv/spirv_info.h: src/compiler/spirv/spirv_info_gen.py src/compiler/spirv/spirv.core.grammar.json
	$(PYTHON) $< --out-c src/compiler/spirv/spirv_info.c --out-h src/compiler/spirv/spirv_info.h --json src/compiler/spirv/spirv.core.grammar.json

# VTN generator IDs (separate script, uses spir-v.xml not the JSON grammar)
src/compiler/spirv/vtn_generator_ids.h: src/compiler/spirv/vtn_generator_ids_h.py ../SPIRV-Headers/include/spirv/spir-v.xml
	$(PYTHON) $< ../SPIRV-Headers/include/spirv/spir-v.xml $@

# VTN gather types
src/compiler/spirv/vtn_gather_types.c: src/compiler/spirv/vtn_gather_types_c.py src/compiler/spirv/spirv.core.grammar.json
	$(PYTHON) $< src/compiler/spirv/spirv.core.grammar.json $@

# AMD register headers
src/amd/common/amdgfxregs.h: src/amd/registers/makeregheader.py $(AMD_REGISTER_FILES)
	$(PYTHON) src/amd/registers/makeregheader.py --sort address --guard AMDGPUregs $(AMD_REGISTER_FILES) > $@
	@echo "" >> $@
	@echo "/* SQ export target values — from GCN ISA spec, not in register JSONs */" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT0   0" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT1   1" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT2   2" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT3   3" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT4   4" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT5   5" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT6   6" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT7   7" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRTZ   8" >> $@
	@echo "#define V_008DFC_SQ_EXP_NULL   9" >> $@
	@echo "#define V_008DFC_SQ_EXP_POS0   12" >> $@
	@echo "#define V_008DFC_SQ_EXP_POS1   13" >> $@
	@echo "#define V_008DFC_SQ_EXP_POS2   14" >> $@
	@echo "#define V_008DFC_SQ_EXP_POS3   15" >> $@
	@echo "#define V_008DFC_SQ_EXP_POS4   16" >> $@
	@echo "#define V_008DFC_SQ_EXP_PRIM   20" >> $@
	@echo "#define V_008DFC_SQ_EXP_PARAM0 32" >> $@
	@echo "#define V_008DFC_SQ_EXP_PARAM  32" >> $@
	@echo "#define V_008DFC_SQ_EXP_MRT    0" >> $@
	@echo "#define V_008DFC_SQ_EXP_POS    12" >> $@

src/amd/common/sid_tables.h: src/amd/common/sid_tables.py src/amd/registers/pkt3.json $(AMD_REGISTER_FILES)
	$(PYTHON) $< src/amd/registers/pkt3.json $(AMD_REGISTER_FILES) > $@

# === Source files ===

# libpsbc sources — the reusable compilation library
PSBC_SRCS = \
	libpsbc/psbc_compile.c \
	cmd/psbc/crc32_sb.c \
	psbc_stubs.c

# CLI wrapper sources — thin main() that calls libpsbc
CLI_SRCS = cmd/psbc/main.c

# NIR sources — exclude nir_stub.c (conflicts with nir_print.c)
NIR_SRCS = $(filter-out \
	src/compiler/nir/nir_stub.c \
	,$(wildcard src/compiler/nir/*.c))

# Compiler common sources (shader_enums, etc.)
COMPILER_COMMON_SRCS = $(wildcard src/compiler/*.c)

# SPIRV-to-NIR sources — exclude standalone tools (they have their own main())
SPIRV_SRCS = $(filter-out \
	src/compiler/spirv/spirv2nir.c \
	src/compiler/spirv/vtn_bindgen2.c \
	,$(wildcard src/compiler/spirv/*.c))

# ACO sources — all .cpp files in src/amd/compiler/ (including instruction_selection/)
ACO_SRCS = $(wildcard src/amd/compiler/*.cpp) \
           $(wildcard src/amd/compiler/instruction_selection/*.cpp)

# AMD common sources — all .c files, excluding platform-specific, test, and standalone tool files
AMD_COMMON_SRCS = $(filter-out \
	src/amd/common/ac_linux_drm.c \
	src/amd/common/ac_rtld.c \
	src/amd/common/ac_rgp_elf_object_pack.c \
	src/amd/common/ac_gpu_info.c \
	src/amd/common/ac_video.c \
	src/amd/common/ac_surface.c \
	src/amd/common/ac_surface_meta_address_test.c \
	src/amd/common/ac_surface_modifier_test.c \
	src/amd/common/ac_vcn_dec.c \
	src/amd/common/amdgpu_devices.c \
	src/amd/common/ac_ib_parser.c \
	,$(wildcard src/amd/common/*.c))

# AMD common NIR passes
AMD_COMMON_NIR_SRCS = $(wildcard src/amd/common/nir/*.c)

# RADV shader-related sources (only shader compilation, not full driver)
# Exclude radv_pipeline.c and radv_pipeline_graphics.c which need full driver infra
RADV_SRCS = \
	src/amd/vulkan/radv_postprocess_nir_standalone.c \
	src/amd/vulkan/radv_shader.c \
	src/amd/vulkan/radv_shader_args.c \
	src/amd/vulkan/radv_shader_info.c

# RADV NIR passes — exclude ray tracing files (PS4 doesn't support RT)
RADV_NIR_SRCS = $(filter-out \
	src/amd/vulkan/nir/radv_nir_lower_ray_queries.c \
	src/amd/vulkan/nir/radv_nir_rt_common.c \
	src/amd/vulkan/nir/radv_nir_rt_stage_common.c \
	src/amd/vulkan/nir/radv_nir_rt_stage_cps.c \
	src/amd/vulkan/nir/radv_nir_rt_stage_functions.c \
	src/amd/vulkan/nir/radv_nir_rt_stage_monolithic.c \
	src/amd/vulkan/nir/radv_nir_rt_traversal_shader.c \
	,$(wildcard src/amd/vulkan/nir/*.c))

# Mesa util sources — exclude platform-specific, profiling, and generated-header files
UTIL_SRCS = $(filter-out \
	src/util/cache_ops_aarch64.c \
	src/util/cache_ops_null.c \
	src/util/cache_ops_x86.c \
	src/util/cache_ops_x86_clflushopt.c \
	src/util/u_debug_stack.c \
	src/util/u_sync_provider.c \
	src/util/os_socket.c \
	src/util/xmlconfig.c \
	src/util/perf/u_perfetto.cc \
	src/util/perf/u_gpuvis.c \
	src/util/perf/u_sysprof.c \
	,$(wildcard src/util/*.c) $(wildcard src/util/*.cpp))

# Mesa util format sources
UTIL_FORMAT_SRCS = $(wildcard src/util/format/*.c)

# Blake3 hashing — need dispatch + portable, plus neon on ARM64 hosts
BLAKE3_SRCS = \
	src/util/blake3/blake3.c \
	src/util/blake3/blake3_dispatch.c \
	src/util/blake3/blake3_portable.c

# blake3_neon.c needs ARM intrinsics, don't build it on x86
BLAKE3_HOST_ARCH ?= $(shell uname -m)
ifneq ($(filter aarch64 arm64 armv8%,$(BLAKE3_HOST_ARCH)),)
BLAKE3_SRCS += src/util/blake3/blake3_neon.c
endif

# C11 threads compat (POSIX implementation)
C11_THREADS_SRCS = src/c11/impl/threads_posix.c src/c11/impl/time.c

# Vulkan runtime/util sources needed for SPIRV-to-NIR and shader compilation
# Most of the Vulkan runtime is too deeply intertwined with the full driver.
# We provide stubs for the functions radv_shader.c calls.
VK_RUNTIME_SRCS = \
	src/vulkan/runtime/vk_nir_lower_descriptor_heaps.c \
	src/vulkan/runtime/vk_nir_convert_ycbcr.c

# === Object files ===
PSBC_OBJS = $(PSBC_SRCS:.c=.o)
CLI_OBJS = $(CLI_SRCS:.c=.o)
NIR_OBJS = $(NIR_SRCS:.c=.o)
SPIRV_OBJS = $(SPIRV_SRCS:.c=.o)
ACO_OBJS = $(ACO_SRCS:.cpp=.cpp.o)
AMD_COMMON_OBJS = $(AMD_COMMON_SRCS:.c=.o)
AMD_COMMON_NIR_OBJS = $(AMD_COMMON_NIR_SRCS:.c=.o)
RADV_OBJS = $(RADV_SRCS:.c=.o)
RADV_NIR_OBJS = $(RADV_NIR_SRCS:.c=.o)
UTIL_OBJS = $(patsubst %.c,%.o,$(filter %.c,$(UTIL_SRCS))) $(patsubst %.cpp,%.cpp.o,$(filter %.cpp,$(UTIL_SRCS)))
UTIL_FORMAT_OBJS = $(UTIL_FORMAT_SRCS:.c=.o)
COMPILER_COMMON_OBJS = $(COMPILER_COMMON_SRCS:.c=.o)
BLAKE3_OBJS = $(BLAKE3_SRCS:.c=.o)
C11_THREADS_OBJS = $(C11_THREADS_SRCS:.c=.o)
VK_RUNTIME_OBJS = $(VK_RUNTIME_SRCS:.c=.o)

# All Mesa + psbc objects go into libpsbc.a
LIBPSBC_OBJS = \
	$(PSBC_OBJS) \
	$(NIR_OBJS) \
	$(SPIRV_OBJS) \
	$(ACO_OBJS) \
	$(AMD_COMMON_OBJS) \
	$(AMD_COMMON_NIR_OBJS) \
	$(RADV_OBJS) \
	$(RADV_NIR_OBJS) \
	$(UTIL_OBJS) \
	$(UTIL_FORMAT_OBJS) \
	$(COMPILER_COMMON_OBJS) \
	$(BLAKE3_OBJS) \
	$(C11_THREADS_OBJS) \
	$(VK_RUNTIME_OBJS)

# === Compile rules ===
# Use order-only prerequisites (| $(GENERATED)) to ensure all codegen files
# are built before any compilation, without causing rebuilds when they change.
%.o: %.c | $(GENERATED)
	$(CC) $(CFLAGS) -c $< -o $@

%.cpp.o: %.cpp | $(GENERATED)
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifndef OPENGNM_PSBC_ORBIS
# === Static library (libpsbc.a) ===
# Contains all Mesa objects + psbc compilation logic. Can be linked
# into both the CLI and the vulkan-ps4 ICD.
libpsbc: $(LIBPSBC)
$(LIBPSBC): $(LIBPSBC_OBJS)
	$(AR) rcs $@ $^

# === Link rule ===
# CLI links against libpsbc.a
$(PSBC): $(CLI_OBJS) $(LIBPSBC)
	$(LD) $(CXXFLAGS) -o $@ $(CLI_OBJS) $(LIBPSBC) $(LDFLAGS)
endif

# === Targets ===
generated: $(GENERATED)

ifndef OPENGNM_PSBC_ORBIS
clean:
	find . -name '*.o' -delete
	find . -name '*.cpp.o' -delete
	rm -f $(PSBC) $(LIBPSBC)
	rm -f $(GENERATED)

install: all
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 $(PSBC) $(DESTDIR)$(BINDIR)/
endif
