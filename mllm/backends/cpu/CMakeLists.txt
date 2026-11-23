file(GLOB_RECURSE MLLM_CPU_BACKEND_OPS_SRC ${CMAKE_CURRENT_LIST_DIR}/ops/*.cpp)
file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_COMMON_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/common/*.cpp)

set(kleidiai_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/vendors/kleidiai)
set(
  MLLM_KAI_W4A32_PACK_SOURCE
  ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c)

if(MLLM_BUILD_ARM_BACKEND)
  file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/*.cpp)
  if(ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
      AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19")
    set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/SparsePVSme.cpp
                 APPEND PROPERTY COMPILE_OPTIONS -march=armv9.2-a+sme)
  endif()

  # We treat kleidiai as a external lib.
  set(KLEIDIAI_BUILD_TESTS OFF)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/vendors/kleidiai EXCLUDE_FROM_ALL)

  set(MLLM_KAI_INCLUDE_DIRS
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/>
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/>
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/>
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/>
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/>
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/kai/ukernels/matmul/pack>
    # f32_qsi8d32p_qai4c32p
    $<BUILD_INTERFACE:${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p>
    $<INSTALL_INTERFACE:include/mllm/backends/arm/vendors/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p>
  )

  set(MLLM_KAI_SOURCES
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_asm.S
    # for f32_qai8dxp_qsi4c32
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.c
    ${MLLM_KAI_W4A32_PACK_SOURCE}
    # for qsi8d32p_qai4c32p
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.c
    # SME Related
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c
  )

  if(MLLM_CPU_BACKEND_USE_SME2)
    set(MLLM_KAI_SOURCES_SME2
      # f32_qsi8d32p_qai4c32p
      ${kleidiai_SOURCE_DIR}/kai/kai_common_sme_asm.S
      ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa_asm.S
      ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.c
      ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot_asm.S
      ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.c
    )
  else()
    set(MLLM_KAI_SOURCES_SME2 "")
  endif()

  add_library(
    MllmCPUBackend SHARED
    ${MLLM_CPU_BACKEND_OPS_SRC}
    ${MLLM_CPU_BACKEND_KERNELS_SRC}
    ${MLLM_CPU_BACKEND_KERNELS_COMMON_SRC}
    ${MLLM_KAI_SOURCES}
    ${MLLM_KAI_SOURCES_SME2}
    CPUAllocator.cpp
    CPUBackend.cpp
    CPUDispatcher.cpp)
  target_include_directories(MllmCPUBackend PUBLIC ${MLLM_KAI_INCLUDE_DIRS})
else()
  # X86 Backend use highway kernels
  CPMAddPackage(
    NAME highway
    GITHUB_REPOSITORY google/highway
    GIT_TAG 1.3.0
  )
  file(GLOB_RECURSE MLLM_CPU_BACKEND_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/kernels/x86/*.cpp)
  add_library(
    MllmCPUBackend SHARED
    ${MLLM_CPU_BACKEND_OPS_SRC}
    ${MLLM_CPU_BACKEND_KERNELS_SRC}
    ${MLLM_CPU_BACKEND_KERNELS_COMMON_SRC}
    ${MLLM_KAI_W4A32_PACK_SOURCE}
    CPUAllocator.cpp
    CPUBackend.cpp
    CPUDispatcher.cpp)
  target_include_directories(
    MllmCPUBackend
    PRIVATE
      ${kleidiai_SOURCE_DIR}
      ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack)
  target_link_libraries(MllmCPUBackend PUBLIC hwy)
endif()

target_link_libraries(MllmCPUBackend PUBLIC MllmRT)
target_compile_options(MllmCPUBackend PUBLIC ${MLLM_CPU_BACKEND_COMPILE_OPTIONS})

# ONLY APPLE CAN DO !
# Processing OpenMP
if(MLLM_KERNEL_USE_THREADS AND MLLM_KERNEL_THREADS_VENDOR_OPENMP)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR APPLE)
    target_link_libraries(MllmRT PUBLIC ${OpenMP_omp_LIBRARY})
    target_compile_options(MllmRT PUBLIC ${OpenMP_CXX_FLAGS})
    # Apple should not use OpenMP
    message(ERROR "Apple platform should not use OpenMP. Pls set MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD=ON")
  else()
    target_link_libraries(MllmRT PUBLIC ${OpenMP_CXX_FLAGS})
    target_compile_options(MllmRT PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(MllmRT PUBLIC ${OpenMP_CXX_INCLUDE_DIR})

    if(MLLM_BUILD_ARM_BACKEND)
      # OpenMP is a translation-unit concern: compile only sources that own a
      # parallel region, including operators that instantiate one from a
      # header-only kernel. Keep this list grouped by ownership so adding a
      # parallel region does not silently broaden the whole backend target.
      set(MLLM_CPU_BACKEND_OPENMP_SOURCES
        # Backend operators with an in-file parallel region.
        ${CMAKE_CURRENT_LIST_DIR}/ops/AvgPool1dOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/Conv1DOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/EmbeddingOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/GroupedQueryAttentionOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/LayerNormOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/RMSNormOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/SoftmaxOp.cpp

        # Operators that instantiate parallel header-only attention kernels.
        ${CMAKE_CURRENT_LIST_DIR}/ops/FlashAttention2Op.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/FlashAttn2WithSinkAndSwaOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/PagedAttnOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/RadixAttnDiffDimOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/RadixAttnOp.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ops/RadixAttnWithSinkAndSwaDiffDimOp.cpp

        # Kernel translation units with an in-file parallel region.
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/cast_types.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/elementwise.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/gelu.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/linear/kai.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/mllm_blas/mllm_blas_sgemm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/relu.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/sigmoid.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/arm/silu.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/common/gdn/gated_delta_net.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernels/common/ggml/matmul.cpp)

      set_property(
        SOURCE ${MLLM_CPU_BACKEND_OPENMP_SOURCES}
        APPEND PROPERTY COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
      target_link_libraries(MllmCPUBackend PUBLIC ${OpenMP_CXX_FLAGS})
      target_include_directories(MllmCPUBackend PUBLIC ${OpenMP_CXX_INCLUDE_DIR})
    endif()
  endif()
endif()

if(MLLM_USE_BLAS)
  include(BLAS.cmake)
  mllm_configure_blas()
  if(MLLM_BLAS_LIBRARIES)
    target_link_libraries(MllmCPUBackend PUBLIC ${MLLM_BLAS_LIBRARIES})
  endif()
  if(MLLM_BLAS_COMPILE_DEFINITIONS)
    target_compile_definitions(MllmCPUBackend PUBLIC ${MLLM_BLAS_COMPILE_DEFINITIONS})
  endif()
  if(MLLM_BLAS_INCLUDE_DIRS)
    target_include_directories(MllmCPUBackend PUBLIC ${MLLM_BLAS_INCLUDE_DIRS})
  endif()
endif()

install(
  TARGETS MllmCPUBackend
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

if(MLLM_BUILD_ARM_BACKEND)
  install(
    DIRECTORY ${kleidiai_SOURCE_DIR}
    DESTINATION include/mllm/backends/arm/vendors
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp")
else()
  install(
    TARGETS hwy
    EXPORT MllmTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
endif()
