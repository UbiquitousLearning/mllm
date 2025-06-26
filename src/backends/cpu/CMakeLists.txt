# CPU Backend
file(GLOB MLLM_CPU_SRC
        ${CMAKE_CURRENT_LIST_DIR}/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/compute/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/third_party/ggml/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/function/*.cpp
)


# --- 初始化 kleidiai 源文件和头文件目录 ---
set(ALL_KLEIDIAI_SOURCES "")
set(ALL_KLEIDIAI_INCLUDE_DIRS "")


# Conditionally compile kleidiai for ARM architectures
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    # --- 设置 kleidiai 库的路径 ---
    set(KLEIDIAI_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/kleidiai)
    if(NOT EXISTS ${KLEIDIAI_SOURCE_DIR})
        message(FATAL_ERROR "kleidiai library not found! Please place it in 'third_party/kleidiai'.")
    endif()
    list(APPEND ALL_KLEIDIAI_INCLUDE_DIRS ${KLEIDIAI_SOURCE_DIR})


    # --- Section for QSI4_C32P (FP32 output) kernels ---
    message(STATUS "Enabling kleidiai QSI4_C32 MatMul implementation.")
    list(APPEND ALL_KLEIDIAI_INCLUDE_DIRS
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p
    )
    set(KLEIDIAI_SOURCES_QSI4_C32P
            ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
            ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S
            ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c
            ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm_asm.S
    )
    list(APPEND ALL_KLEIDIAI_SOURCES ${KLEIDIAI_SOURCES_QSI4_C32P})

    # --- Section for QSI4_CXP (FP16 output) kernels ---
    message(STATUS "Enabling kleidiai QSI4_CXP (to FP16) MatMul implementation.")
    list(APPEND ALL_KLEIDIAI_INCLUDE_DIRS
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp
    )
    set(KLEIDIAI_SOURCES_QSI4_CXP
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod_asm.S
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_16x4_neon_i8mm_asm.S
    )
    list(APPEND ALL_KLEIDIAI_SOURCES ${KLEIDIAI_SOURCES_QSI4_CXP})

    # --- Section for FP16 (f16*f16) kernels ---
    message(STATUS "Enabling kleidiai FP16 MatMul implementation.")
    list(APPEND ALL_KLEIDIAI_INCLUDE_DIRS
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p
    )
    set(KLEIDIAI_SOURCES_FP16
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    )
    list(APPEND ALL_KLEIDIAI_SOURCES ${KLEIDIAI_SOURCES_FP16})

    # --- Section for FP32 (f32*f32) kernels ---
    message(STATUS "Enabling kleidiai FP32 MatMul implementation.")
    list(APPEND ALL_KLEIDIAI_INCLUDE_DIRS
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p
    )
    set(KLEIDIAI_SOURCES_FP32
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_asm.S
    )
    list(APPEND ALL_KLEIDIAI_SOURCES ${KLEIDIAI_SOURCES_FP32})

    # --- Section for all packing functions ---
    list(APPEND ALL_KLEIDIAI_INCLUDE_DIRS
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack
    )
    set(KLEIDIAI_PACK_SOURCES
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
        ${KLEIDIAI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    )
    list(APPEND ALL_KLEIDIAI_SOURCES ${KLEIDIAI_PACK_SOURCES})

    # 将所有 kleidiai 源文件添加到主源文件列表
    list(APPEND MLLM_CPU_SRC ${ALL_KLEIDIAI_SOURCES})
endif() # End ARM check for kleidiai sources


if (MLLM_OPENMP)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found openmp")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "openmp not found!")
endif()
endif()



# 架构优化标志
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    add_compile_options(-march=armv8.2-a+dotprod+fp16+fp16fml)

    # 特殊处理：为FP16源文件添加格式标志
    if(KLAI_USE_FP16 AND CMAKE_CROSS_COMPILING)
        message(STATUS "Cross-compilation for ARM detected. Applying FP16 compiler flags.")
        set_source_files_properties(${KLEIDIAI_FP16_SOURCES} ${KLEIDIAI_SOURCES_QSI4_CXP}
            PROPERTIES COMPILE_FLAGS "-mfp16-format=ieee"
        )
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86_64 detected")
    add_compile_options(-mavx2)
    add_compile_options(-march=native)
endif()

if(${MLLM_ENABLE_PYTHON})
add_library(
    mllm_cpu
    SHARED
    ${MLLM_CPU_SRC}
)
else()
add_library(
        mllm_cpu
        OBJECT
        ${MLLM_CPU_SRC}
)
endif()

# Conditionally add kleidiai compile definitions for ARM
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    target_compile_definitions(mllm_cpu PUBLIC KLAI_USE_QSI4_C32)
    target_compile_definitions(mllm_cpu PUBLIC KLAI_USE_QSI4_CXP)
    target_compile_definitions(mllm_cpu PUBLIC KLAI_USE_FP16)
    target_compile_definitions(mllm_cpu PUBLIC KLAI_USE_FP32)
endif()

# 包含头文件目录
list(REMOVE_DUPLICATES ALL_KLEIDIAI_INCLUDE_DIRS)
target_include_directories(
        mllm_cpu
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${ALL_KLEIDIAI_INCLUDE_DIRS}
)

if(OpenMP_FOUND)
    message(STATUS "found openmp")
    if(ARM AND NOT APK)
        message(STATUS "[ARM]  found openmp")
        target_compile_options(mllm_cpu PUBLIC -fopenmp)
        # tmac or for mac
        if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_CROSSCOMPILING)
            target_link_libraries(mllm_cpu PUBLIC OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(mllm_cpu PUBLIC -fopenmp -static-openmp)
        endif ()
    else()
    target_link_libraries(mllm_cpu
            PUBLIC
            OpenMP::OpenMP_CXX
        )
    endif()
endif()

target_link_libraries(mllm_cpu PUBLIC fmt::fmt-header-only)

set_target_properties(mllm_cpu PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
