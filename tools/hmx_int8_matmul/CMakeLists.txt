cmake_minimum_required(VERSION 3.14.3)
project(hmx_int8_matmul C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# The fixed-shape ABI bakes these three dimensions into both ARM and Hexagon
# objects. They remain CMake cache values so a new shape only needs a rebuild.
set(HMX_FIXED_M "73" CACHE STRING "Compile-time GEMM M dimension")
set(HMX_FIXED_K "65" CACHE STRING "Compile-time GEMM K dimension")
set(HMX_FIXED_N "47" CACHE STRING "Compile-time GEMM N dimension")
set(HMX_FIXED_HEADS "1" CACHE STRING "Compile-time QK attention head count")
set(HMX_OPERATOR_OUTPUT_NAME "" CACHE STRING
    "Optional fixed-scale QK operator library name, without lib/.so")
set(HMX_OPERATOR_Q_SCALE "" CACHE STRING
    "Compile-time Q tensor scale for the fixed QK operator")
set(HMX_OPERATOR_K_SCALE "" CACHE STRING
    "Compile-time K tensor scale for the fixed QK operator")
set(HMX_OPERATOR_OUTPUT_SCALE "" CACHE STRING
    "Compile-time INT8 output tensor scale for the fixed QK operator")
set(HMX_RPC_SKEL_OUTPUT_NAME "hmx_int8_rpc_skel" CACHE STRING
    "FastRPC skel library name, without lib/.so")
set(HMX_HVX_LANES "4" CACHE STRING
    "Maximum HVX lanes including the FastRPC handler")
set(HMX_HVX_EPILOGUE_LANES "1" CACHE STRING
    "HVX lanes used by the HMX accumulator epilogue (1 or 4)")
set(HMX_HVX_EPILOGUE_BATCH_TILES "8" CACHE STRING
    "Accumulator tiles in one coarse HVX epilogue dispatch")
set(HMX_LONG_RPC_READY_GROUP_HEADS "1" CACHE STRING
    "Heads published by one long-RPC output flush/ready event")
foreach(fixed_dimension HMX_FIXED_M HMX_FIXED_K HMX_FIXED_N HMX_FIXED_HEADS)
    if(NOT "${${fixed_dimension}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${fixed_dimension} must be a positive integer")
    endif()
endforeach()
if(NOT "${HMX_HVX_LANES}" MATCHES "^[1-4]$")
    message(FATAL_ERROR "HMX_HVX_LANES must be between 1 and 4")
endif()
if(NOT "${HMX_HVX_EPILOGUE_LANES}" MATCHES "^(1|4)$")
    message(FATAL_ERROR "HMX_HVX_EPILOGUE_LANES must be 1 or 4")
endif()
if(NOT "${HMX_HVX_EPILOGUE_BATCH_TILES}" MATCHES "^(1|2|4|8|16)$")
    message(FATAL_ERROR
        "HMX_HVX_EPILOGUE_BATCH_TILES must be 1, 2, 4, 8, or 16")
endif()
if(NOT "${HMX_LONG_RPC_READY_GROUP_HEADS}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
        "HMX_LONG_RPC_READY_GROUP_HEADS must be a positive integer")
endif()
if(HMX_LONG_RPC_READY_GROUP_HEADS GREATER HMX_FIXED_HEADS)
    message(FATAL_ERROR
        "HMX_LONG_RPC_READY_GROUP_HEADS must not exceed HMX_FIXED_HEADS")
endif()
math(EXPR HMX_FIXED_M_PADDED "((${HMX_FIXED_M} + 63) / 64) * 64")
math(EXPR HMX_FIXED_K_PADDED "((${HMX_FIXED_K} + 31) / 32) * 32")
math(EXPR HMX_FIXED_N_PADDED "((${HMX_FIXED_N} + 31) / 32) * 32")
if(HMX_FIXED_K_PADDED GREATER 65536)
    message(FATAL_ERROR "Padded HMX_FIXED_K must not exceed 65536")
endif()
set(hmx_fixed_shape_definitions
    HMX_FIXED_M=${HMX_FIXED_M}
    HMX_FIXED_K=${HMX_FIXED_K}
    HMX_FIXED_N=${HMX_FIXED_N}
    HMX_FIXED_HEADS=${HMX_FIXED_HEADS}
    HMX_FIXED_M_PADDED=${HMX_FIXED_M_PADDED}
    HMX_FIXED_K_PADDED=${HMX_FIXED_K_PADDED}
    HMX_FIXED_N_PADDED=${HMX_FIXED_N_PADDED}
)
message(STATUS
    "HMX fixed shape: H=${HMX_FIXED_HEADS}, "
    "${HMX_FIXED_M}x${HMX_FIXED_K}x${HMX_FIXED_N} "
    "(padded ${HMX_FIXED_M_PADDED}x${HMX_FIXED_K_PADDED}x${HMX_FIXED_N_PADDED})")

if(NOT "${HMX_RPC_SKEL_OUTPUT_NAME}" MATCHES "^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    message(FATAL_ERROR
        "HMX_RPC_SKEL_OUTPUT_NAME must be a filename-safe basename")
endif()

if(NOT "${HMX_OPERATOR_OUTPUT_NAME}" STREQUAL "")
    if(NOT "${HMX_OPERATOR_OUTPUT_NAME}" MATCHES "^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
        message(FATAL_ERROR
            "HMX_OPERATOR_OUTPUT_NAME must be a filename-safe basename")
    endif()
    foreach(scale_name
            HMX_OPERATOR_Q_SCALE
            HMX_OPERATOR_K_SCALE
            HMX_OPERATOR_OUTPUT_SCALE)
        if(NOT "${${scale_name}}" MATCHES
                "^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$")
            message(FATAL_ERROR "${scale_name} must be a positive decimal")
        endif()
        if("${${scale_name}}" MATCHES
                "^0+([.]0+)?([eE][-+]?[0-9]+)?$")
            message(FATAL_ERROR "${scale_name} must be greater than zero")
        endif()
    endforeach()
    message(STATUS
        "HMX fixed QK operator: ${HMX_OPERATOR_OUTPUT_NAME}, "
        "Q scale=${HMX_OPERATOR_Q_SCALE}, K scale=${HMX_OPERATOR_K_SCALE}, "
        "output scale=${HMX_OPERATOR_OUTPUT_SCALE}, "
        "DSP skel=${HMX_RPC_SKEL_OUTPUT_NAME}")
    set(hmx_operator_scale_definitions
        HMX_OPERATOR_Q_SCALE=${HMX_OPERATOR_Q_SCALE}
        HMX_OPERATOR_K_SCALE=${HMX_OPERATOR_K_SCALE}
        HMX_OPERATOR_OUTPUT_SCALE=${HMX_OPERATOR_OUTPUT_SCALE}
    )
endif()

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()
set(ignoreVar ${DEPENDENCIES}${V})

# Hexagon SDK releases do not all provide this helper.
if(NOT COMMAND set_common_compile_and_link_options)
    function(set_common_compile_and_link_options target)
        if(COMMAND link_options)
            link_options(${target})
        endif()
    endfunction()
endif()

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/rtos/qurt
    ${HEXAGON_SDK_ROOT}/utils/examples
)
include_directories(${common_incs})

if(${OS_TYPE} MATCHES "HLOS")
    # Hexagon SDK 6.5's helper compares fastrpc_domain_type with an int.
    # Keep -Werror for project sources while allowing that SDK-owned file to
    # build with the NDK clang shipped by the same SDK.
    set_source_files_properties(
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        PROPERTIES COMPILE_OPTIONS -Wno-sign-compare
    )

    # ARM-side library. Applications dlopen this one library; it opens the
    # Hexagon skel below through FastRPC.
    add_library(hmx_int8_matmul SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/hmx_int8_rpc_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/hmx_i8_gemm.c
    )
    build_idl(include/hmx_int8_rpc.idl hmx_int8_matmul)

    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(hmx_int8_matmul ${dsprpc})
    target_link_libraries(hmx_int8_matmul m)
    target_compile_definitions(hmx_int8_matmul PRIVATE ${hmx_fixed_shape_definitions})
    target_compile_options(hmx_int8_matmul PRIVATE -Wall -Wextra -Werror)
    set_target_properties(hmx_int8_matmul PROPERTIES OUTPUT_NAME hmx_int8_matmul)

    # Optional paper-style QK estimator variant. Shape and all three tensor
    # scales are part of this library's identity and ABI. The public call has
    # no runtime dimensions or scale arguments.
    if(NOT "${HMX_OPERATOR_OUTPUT_NAME}" STREQUAL "")
        add_library(hmx_qk_i8_operator SHARED
            ${CMAKE_CURRENT_BINARY_DIR}/hmx_int8_rpc_stub.c
            ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/host/hmx_i8_gemm.c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/host/hmx_qk_i8_operator.c
        )
        add_dependencies(hmx_qk_i8_operator hmx_int8_matmul)
        link_custom_library(hmx_qk_i8_operator ${dsprpc})
        target_compile_definitions(hmx_qk_i8_operator PRIVATE
            ${hmx_fixed_shape_definitions}
            ${hmx_operator_scale_definitions}
            HMX_I8_FIXED_ONLY=1
            HMX_QK_OPERATOR_BUILD=1
            HMX_RPC_SKEL_FILENAME="lib${HMX_RPC_SKEL_OUTPUT_NAME}.so"
        )
        target_compile_options(hmx_qk_i8_operator PRIVATE
            -Wall -Wextra -Werror
            -ffunction-sections -fdata-sections
        )
        target_link_options(hmx_qk_i8_operator PRIVATE
            -Wl,--gc-sections
            -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/host/hmx_qk_i8_operator.exports.map
        )
        set_target_properties(hmx_qk_i8_operator PROPERTIES
            OUTPUT_NAME ${HMX_OPERATOR_OUTPUT_NAME}
            LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/host/hmx_qk_i8_operator.exports.map
        )
    endif()

    # Generic loader for any fixed-scale operator variant. It obtains the
    # compiled shape/scales from metadata before allocating test matrices.
    add_executable(hmx_qk_i8_operator_dlopen_example
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/dlopen_operator_example.c
    )
    set_common_compile_and_link_options(hmx_qk_i8_operator_dlopen_example)
    target_link_libraries(hmx_qk_i8_operator_dlopen_example dl m)
    target_compile_options(
        hmx_qk_i8_operator_dlopen_example PRIVATE -Wall -Wextra -Werror)

    copy_binaries(
        hmx_qk_i8_operator_dlopen_example
        hmx_int8_matmul
    )
    if(TARGET hmx_qk_i8_operator)
        copy_binaries(hmx_qk_i8_operator)
    endif()
else()
    # Hexagon-side FastRPC skel. The GEMM source contains the raw HMX
    # instructions; it does not link HexKL/SDKL matmul implementations.
    add_library(hmx_int8_rpc_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/hmx_int8_rpc_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hvx_worker_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_i8_kernel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/runtime.c
    )
    build_idl(include/hmx_int8_rpc.idl hmx_int8_rpc_skel)

    # QAIC 8.7 emits int-vs-size_t loops in its optional IDL version check.
    # Keep -Werror for authored sources and silence only that generated file.
    set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/hmx_int8_rpc_skel.c
        PROPERTIES COMPILE_OPTIONS -Wno-sign-compare
    )

    target_compile_options(hmx_int8_rpc_skel PRIVATE
        -Wall -Wextra -Werror
        -mhmx -mhvx -mhvx-length=128B
    )
    target_compile_definitions(hmx_int8_rpc_skel PRIVATE
        ${hmx_fixed_shape_definitions}
        ${hmx_operator_scale_definitions}
        HMX_HVX_COMPILED_LANES=${HMX_HVX_LANES}
        HMX_HVX_EPILOGUE_LANES=${HMX_HVX_EPILOGUE_LANES}
        HMX_HVX_EPILOGUE_BATCH_TILES=${HMX_HVX_EPILOGUE_BATCH_TILES}
        HMX_LONG_RPC_READY_GROUP_HEADS=${HMX_LONG_RPC_READY_GROUP_HEADS}
    )
    set_target_properties(hmx_int8_rpc_skel PROPERTIES
        OUTPUT_NAME ${HMX_RPC_SKEL_OUTPUT_NAME})
    copy_binaries(hmx_int8_rpc_skel)
endif()
