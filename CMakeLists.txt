cmake_minimum_required(VERSION 3.10)

project(mllm)

# 添加编译选项来禁用所有警告
# if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     add_compile_options(-w)
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#     add_compile_options(/w)
# endif()

cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)
option(ARM "build on ARM" OFF)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

option(MLLM_ENABLE_PYTHON "build mllm pybinding" OFF)

if(MSVC)
    message(STATUS "Using MSVC as the compiler")
else()
    add_compile_options(-Wno-deprecated-declarations)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()
add_compile_options(-Wno-gnu-string-literal-operator-template)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    set(ARM ON)
    set(ANDROID_PLATFORM android-28)
endif ()

if (ARM)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin-arm)
    add_compile_definitions(__ARM_FEATURE_DOTPROD)
    # 检查是否使用的是 GCC 或 Clang 编译器
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod")
    endif()
else ()
    if (MSVC)
        # 设置多配置生成器的输出路径
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../bin)
        # 如果使用多配置生成器，还需要为每个配置单独设置输出路径
        foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_BINARY_DIR}/../bin/)
        endforeach()
    else()
        # 设置单配置生成器的输出路径
        set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
    endif()
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    option(TEST "test mode" OFF)
else ()
    option(TEST "test mode" ON)
endif ()
option(QUANT "quantize tools" ON)
option(APK "Build for Android APK Lib." OFF)
option(FROM_GGUF "convert from gguf" OFF)

option(DEBUG "debug print" OFF)
if(DEBUG)
    add_definitions(-DDEBUGPRINT)
endif()

# backend options
option(QNN "Enable QNN" OFF)
option(QNN_OLD_FRONTEND "Enable Old QNN" OFF)
if(QNN)
    add_definitions(-DUSE_QNN) # the USE_QNN should come before cpu subdirectory
endif()
option(QNN_VALIDATE_NODE "Enable QNN Validate Node When Building Graph" ON)
if(QNN_VALIDATE_NODE)
    add_definitions(-DQNN_VALIDATE_NODE)
endif()

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

# for XNNPACK, avoid invovle googltest twice.
set(GOOGLETEST_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/googletest)
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
add_subdirectory(third_party/fmt EXCLUDE_FROM_ALL)

option(MLLM_OPENMP "openmp" ON)
option(MLLM_OPENMP_STATIC "openmp static" OFF)

if (CMAKE_HOST_UNIX)
    message(STATUS "current platform: Linux ")
elseif (CMAKE_HOST_WIN32)
    message(STATUS "current platform: Windows ")
else ()
    message(STATUS "current platform: unknown ")
endif ()
if (ARM AND NOT APK)
    set(MLLM_OPENMP_STATIC ON)
endif ()
# turn off openmp when build on mac or for mac
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_CROSSCOMPILING)
    message(STATUS "mac detected, turn off openmp")
    set(MLLM_OPENMP OFF)
    set(MLLM_OPENMP_STATIC OFF)
endif ()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86_64 detected")
    add_compile_options(-mf16c)
    add_compile_options(-mavx2)
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    add_definitions(-DARM)

    if(APK)
        message(STATUS "Build for Android APK Lib.")
        add_definitions("-DANDROID_API")
    endif()
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRC)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/express DIR_SRC_EXP)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/processor DIR_SRC_PROCESSOE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/memory DIR_SRC_MEM_MANAGER)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test TEST_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/third_party/wenet_audio DIR_THIRDPARTY_AUDIO)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_SOURCE_DIR}/third_party/fmt/include)

# NOTE: The include below is just for clang to get the include path
# You can remove those lines if you just want to build mllm instead dev on it.
include_directories(${PROJECT_SOURCE_DIR}/third_party/pybind11/include)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/cpu)

if(QNN) # QNN lib
    include_directories(
        # $ENV{QNN_SDK_ROOT}/include/QNN # QNN SDK include
        ${PROJECT_SOURCE_DIR}/src/backends/qnn/sdk/include/QNN # QNN SDK include
        ${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn
        ${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn/Log
        ${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn/PAL/include
        ${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn/Model
        ${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn/Utils
        ${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn/WrapperUtils
    )   
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/qnn)
endif()

option(MLLM_BUILD_XNNPACK_BACKEND "Build mllm's XNNPACK backend" OFF)

if(MLLM_BUILD_XNNPACK_BACKEND)
    if(NOT WIN32)
        add_compile_options(-fPIC)
    else()
        # -fPIC is not a windows flag
        set(CMAKE_POSITION_INDEPENDENT_CODE FALSE)
    endif()

    set(XNNPACK_BUILD_TESTS OFF)
    set(XNNPACK_BUILD_BENCHMARKS OFF)
    add_definitions(-DMLLM_BUILD_XNNPACK_BACKEND=1)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/xnnpack)
endif()

# add tokenizers
file(GLOB_RECURSE SRC_TOKENIZERS
    ${PROJECT_SOURCE_DIR}/src/tokenizers/*.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/*.hpp
)

# if compile to x86_64
if(QUANT)
    include_directories(${PROJECT_SOURCE_DIR}/src/quantizer)
    file(GLOB_RECURSE MLLM_QUANT
        ${PROJECT_SOURCE_DIR}/src/backends/cpu/compute/GEMM_AArch64.hpp
        ${PROJECT_SOURCE_DIR}/src/backends/cpu/compute/GEMM_AArch64.cpp
        ${PROJECT_SOURCE_DIR}/src/backends/cpu/quantize/*.hpp
        ${PROJECT_SOURCE_DIR}/src/backends/cpu/quantize/*.cpp
    )

    file(GLOB_RECURSE MLLM_QUANTIZER
        ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/*.hpp)
    list(REMOVE_ITEM MLLM_QUANTIZER ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/main.cpp)

    add_executable(
        quantize
        ${PROJECT_SOURCE_DIR}/src/quantizer/main.cpp
        ${MLLM_QUANT}
        ${MLLM_QUANTIZER}

        # ${DIR_SRC}
        ${PROJECT_SOURCE_DIR}/src/ParamLoader.cpp
    )
    target_link_libraries(quantize fmt::fmt-header-only)

    if(FROM_GGUF)
        add_executable(
            from_gguf
            ${PROJECT_SOURCE_DIR}/tools/gguf_convertor/gguf.cpp
            ${PROJECT_SOURCE_DIR}/tools/gguf_convertor/gguf.hpp
            ${MLLM_QUANT}
            ${MLLM_QUANTIZER}

            # ${DIR_SRC}
            ${PROJECT_SOURCE_DIR}/src/ParamLoader.cpp
        )
        target_link_libraries(from_gguf fmt::fmt-header-only)
    endif()
endif()

if(TEST)
    add_subdirectory(test)
endif()

# add_executable(demo examples/demo.cpp)
# target_link_libraries(demo fmt::fmt-header-only)
add_subdirectory(examples)

if(APK)
    add_library(mllm_lib STATIC ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_PROCESSOE}
        ${DIR_THIRDPARTY_AUDIO}
        src/tokenizers/Tokenizer.cpp
        tools/jni/LibHelper.cpp

        # src/tokenizers/Tokenizer.hpp
        # src/tokenizers/Unigram/Unigram.hpp
        src/tokenizers/Unigram/Unigram.cpp

        # src/tokenizers/Unigram/trie.hpp
        src/tokenizers/BPE/Bpe.cpp

        # src/tokenizers/BPE/Bpe.hpp
        src/tokenizers/Unicode.cpp
        src/tokenizers/UnicodeData.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/WordPiece/WordPiece.cpp

        # models/bert/configuration_bert.hpp
        # models/bert/modeling_bert.hpp
        # models/bert/tokenization_bert.hpp
        # models/fuyu/configuration_fuyu.hpp
        # models/fuyu/modeling_fuyu.hpp
        # models/fuyu/processing_fuyu.hpp
        # models/phonelm/configuration_phonelm.hpp
        # models/phonelm/modeling_phonelm.hpp
        # models/qwen/configuration_qwen.hpp
        # models/qwen/modeling_qwen.hpp
        # models/qwen/tokenization_qwen.hpp
        # models/smollm/tokenization_smollm.hpp
        # tokenizers/Unigram/Unigram.hpp
    )
    target_link_libraries(mllm_lib MLLM_CPU)

    if(QNN)
        target_link_libraries(mllm_lib MLLM_QNN)
    endif()
endif()

if(MLLM_ENABLE_PYTHON)
    target_compile_options(MLLM_CPU PRIVATE -fPIC)

    find_package(Python3 COMPONENTS Interpreter Development)
    include_directories(${Python3_INCLUDE_DIRS})
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)

    set(_py_dep_libs
        MLLM_CPU

        # MLLM_QNN
        # ${CMAKE_DL_LIBS}
    )

    # pybind compile options
    set(_py_compile_opts

        # Override depends on RTTI.
        # -frtti
        # -fexceptions
        -fPIC
    )

    # pybind portable lib _C
    pybind11_add_module(_C
        SHARED
        ${PROJECT_SOURCE_DIR}/python/src/_C/PyWarp.cpp
        ${PROJECT_SOURCE_DIR}/python/src/_C/Core.cpp
    )
    target_compile_options(_C PUBLIC ${_py_compile_opts})
    target_link_libraries(_C PRIVATE ${_py_dep_libs})

    install(
        TARGETS _C
        LIBRARY DESTINATION mllm/
    )
endif()
