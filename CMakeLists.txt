cmake_minimum_required(VERSION 3.10)

project(mllm)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)
option(ARM "build on ARM" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    set(ARM ON)
    set(ANDROID_PLATFORM android-28)

    # add_compile_options(-std=c++17)
endif()

if(ARM)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin-arm)
else()
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
endif()

# temp executable name
set(TEST_EXE main_test)
option(TEST "test mode" ON)
option(QUANT "quantize tools" ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
option(DEBUG "debug print" OFF)

# backend options
option(NNAPI "Enable NNAPI" OFF)

# backend options
option(QNN "Enable QNN" OFF)

if(CMAKE_HOST_UNIX)
    message("current platform: Linux ")

# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14 -stdlib=libc++")
elseif(CMAKE_HOST_WIN32)
    message("current platform: Windows ")
else()
    message("current platform: unknown ")
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRC)

# aux_source_directory(${PROJECT_SOURCE_DIR}/src/backends/cpu  DIR_SRC_CPU)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/express DIR_SRC_EXP)

# aux_source_directory(${PROJECT_SOURCE_DIR}/src/quantize  DIR_SRC_QUANT)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/memory DIR_SRC_MEM_MANAGER)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test TEST_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/quantizer QUANT_SRC)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdpartys/include)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/cpu)

# if compile to x86_64
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86_64 detected")
    add_compile_options(-mf16c)
    add_compile_options(-mavx2)
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
endif()

if(QUANT)
    include_directories(${PROJECT_SOURCE_DIR}/src/quantizer)
    file(GLOB_RECURSE MLLM_QUANT

        ${PROJECT_SOURCE_DIR}/src/backends/cpu/quantize/*.hpp
        ${PROJECT_SOURCE_DIR}/src/backends/cpu/quantize/*.cpp
    )

    # MLLM Quantizer is /src/quantizer/*.cpp but except main.cpp
    file(GLOB_RECURSE MLLM_QUANTIZER
        ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/*.hpp)
    list(REMOVE_ITEM MLLM_QUANTIZER ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/main.cpp)
    message(STATUS "MLLM_Quant: ${MLLM_QUANT}")
    add_executable(
        quantize
        ${PROJECT_SOURCE_DIR}/src/quantizer/main.cpp
        ${MLLM_QUANT}
        ${MLLM_QUANTIZER}

        # ${DIR_SRC}
        ${PROJECT_SOURCE_DIR}/src/ParamLoader.cpp
    )
    add_executable(
        from_gguf
        ${PROJECT_SOURCE_DIR}/src/quantizer/gguf.cpp
        ${MLLM_QUANT}
        ${MLLM_QUANTIZER}

        # ${DIR_SRC}
        ${PROJECT_SOURCE_DIR}/src/ParamLoader.cpp
    )

endif()

if(TEST)
    add_definitions(-DTEST)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include_directories(${PROJECT_SOURCE_DIR}/test)

    # MLLM Tests
    file(GLOB_RECURSE MLLM_TEST
        ${CMAKE_CURRENT_LIST_DIR}/test/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/test/*.hpp
    )
    list(REMOVE_ITEM MLLM_TEST ${CMAKE_CURRENT_LIST_DIR}/test/TestSystemMemoryManager.cpp)
    list(REMOVE_ITEM MLLM_TEST ${CMAKE_CURRENT_LIST_DIR}/test/TestMemoryPoolManager.cpp)

    message(STATUS "MLLM_TEST: ${MLLM_TEST}")

    # Add ${DIR_SRC_QUANT} to ${DIR_SRC}
    # if ()
    add_executable(
        MLLM_TEST
        ${PROJECT_SOURCE_DIR}/test/main.cpp
        ${MLLM_TEST}
        ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} ${MLLM_QUANTIZER}
    )

    target_link_libraries(
        MLLM_TEST
        GTest::gtest_main
        MLLM_CPU
    )
    add_executable(
        memoryPoolTest
        ${PROJECT_SOURCE_DIR}/test/TestMemoryPoolManager.cpp
        ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
    )
    include(GoogleTest)

    # if not cross-compiling
    if(NOT ARM)
        gtest_discover_tests(MLLM_TEST)
    endif()
endif()

# else ()
# No Unit Test.
add_executable(${TEST_EXE} ${PROJECT_SOURCE_DIR}/demo/main_test.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
    src/tokenizers/Tokenizer.cpp
    src/tokenizers/Tokenizer.hpp
    src/tokenizers/BPE/Bpe.cpp
    src/tokenizers/BPE/Bpe.hpp)
target_link_libraries(${TEST_EXE} MLLM_CPU)

# nnapi temp test
if(NNAPI)
    include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    add_definitions(-DNNAPI_ENABLED)

    # TODO: better debug log
    add_definitions(-DDEBUG)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/nnapi)
    add_executable(nnapi_test ${PROJECT_SOURCE_DIR}/demo/nnapi_test.cpp ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC}) # ${DIR_SRC_QUANT})
    target_link_libraries(nnapi_test MLLM_CPU MLLM_NNAPI)
endif()


# QNN test
if(QNN)
    if (ARM)
        include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    endif()
    add_definitions(-DQNN_ENABLED)
    
    include_directories(${PROJECT_SOURCE_DIR}/src/backends/QNN/Log)
    include_directories(${PROJECT_SOURCE_DIR}/src/backends/QNN/PAL/include)
    include_directories(${PROJECT_SOURCE_DIR}/src/backends/QNN/Utils)
    include_directories(${PROJECT_SOURCE_DIR}/src/backends/QNN/WrapperUtils)
    include_directories(${PROJECT_SOURCE_DIR}/src/backends/QNN/QNN)

    

    set(CMAKE_CXX_FLAGS " -fPIC -Wall  -pg -pthread -march=x86-64 -O3 -g -Wno-write-strings -fvisibility=hidden ")
    set(CMAKE_LD_FLAGS "-shared -s -fPIC -pthread -fvisibility=hidden -flto")

    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/QNN)
    add_executable(qnn_test ${PROJECT_SOURCE_DIR}/demo/qnn/qnn_test.cpp ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC} ) 
    target_link_libraries(qnn_test MLLM_CPU MLLM_QNN ${CMAKE_DL_LIBS})

endif()