cmake_minimum_required(VERSION 3.7.2)

project(mllm)

set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# temp executable name
set(TEST_EXE main_test)
option(TEST "test mode" ON)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)


# find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

option(DEBUG "debug print" OFF)
# backend options
option(NNAPI "Enable NNAPI" OFF)

if(CMAKE_HOST_UNIX)
    message("current platform: Linux ")
#    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14 -stdlib=libc++")
elseif(CMAKE_HOST_WIN32)
    message("current platform: Windows ")
else()
    message("current platform: unkonw ")
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src  DIR_SRC)
# aux_source_directory(${PROJECT_SOURCE_DIR}/src/backends/cpu  DIR_SRC_CPU)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/express  DIR_SRC_EXP)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/quantize  DIR_SRC_QUANT)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples  EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test  TEST_SRC)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdpartys/include)
# include_directories(${PROJECT_SOURCE_DIR}/src/backends/cpu)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/cpu)
add_compile_options(-mf16c)
add_compile_options(-mavx2)
if (TEST)
    add_definitions(-DTEST)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include_directories(${PROJECT_SOURCE_DIR}/test)
    # MLLM Tests
    file(GLOB_RECURSE MLLM_TEST
            ${CMAKE_CURRENT_LIST_DIR}/test/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/test/*.hpp

    )
    message(STATUS "MLLM_TEST: ${MLLM_TEST}")
    add_executable(
            MLLM_TEST
            ${PROJECT_SOURCE_DIR}/test/main.cpp
            ${MLLM_TEST}
            ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC} ${DIR_SRC_QUANT}
    )
    target_link_libraries(
            MLLM_TEST
            GTest::gtest_main
            MLLM_CPU
    )

    include(GoogleTest)
    gtest_discover_tests(MLLM_TEST)
    #else ()
    #    No Unit Test.
    add_executable(${TEST_EXE} ${PROJECT_SOURCE_DIR}/demo/main_test.cpp ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC} ${DIR_SRC_QUANT}
            src/tokenizers/Tokenizer.cpp
            src/tokenizers/Tokenizer.hpp
            src/tokenizers/BPE/Bpe.cpp
            src/tokenizers/BPE/Bpe.hpp)
    target_link_libraries(${TEST_EXE} MLLM_CPU)

endif ()

# nnapi temp test
if(NNAPI)
    add_definitions(-DNNAPI_ENABLED)
    # TODO: better debug log
    add_definitions(-DDEBUG)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/nnapi)
    add_executable(nnapi_test ${PROJECT_SOURCE_DIR}/demo/nnapi_test.cpp ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC} ${DIR_SRC_QUANT})
    target_link_libraries(nnapi_test MLLM_CPU MLLM_NNAPI)
endif()