# FIXME. TEST was defined twice.
add_definitions(-DTEST)

enable_testing()
include_directories(${PROJECT_SOURCE_DIR}/test)

# MLLM Tests
file(GLOB_RECURSE MLLM_TEST
        ${PROJECT_SOURCE_DIR}/test/*.cpp
        ${PROJECT_SOURCE_DIR}/test/*.hpp
)
list(REMOVE_ITEM MLLM_TEST ${PROJECT_SOURCE_DIR}/test/TestSystemMemoryManager.cpp)
list(REMOVE_ITEM MLLM_TEST ${PROJECT_SOURCE_DIR}/test/TestMemoryPoolManager.cpp)
# list(REMOVE_ITEM MLLM_TEST ${PROJECT_SOURCE_DIR}/test/clip_tokenizer_test.cpp)

message(STATUS "MLLM_TEST: ${MLLM_TEST}")

# Add ${DIR_SRC_QUANT} to ${DIR_SRC}
# if ()
add_executable(
        MLLM_TEST
        ${PROJECT_SOURCE_DIR}/test/main.cpp
        ${MLLM_TEST}
        ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} ${MLLM_QUANTIZER} ${SRC_TOKENIZERS}
        ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.hpp
        ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
        ${PROJECT_SOURCE_DIR}/src/processor/FuyuPreProcess.hpp
        ${PROJECT_SOURCE_DIR}/src/processor/FuyuPreProcess.cpp
        ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.hpp
        ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
        ${PROJECT_SOURCE_DIR}/test/processor/ClipPreprocessorTest.cpp
)

target_link_libraries(
        MLLM_TEST
        GTest::gtest_main
        MLLM_CPU
        -fopenmp
)
add_executable(
        memoryPoolTest
        ${PROJECT_SOURCE_DIR}/test/TestMemoryPoolManager.cpp
        ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
)
add_executable(
        SystemMemoryTest
        ${PROJECT_SOURCE_DIR}/test/TestSystemMemoryManager.cpp
        ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
)
include(GoogleTest)

# if not cross-compiling
if (NOT ARM)
    gtest_discover_tests(MLLM_TEST)
endif ()