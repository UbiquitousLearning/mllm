enable_testing()

if(NOT QNN AND NOT MLLM_BUILD_XNNPACK_BACKEND)
# FIXME. TEST was defined twice.
add_definitions(-DTEST)
include_directories(${PROJECT_SOURCE_DIR}/test)

# MLLM Tests
file(GLOB_RECURSE MLLM_TEST
        ${PROJECT_SOURCE_DIR}/test/cpu/*.cpp
        ${PROJECT_SOURCE_DIR}/test/cpu/*.hpp
        ${PROJECT_SOURCE_DIR}/test/quantizer/*.cpp
        ${PROJECT_SOURCE_DIR}/test/quantizer/*.hpp
        ${PROJECT_SOURCE_DIR}/test/processor/*.cpp
        ${PROJECT_SOURCE_DIR}/test/processor/*.hpp
)
list(REMOVE_ITEM MLLM_TEST ${PROJECT_SOURCE_DIR}/test/TestSystemMemoryManager.cpp)
list(REMOVE_ITEM MLLM_TEST ${PROJECT_SOURCE_DIR}/test/TestMemoryPoolManager.cpp)
# list(REMOVE_ITEM MLLM_TEST ${PROJECT_SOURCE_DIR}/test/clip_tokenizer_test.cpp)

# Add ${DIR_SRC_QUANT} to ${DIR_SRC}
# if ()
add_executable(
        MLLM_TEST
        ${PROJECT_SOURCE_DIR}/test/main.cpp
        ${PROJECT_SOURCE_DIR}/test/TestLoader.cpp
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
# add_executable(
#         memoryPoolTest
#         ${PROJECT_SOURCE_DIR}/test/TestMemoryPoolManager.cpp
#         ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
# )
# add_executable(
#         SystemMemoryTest
#         ${PROJECT_SOURCE_DIR}/test/TestSystemMemoryManager.cpp
#         ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
# )
endif()

if(MLLM_BUILD_XNNPACK_BACKEND)
    add_subdirectory(xnnpack)
endif()

include(GoogleTest)

if(NOT QNN AND NOT MLLM_BUILD_XNNPACK_BACKEND)
# if not cross-compiling
if (NOT ARM)
    gtest_discover_tests(MLLM_TEST)
endif ()
endif() # the above is for cpu