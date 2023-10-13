# MLLM Tests
# ===================
file(GLOB MLLM_TEST
        ${CMAKE_CURRENT_LIST_DIR}/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cpu/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpu/*.hpp
)
message(STATUS "MLLM_TEST: ${MLLM_TEST}")
add_executable(
        MLLM_TEST
        ${MLLM_TEST}
)
target_link_libraries(
        MLLM_TEST
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(MLLM_TEST)

