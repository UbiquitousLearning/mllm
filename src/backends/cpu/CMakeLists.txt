# CPU Backend
file(GLOB MLLM_CPU_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)


add_library(
    MLLM_CPU
    OBJECT
    ${MLLM_CPU_SRC}
)

target_include_directories(
    MLLM_CPU
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)