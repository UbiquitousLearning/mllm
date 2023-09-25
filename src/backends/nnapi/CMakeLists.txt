# CPU Backend
file(GLOB MLLM_NNAPI_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.hpp
)


add_library(
    MLLM_NNAPI
    OBJECT
    ${MLLM_NNAPI_SRC}
)

target_include_directories(
    MLLM_NNAPI
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)