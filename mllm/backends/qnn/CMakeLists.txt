
# QNN Backend
file(GLOB MLLM_QNN_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h

    ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.hpp

    ${CMAKE_CURRENT_LIST_DIR}/passes/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/passes/*.hpp
    
)

add_library(
    MllmQNNBackend
    SHARED
    ${MLLM_QNN_SRC}
)

target_include_directories(MllmQNNBackend PUBLIC
    $ENV{QNN_SDK_ROOT}/include/QNN # QNN SDK include
    ${MLLM_INCLUDE_DIR}
)

message(STATUS "QNN SDK root: $ENV{QNN_SDK_ROOT}")

get_property(current_includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "MLLM_QNN INCLUDES: ${current_includes}") #print include directories

target_link_libraries(MllmQNNBackend PUBLIC MllmRT)
