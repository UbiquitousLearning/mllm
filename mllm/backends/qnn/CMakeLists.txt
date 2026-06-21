
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

if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  file(GLOB_RECURSE MLLM_QUALCOMM_AOT_SRC 
    ${CMAKE_CURRENT_LIST_DIR}/aot/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/aot/*.cpp
  )
  list(APPEND MLLM_QNN_SRC ${MLLM_QUALCOMM_AOT_SRC})
endif()

file(GLOB_RECURSE MLLM_QUALCOMM_AOT_RT_SRC 
    ${CMAKE_CURRENT_LIST_DIR}/aot_rt/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/aot_rt/*.cpp
  )
list(APPEND MLLM_QNN_SRC ${MLLM_QUALCOMM_AOT_RT_SRC})

add_library(
    MllmQNNBackend
    SHARED
    ${MLLM_QNN_SRC}
)

target_include_directories(MllmQNNBackend PUBLIC
    $ENV{QAIRT_SDK_ROOT}/include/QNN # QNN SDK include
    ${MLLM_INCLUDE_DIR}
)

message(STATUS "QNN SDK root: $ENV{QAIRT_SDK_ROOT}")

get_property(current_includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "MLLM_QNN INCLUDES: ${current_includes}") #print include directories

target_link_libraries(MllmQNNBackend PUBLIC MllmRT)

install(
  TARGETS MllmQNNBackend
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
