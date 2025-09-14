file(GLOB_RECURSE MLLM_FFI_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_library(MllmFFI SHARED ${MLLM_FFI_SRC})
target_link_libraries(MllmFFI PUBLIC MllmRT)
target_include_directories(MllmFFI PUBLIC vendors/dlpack/include)
