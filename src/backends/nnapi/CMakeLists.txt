# NNAPI Backend
file(GLOB MLLM_NNAPI_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
)

# import android ndk cmake toolchain
#include(/Users/luis/Library/Android/sdk/ndk/25.1.8937393/build/cmake/android.toolchain.cmake)
include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)

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