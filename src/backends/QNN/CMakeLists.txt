include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/QNN
    ${CMAKE_CURRENT_LIST_DIR}/PAL/include
)

get_property(current_includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "MLLM_QNN INCLUDES: ${current_includes}") #print include directories

# QNN Backend
file(GLOB MLLM_QNN_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h

    ${CMAKE_CURRENT_LIST_DIR}/Log/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Log/*.hpp

    ${CMAKE_CURRENT_LIST_DIR}/PAL/include/PAL/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/PAL/src/common/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/PAL/src/linux/*.cpp

    ${CMAKE_CURRENT_LIST_DIR}/Utils/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Utils/*.hpp

    ${CMAKE_CURRENT_LIST_DIR}/WrapperUtils/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/WrapperUtils/*.hpp

    ${CMAKE_CURRENT_LIST_DIR}/Model/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Model/*.hpp

    ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.hpp
    
)

# import android ndk cmake toolchain
if (ARM)
    include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
endif()


add_library(
    MLLM_QNN
    OBJECT
    ${MLLM_QNN_SRC}
)
