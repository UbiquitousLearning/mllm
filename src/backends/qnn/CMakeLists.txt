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

if (MLLM_OPENMP)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found openmp")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "openmp not found!")
endif()
endif()

# import android ndk cmake toolchain
if (ARM)
    include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    add_definitions(-DQNN_ARM)
    message(STATUS "QNN ARM detected")
endif()


add_library(
    MLLM_QNN
    OBJECT
    ${MLLM_QNN_SRC}
)

if(OpenMP_FOUND)
    message(STATUS "found openmp")
    if(ARM AND NOT APK)
        message(STATUS "[ARM]  found openmp")
        target_compile_options(MLLM_QNN PRIVATE -fopenmp)
        target_link_libraries(MLLM_QNN PUBLIC -fopenmp -static-openmp)
    else()
    target_link_libraries(MLLM_QNN
            PUBLIC
            OpenMP::OpenMP_CXX )
    endif()
endif()
target_link_libraries(MLLM_QNN PUBLIC fmt::fmt-header-only)