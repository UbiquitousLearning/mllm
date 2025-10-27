get_property(current_includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "MLLM_QNN INCLUDES: ${current_includes}") #print include directories

# QNN Backend
file(GLOB MLLM_QNN_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h

    ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.hpp
    
)

if (MLLM_OPENMP)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found openmp for QNN")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "openmp not found!")
endif()
endif()

# import android ndk cmake toolchain
if (ARM)
    add_definitions(-DQNN_ARM)
    message(STATUS "QNN ARM detected")
endif()


add_library(
    mllm_qnn
    OBJECT
    ${MLLM_QNN_SRC}
)

if(OpenMP_FOUND)
    if(ARM AND NOT APK)
        message(STATUS "[ARM]  found openmp")
        target_compile_options(mllm_qnn PRIVATE -fopenmp)
        target_link_libraries(mllm_qnn PUBLIC -fopenmp -static-openmp)
    else()
    target_link_libraries(mllm_qnn
            PUBLIC
            OpenMP::OpenMP_CXX )
    endif()
endif()
target_link_libraries(mllm_qnn PUBLIC fmt::fmt-header-only)