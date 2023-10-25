# CPU Backend
file(GLOB MLLM_CPU_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/compute/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/compute/*.hpp
)


find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found openmp")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "openmp not found!")
endif()
add_compile_options(-mavx2)
add_compile_options(-march=native)
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

target_link_libraries(MLLM_CPU
        PUBLIC
        OpenMP::OpenMP_CXX )