# CPU Backend
file(GLOB MLLM_CPU_SRC
        ${CMAKE_CURRENT_LIST_DIR}/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/compute/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/compute/*.hpp
#        ${CMAKE_CURRENT_LIST_DIR}/neon/*.hpp
#        ${CMAKE_CURRENT_LIST_DIR}/neon/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/quantize/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/quantize/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/type/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/type/*.hpp
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
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86_64 detected")
add_compile_options(-mavx2)
add_compile_options(-march=native)
endif()

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

if(OpenMP_FOUND)
    message(STATUS "found openmp")
    if(ARM AND NOT APK)
        message(STATUS "[ARM]  found openmp")
        target_compile_options(MLLM_CPU PRIVATE -fopenmp)
        target_link_libraries(MLLM_CPU PUBLIC -fopenmp -static-openmp)
    else()
    target_link_libraries(MLLM_CPU
            PUBLIC
            OpenMP::OpenMP_CXX )
    endif()
endif()