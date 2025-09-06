include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else()
  message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif()

file(GLOB_RECURSE MLLM_CUDA_BACKEND_CU_OP ${CMAKE_CURRENT_LIST_DIR}/ops/*.cu)
file(GLOB_RECURSE MLLM_CUDA_BACKEND_SRC_OP_CPP ${CMAKE_CURRENT_LIST_DIR}/ops/*.cpp)
file(GLOB_RECURSE MLLM_CUDA_BACKEND_SRC_PASSES_CPP ${CMAKE_CURRENT_LIST_DIR}/passes/*.cpp)
file(GLOB_RECURSE MLLM_CUDA_BACKEND_CU_KERNELS ${CMAKE_CURRENT_LIST_DIR}/kernels/*.cu)

find_package(CUDAToolkit REQUIRED)

add_library(MllmCUDABackendCudaOps SHARED ${MLLM_CUDA_BACKEND_CU_OP} ${MLLM_CUDA_BACKEND_CU_KERNELS})
set_target_properties(MllmCUDABackendCudaOps PROPERTIES CUDA_RUNTIME_LIBRARY Shared POSITION_INDEPENDENT_CODE ON)
target_include_directories(MllmCUDABackendCudaOps PUBLIC ${MLLM_INCLUDE_DIR})

# handle cutlass stuff
target_include_directories(
  MllmCUDABackendCudaOps
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/vendors/cutlass/include
         ${CMAKE_CURRENT_LIST_DIR}/vendors/cccl/include)
target_compile_options(MllmCUDABackendCudaOps PRIVATE --expt-relaxed-constexpr)

add_library(
  MllmCUDABackend SHARED
  ${MLLM_CUDA_BACKEND_SRC_OP_CPP} ${MLLM_CUDA_BACKEND_SRC_PASSES_CPP}
  CudaAllocator.cpp CudaBackend.cpp CudaCommons.cpp CudaDispatcher.cpp)

target_link_directories(MllmCUDABackend PUBLIC ${CUDAToolkit_LIBRARY_DIR}/stubs/)
target_link_libraries(MllmCUDABackend PUBLIC MllmRT MllmCUDABackendCudaOps CUDA::cudart -lcuda -lnvidia-ml)
