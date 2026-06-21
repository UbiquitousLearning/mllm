// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cuda.h>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/cuda/CudaCommons.hpp"

namespace mllm::cuda {

NvGpuMetaInfo::NvGpuMetaInfo() {
  MLLM_NVML_CHECK(nvmlInit());

  CUresult cu_result = cuInit(0);
  if (cu_result != CUDA_SUCCESS) {
    MLLM_ERROR("Failed to initialize CUDA Driver API.");
    MLLM_NVML_CHECK(nvmlShutdown());
    return;
  }

  unsigned int device_count = 0;
  MLLM_NVML_CHECK(nvmlDeviceGetCount(&device_count));

  for (unsigned int i = 0; i < device_count; ++i) {
    nvmlDevice_t device;
    auto result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result != NVML_SUCCESS) {
      MLLM_ERROR("Failed to get device handle for GPU {} : {}.", i, nvmlErrorString(result));
      continue;
    }

    NvGpuInfo info;
    info.id = i;

    char name[256];
    result = nvmlDeviceGetName(device, name, sizeof(name));
    if (result == NVML_SUCCESS) {
      info.name = name;
    } else {
      info.name = "Unknown";
    }

    CUdevice cu_devices;
    cu_result = cuDeviceGet(&cu_devices, i);
    if (cu_result != CUDA_SUCCESS) {
      MLLM_ERROR("Failed to get CUDA device for GPU {}.", i);
      continue;
    }

    int sm_count;
    cu_result = cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.sm_count = sm_count;
    } else {
      info.sm_count = 0;
    }

    int compute_cap_major = 0;
    int compute_cap_minor = 0;
    CUresult cu_result_major =
        cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_devices);
    CUresult cu_result_minor =
        cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_devices);

    int cuda_core_per_sm = 0;
    int tensor_core_per_sm = 0;

    if (cu_result_major == CUDA_SUCCESS && cu_result_minor == CUDA_SUCCESS) {
      if (compute_cap_major == 7) {
        // Volta(7.0) Turing (7.5)
        cuda_core_per_sm = 64;
      } else if (compute_cap_major == 8) {
        // Ampere
        if (compute_cap_minor == 0 || compute_cap_minor == 7) {
          // A100(8.0)H100 (8.7)
          cuda_core_per_sm = 64;
        } else if (compute_cap_minor == 6 || compute_cap_minor == 9) {
          // GA10x(8.6) Ada Lovelace (8.9)
          cuda_core_per_sm = 128;
        } else {
          cuda_core_per_sm = 0;
        }
      } else if (compute_cap_major == 9) {
        // Hopper(9.0)
        cuda_core_per_sm = 128;
      } else {
        cuda_core_per_sm = 0;
      }

      switch (compute_cap_major) {
        case 7:
          // Volta Turing
          tensor_core_per_sm = 8;
          break;
        case 8:
        case 9:
          // Ampere Ada Hopper
          tensor_core_per_sm = 4;
          break;
        default: tensor_core_per_sm = 0; break;
      }
    } else {
      cuda_core_per_sm = 0;
      tensor_core_per_sm = 0;
    }

    info.cuda_core_per_sm = cuda_core_per_sm;
    info.tensor_core_per_sm = tensor_core_per_sm;

    int l1_cache;
    cu_result = cuDeviceGetAttribute(&l1_cache, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.l1_cache = l1_cache;
    } else {
      info.l1_cache = 0;
    }

    int shared_mem_per_sm;
    cu_result = cuDeviceGetAttribute(&shared_mem_per_sm, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.shared_mem_per_sm = shared_mem_per_sm;
    } else {
      info.shared_mem_per_sm = 0;
    }

    size_t global_mem;
    cu_result = cuDeviceTotalMem_v2(&global_mem, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.global_mem = global_mem;
    } else {
      info.global_mem = 0;
    }

    int max_thread_per_sm;
    cu_result = cuDeviceGetAttribute(&max_thread_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.max_thread_per_sm = max_thread_per_sm;
    } else {
      info.max_thread_per_sm = 0;
    }

    int warp_size_in_thread;
    cu_result = cuDeviceGetAttribute(&warp_size_in_thread, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.warp_size_in_thread = warp_size_in_thread;
    } else {
      info.warp_size_in_thread = 0;
    }

    int architecture;
    cu_result = cuDeviceGetAttribute(&architecture, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_devices);
    if (cu_result == CUDA_SUCCESS) {
      info.architecture = architecture * 10 + compute_cap_major;
    } else {
      info.architecture = 0;
    }

    this->devices.emplace_back(info);
  }

  MLLM_NVML_CHECK(nvmlShutdown());
}
}  // namespace mllm::cuda
