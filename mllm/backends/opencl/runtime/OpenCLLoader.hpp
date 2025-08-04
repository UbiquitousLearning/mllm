// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// NOTE:
// This file is highly inspired by MNN's impl.
// see:
// https://github.com/alibaba/MNN/blob/master/source/backend/opencl/core/runtime/OpenCLWrapper.hpp

#pragma once

// The CL_TARGET_OPENCL_VERSION is set for devices support OpenCL 3.0.
#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "mllm/utils/Common.hpp"

// The cl.hpp is deprecated. Pls use opencl.hpp instead
#include <CL/opencl.hpp>

// Special features from hardware vendors.
#include <CL/cl_ext.h>
#ifdef MLLM_OPENCL_GPU_ADRENO
// TODO include and process some adreno based features.
#endif

// Special features from sys side.
#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif

#define MLLM_CHECK_OPENCL_SUCCESS(error, info) \
  if (error != CL_SUCCESS) { MLLM_ASSERT_EXIT(kError, "OpenCL device side error. Error code: {}, info: {}", (int)error, info); }

namespace mllm::opencl {

class OpenCLLoader {
 public:
  static OpenCLLoader& instance() {
    static OpenCLLoader instance;
    return instance;
  }

  OpenCLLoader() = default;

  OpenCLLoader(const OpenCLLoader&) = delete;

  OpenCLLoader& operator=(const OpenCLLoader&) = delete;

  bool loadOpenCLDynLib();

  // Normal
  using clGetPlatformIDs_f_t = cl_int(CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
  using clGetPlatformInfo_f_t = cl_int(CL_API_CALL*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
  using clGetDeviceIDs_f_t = cl_int(CL_API_CALL*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
  using clGetDeviceInfo_f_t = cl_int(CL_API_CALL*)(cl_device_id, cl_device_info, size_t, void*, size_t*);

  using clReleaseDevice_f_t = cl_int(CL_API_CALL*)(cl_device_id);

  using clReleaseContext_f_t = cl_int(CL_API_CALL*)(cl_context);

  using clReleaseCommandQueue_f_t = cl_int(CL_API_CALL*)(cl_command_queue);

  using clRetainDevice_f_t = cl_int(CL_API_CALL*)(cl_device_id);

  using clCreateCommandQueueWithProperties_f_t = cl_command_queue(CL_API_CALL*)(cl_context, cl_device_id,
                                                                                const cl_queue_properties*, cl_int*);

  using clCreateContext_f_t = cl_context(CL_API_CALL*)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                                       void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*,
                                                       cl_int*);

  // SVM related
  using clSVMAlloc_f_t = void*(CL_API_CALL*)(cl_context, cl_svm_mem_flags, size_t, cl_uint);

  using clSVMFree_f_t = void(CL_API_CALL*)(cl_context, void*);

  using clSetKernelArgSVMPointer_f_t = cl_int(CL_API_CALL*)(cl_kernel, cl_uint, const void*);

  using clEnqueueSVMMap_f_t = cl_int(CL_API_CALL*)(cl_command_queue, cl_bool, cl_map_flags, void*, size_t, cl_uint,
                                                   const cl_event*, cl_event*);

  using clEnqueueSVMUnmap_f_t = cl_int(CL_API_CALL*)(cl_command_queue, void*, cl_uint, const cl_event*, cl_event*);

  using clRetainCommandQueue_f_t = cl_int(CL_API_CALL*)(cl_command_queue);

#define DEFINE_FUNC_PTR_MEMBER(func) func##_f_t func = nullptr
  // Normal
  DEFINE_FUNC_PTR_MEMBER(clGetPlatformIDs);
  DEFINE_FUNC_PTR_MEMBER(clGetPlatformInfo);
  DEFINE_FUNC_PTR_MEMBER(clGetDeviceIDs);
  DEFINE_FUNC_PTR_MEMBER(clGetDeviceInfo);
  DEFINE_FUNC_PTR_MEMBER(clReleaseDevice);
  DEFINE_FUNC_PTR_MEMBER(clReleaseContext);
  DEFINE_FUNC_PTR_MEMBER(clReleaseCommandQueue);
  DEFINE_FUNC_PTR_MEMBER(clRetainDevice);
  DEFINE_FUNC_PTR_MEMBER(clCreateCommandQueueWithProperties);
  DEFINE_FUNC_PTR_MEMBER(clCreateContext);
  DEFINE_FUNC_PTR_MEMBER(clRetainCommandQueue);

  // SVM related
  DEFINE_FUNC_PTR_MEMBER(clSVMAlloc);
  DEFINE_FUNC_PTR_MEMBER(clSVMFree);
  DEFINE_FUNC_PTR_MEMBER(clSetKernelArgSVMPointer);
  DEFINE_FUNC_PTR_MEMBER(clEnqueueSVMMap);
  DEFINE_FUNC_PTR_MEMBER(clEnqueueSVMUnmap);

#undef DEFINE_FUNC_PTR_MEMBER

  inline bool isSVMSymbolLoaded() { return !svm_load_error_; }

 private:
  bool tryingToLoadOpenCLDynLibAndParseSymbols(const std::string& lib_path);

  void* opencl_dynlib_handle_ = nullptr;

  bool svm_load_error_ = false;
};

/// wrap to same name symbols
extern "C" {
cl_int CL_API_CALL clGetPlatformIDs(cl_uint _0, cl_platform_id* _1, cl_uint* _2);

cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id _0, cl_platform_info _1, size_t _2, void* _3, size_t* _4);

cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id _0, cl_device_type _1, cl_uint _2, cl_device_id* _3, cl_uint* _4);

cl_int CL_API_CALL clGetDeviceInfo(cl_device_id _0, cl_device_info _1, size_t _2, void* _3, size_t* _4);

cl_int CL_API_CALL clReleaseDevice(cl_device_id _0);

cl_int CL_API_CALL clReleaseContext(cl_context _0);

cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue _0);

cl_context CL_API_CALL clCreateContext(const cl_context_properties* _0, cl_uint _1, const cl_device_id* _2,
                                       void(CL_CALLBACK* _3)(const char*, const void*, size_t, void*), void* _4, cl_int* _5);

cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(cl_context _0, cl_device_id _1, const cl_queue_properties* _2,
                                                                cl_int* _3);

cl_context CL_API_CALL clCreateContext(const cl_context_properties* _0, cl_uint _1, const cl_device_id* _2,
                                       void(CL_CALLBACK* _3)(const char*, const void*, size_t, void*), void* _4, cl_int* _5);

cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue _0);

// SVM
void* CL_API_CALL clSVMAlloc(cl_context _0, cl_svm_mem_flags _1, size_t _2, cl_uint _3);

void CL_API_CALL clSVMFree(cl_context _0, void* _1);

cl_int CL_API_CALL clSetKernelArgSVMPointer(cl_kernel _0, cl_uint _1, const void* _2);

cl_int CL_API_CALL clEnqueueSVMMap(cl_command_queue _0, cl_bool _1, cl_map_flags _2, void* _3, size_t _4, cl_uint _5,
                                   const cl_event* _6, cl_event* _7);

cl_int CL_API_CALL clEnqueueSVMUnmap(cl_command_queue _0, void* _1, cl_uint _2, const cl_event* _3, cl_event* _4);
}

}  // namespace mllm::opencl