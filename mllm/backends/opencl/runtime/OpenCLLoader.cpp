// Copyright (c) MLLM Team.
// Licensed under the MIT License.
// NOTE:
// This file is highly inspired by MNN's impl.
// see:
// https://github.com/alibaba/MNN/blob/master/source/backend/opencl/core/runtime/OpenCLWrapper.cpp

#include <vector>
#include <string>

// This head only works on linux
#include <dlfcn.h>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"

namespace mllm::opencl {

bool OpenCLLoader::loadOpenCLDynLib() {
  if (opencl_dynlib_handle_) {
    MLLM_WARN("OpenCL dyn lib is already loaded.");
    return true;
  }

  // This path is only for android platform.
  // Support GPU:
  // Adreno and Mali
  static const std::vector<std::string> possible_opencl_dyn_lib_paths{
      /// -- Android sys path?
      "libOpenCL.so",
      "libGLES_mali.so",
      "libmali.so",
      "libOpenCL-pixel.so",

      /// -- __aarch64__ path?
      // Qualcomm Adreno
      "/system/vendor/lib64/libOpenCL.so",
      "/system/lib64/libOpenCL.so",
      // Mali
      "/system/vendor/lib64/egl/libGLES_mali.so",
      "/system/lib64/egl/libGLES_mali.so",
  };

  for (const auto& lib_path : possible_opencl_dyn_lib_paths) {
    if (tryingToLoadOpenCLDynLibAndParseSymbols(lib_path)) { return true; }
  }

  return false;
}

bool OpenCLLoader::tryingToLoadOpenCLDynLibAndParseSymbols(const std::string& lib_path) {
  opencl_dynlib_handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (opencl_dynlib_handle_ == nullptr) { return false; }

  MLLM_INFO("Load opencl dyn lib: {}", lib_path);

  // Load opencl symbols
  using enable_opencl_f_t = void (*)();
  using load_opencl_ptr_f_t = void* (*)(const char*);

  load_opencl_ptr_f_t load_opencl_ptr_f = nullptr;
  enable_opencl_f_t enable_opencl_f = reinterpret_cast<enable_opencl_f_t>(dlsym(opencl_dynlib_handle_, "enableOpenCL"));
  if (enable_opencl_f != nullptr) {
    enable_opencl_f();
    load_opencl_ptr_f = reinterpret_cast<load_opencl_ptr_f_t>(dlsym(opencl_dynlib_handle_, "loadOpenCLPointer"));
  }

#define LOAD_FUNCTION_PTR(func_name)                                                       \
  func_name = reinterpret_cast<func_name##_f_t>(dlsym(opencl_dynlib_handle_, #func_name)); \
  if (func_name == nullptr && load_opencl_ptr_f != nullptr) {                              \
    func_name = reinterpret_cast<func_name##_f_t>(load_opencl_ptr_f(#func_name));          \
  }                                                                                        \
  if (func_name == nullptr) { MLLM_ERROR_EXIT(kError, "Failed to load OpenCL function: {}", #func_name); }

  LOAD_FUNCTION_PTR(clGetPlatformIDs);
  LOAD_FUNCTION_PTR(clGetPlatformInfo);
  LOAD_FUNCTION_PTR(clGetDeviceIDs);
  LOAD_FUNCTION_PTR(clGetDeviceInfo);
  LOAD_FUNCTION_PTR(clReleaseDevice);
  LOAD_FUNCTION_PTR(clReleaseContext);
  LOAD_FUNCTION_PTR(clReleaseCommandQueue);
  LOAD_FUNCTION_PTR(clRetainDevice);
  LOAD_FUNCTION_PTR(clCreateCommandQueueWithProperties);
  LOAD_FUNCTION_PTR(clCreateContext);
  LOAD_FUNCTION_PTR(clRetainCommandQueue);

#define LOAD_SVM_FUNCTION_PTR(func_name)                                                   \
  func_name = reinterpret_cast<func_name##_f_t>(dlsym(opencl_dynlib_handle_, #func_name)); \
  if (func_name == nullptr && load_opencl_ptr_f != nullptr) {                              \
    func_name = reinterpret_cast<func_name##_f_t>(load_opencl_ptr_f(#func_name));          \
  }                                                                                        \
  if (func_name == nullptr) { svm_load_error_ = true; }

  LOAD_SVM_FUNCTION_PTR(clSVMAlloc);
  LOAD_SVM_FUNCTION_PTR(clSVMFree);
  LOAD_SVM_FUNCTION_PTR(clEnqueueSVMMap);
  LOAD_SVM_FUNCTION_PTR(clEnqueueSVMUnmap);
  LOAD_SVM_FUNCTION_PTR(clSetKernelArgSVMPointer);

  // TODO More functions and special features to load

#undef LOAD_FUNCTION_PTR
#undef LOAD_SVM_FUNCTION_PTR

  return true;
}

/// wrap to same name symbols
extern "C" {
cl_int CL_API_CALL clGetPlatformIDs(cl_uint _0, cl_platform_id* _1, cl_uint* _2) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetPlatformIDs;
  return func(_0, _1, _2);
}

cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id _0, cl_platform_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetPlatformInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id _0, cl_device_type _1, cl_uint _2, cl_device_id* _3, cl_uint* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetDeviceIDs;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clGetDeviceInfo(cl_device_id _0, cl_device_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetDeviceInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clReleaseDevice(cl_device_id _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseDevice;
  return func(_0);
}

cl_int CL_API_CALL clReleaseContext(cl_context _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseContext;
  return func(_0);
}

cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseCommandQueue;
  return func(_0);
}

cl_context CL_API_CALL clCreateContext(const cl_context_properties* _0, cl_uint _1, const cl_device_id* _2,
                                       void(CL_CALLBACK* _3)(const char*, const void*, size_t, void*), void* _4, cl_int* _5) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateContext;
  return func(_0, _1, _2, _3, _4, _5);
}

cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(cl_context _0, cl_device_id _1, const cl_queue_properties* _2,
                                                                cl_int* _3) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateCommandQueueWithProperties;
  return func(_0, _1, _2, _3);
}

cl_int CL_API_CALL clRetainDevice(cl_device_id _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainDevice;
  return func(_0);
}

cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainCommandQueue;
  return func(_0);
}

void* CL_API_CALL clSVMAlloc(cl_context _0, cl_svm_mem_flags _1, size_t _2, cl_uint _3) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clSVMAlloc;
  return func(_0, _1, _2, _3);
}

void CL_API_CALL clSVMFree(cl_context _0, void* _1) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clSVMFree;
  func(_0, _1);
}

cl_int CL_API_CALL clSetKernelArgSVMPointer(cl_kernel _0, cl_uint _1, const void* _2) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clSetKernelArgSVMPointer;
  return func(_0, _1, _2);
}

cl_int CL_API_CALL clEnqueueSVMMap(cl_command_queue _0, cl_bool _1, cl_map_flags _2, void* _3, size_t _4, cl_uint _5,
                                   const cl_event* _6, cl_event* _7) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueSVMMap;
  return func(_0, _1, _2, _3, _4, _5, _6, _7);
}

cl_int CL_API_CALL clEnqueueSVMUnmap(cl_command_queue _0, void* _1, cl_uint _2, const cl_event* _3, cl_event* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueSVMUnmap;
  return func(_0, _1, _2, _3, _4);
}
}

}  // namespace mllm::opencl