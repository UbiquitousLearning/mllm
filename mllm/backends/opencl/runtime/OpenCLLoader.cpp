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

#include "mllm/utils/Log.hpp"
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
  if (func_name == nullptr) { MLLM_ERROR_EXIT(mllm::LogLevel::kError, "Failed to load OpenCL function: {}", #func_name); }

  // Normal
  LOAD_FUNCTION_PTR(clGetPlatformIDs);
  LOAD_FUNCTION_PTR(clGetPlatformInfo);
  LOAD_FUNCTION_PTR(clBuildProgram);
  LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel);
  LOAD_FUNCTION_PTR(clSetKernelArg);
  LOAD_FUNCTION_PTR(clReleaseKernel);
  LOAD_FUNCTION_PTR(clCreateProgramWithSource);
  LOAD_FUNCTION_PTR(clCreateBuffer);
  LOAD_FUNCTION_PTR(clCreateImage2D);
  LOAD_FUNCTION_PTR(clRetainKernel);
  LOAD_FUNCTION_PTR(clCreateKernel);
  LOAD_FUNCTION_PTR(clGetProgramInfo);
  LOAD_FUNCTION_PTR(clFlush);
  LOAD_FUNCTION_PTR(clFinish);
  LOAD_FUNCTION_PTR(clReleaseProgram);
  LOAD_FUNCTION_PTR(clRetainContext);
  LOAD_FUNCTION_PTR(clGetContextInfo);
  LOAD_FUNCTION_PTR(clCreateProgramWithBinary);
  LOAD_FUNCTION_PTR(clCreateCommandQueue);
  LOAD_FUNCTION_PTR(clReleaseCommandQueue);
  LOAD_FUNCTION_PTR(clEnqueueCopyBuffer);
  LOAD_FUNCTION_PTR(clEnqueueMapBuffer);
  LOAD_FUNCTION_PTR(clEnqueueMapImage);
  LOAD_FUNCTION_PTR(clEnqueueCopyImage);
  LOAD_FUNCTION_PTR(clRetainProgram);
  LOAD_FUNCTION_PTR(clGetProgramBuildInfo);
  LOAD_FUNCTION_PTR(clEnqueueReadBuffer);
  LOAD_FUNCTION_PTR(clEnqueueWriteBuffer);
  LOAD_FUNCTION_PTR(clWaitForEvents);
  LOAD_FUNCTION_PTR(clReleaseEvent);
  LOAD_FUNCTION_PTR(clCreateContext);
  LOAD_FUNCTION_PTR(clCreateContextFromType);
  LOAD_FUNCTION_PTR(clReleaseContext);
  LOAD_FUNCTION_PTR(clRetainCommandQueue);
  LOAD_FUNCTION_PTR(clEnqueueUnmapMemObject);
  LOAD_FUNCTION_PTR(clRetainMemObject);
  LOAD_FUNCTION_PTR(clReleaseMemObject);
  LOAD_FUNCTION_PTR(clGetDeviceInfo);
  LOAD_FUNCTION_PTR(clGetDeviceIDs);
  LOAD_FUNCTION_PTR(clRetainEvent);
  LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo);
  LOAD_FUNCTION_PTR(clGetEventInfo);
  LOAD_FUNCTION_PTR(clGetEventProfilingInfo);
  LOAD_FUNCTION_PTR(clGetMemObjectInfo);
  LOAD_FUNCTION_PTR(clGetImageInfo);
  LOAD_FUNCTION_PTR(clReleaseDevice);
  LOAD_FUNCTION_PTR(clEnqueueReadImage);
  LOAD_FUNCTION_PTR(clEnqueueWriteImage);
  LOAD_FUNCTION_PTR(clCreateCommandQueueWithProperties);
  LOAD_FUNCTION_PTR(clRetainDevice);

#define LOAD_SVM_FUNCTION_PTR(func_name)                                                   \
  func_name = reinterpret_cast<func_name##_f_t>(dlsym(opencl_dynlib_handle_, #func_name)); \
  if (func_name == nullptr && load_opencl_ptr_f != nullptr) {                              \
    func_name = reinterpret_cast<func_name##_f_t>(load_opencl_ptr_f(#func_name));          \
  }                                                                                        \
  if (func_name == nullptr) { svm_load_error_ = true; }

  LOAD_SVM_FUNCTION_PTR(clSVMAlloc);
  LOAD_SVM_FUNCTION_PTR(clSVMFree);
  LOAD_SVM_FUNCTION_PTR(clSetKernelArgSVMPointer);
  LOAD_SVM_FUNCTION_PTR(clEnqueueSVMMap);
  LOAD_SVM_FUNCTION_PTR(clEnqueueSVMUnmap);

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

cl_context CL_API_CALL clCreateContextFromType(const cl_context_properties* _0, cl_device_type _1,
                                               void(CL_CALLBACK* _2)(const char*, const void*, size_t, void*), void* _3,
                                               cl_int* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateContextFromType;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainCommandQueue;
  return func(_0);
}

cl_program CL_API_CALL clCreateProgramWithSource(cl_context _0, cl_uint _1, const char** _2, const size_t* _3, cl_int* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateProgramWithSource;
  return func(_0, _1, _2, _3, _4);
}

cl_program CL_API_CALL clCreateProgramWithBinary(cl_context _0, cl_uint _1, const cl_device_id* _2, const size_t* _3,
                                                 const unsigned char** _4, cl_int* _5, cl_int* _6) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateProgramWithBinary;
  return func(_0, _1, _2, _3, _4, _5, _6);
}

cl_int CL_API_CALL clGetProgramInfo(cl_program _0, cl_program_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetProgramInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clGetProgramBuildInfo(cl_program _0, cl_device_id _1, cl_program_build_info _2, size_t _3, void* _4,
                                         size_t* _5) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetProgramBuildInfo;
  return func(_0, _1, _2, _3, _4, _5);
}

cl_int CL_API_CALL clRetainProgram(cl_program _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainProgram;
  return func(_0);
}

cl_int CL_API_CALL clReleaseProgram(cl_program _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseProgram;
  return func(_0);
}

cl_int CL_API_CALL clBuildProgram(cl_program _0, cl_uint _1, const cl_device_id* _2, const char* _3,
                                  void(CL_CALLBACK* _4)(cl_program program, void* user_data), void* _5) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clBuildProgram;
  return func(_0, _1, _2, _3, _4, _5);
}

cl_kernel CL_API_CALL clCreateKernel(cl_program _0, const char* _1, cl_int* _2) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateKernel;
  return func(_0, _1, _2);
}

cl_int CL_API_CALL clRetainKernel(cl_kernel _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainKernel;
  return func(_0);
}

cl_int CL_API_CALL clReleaseKernel(cl_kernel _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseKernel;
  return func(_0);
}

cl_int CL_API_CALL clSetKernelArg(cl_kernel _0, cl_uint _1, size_t _2, const void* _3) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clSetKernelArg;
  return func(_0, _1, _2, _3);
}

cl_mem CL_API_CALL clCreateBuffer(cl_context _0, cl_mem_flags _1, size_t _2, void* _3, cl_int* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateBuffer;
  return func(_0, _1, _2, _3, _4);
}

cl_mem CL_API_CALL clCreateImage2D(cl_context _0, cl_mem_flags _1, const cl_image_format* _2, size_t _3, size_t _4, size_t _5,
                                   void* _6, cl_int* _7) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateImage2D;
  return func(_0, _1, _2, _3, _4, _5, _6, _7);
}

cl_int CL_API_CALL clRetainMemObject(cl_mem _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainMemObject;
  return func(_0);
}

cl_int CL_API_CALL clReleaseMemObject(cl_mem _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseMemObject;
  return func(_0);
}

cl_int CL_API_CALL clGetImageInfo(cl_mem _0, cl_image_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetImageInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clGetKernelWorkGroupInfo(cl_kernel _0, cl_device_id _1, cl_kernel_work_group_info _2, size_t _3, void* _4,
                                            size_t* _5) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetKernelWorkGroupInfo;
  return func(_0, _1, _2, _3, _4, _5);
}

cl_int CL_API_CALL clRetainDevice(cl_device_id _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainDevice;
  return func(_0);
}

cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context _0, cl_device_id _1, cl_command_queue_properties _2, cl_int* _3) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateCommandQueue;
  return func(_0, _1, _2, _3);
}

cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(cl_context _0, cl_device_id _1, const cl_queue_properties* _2,
                                                                cl_int* _3) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clCreateCommandQueueWithProperties;
  return func(_0, _1, _2, _3);
}

cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue _0, cl_kernel _1, cl_uint _2, const size_t* _3, const size_t* _4,
                                          const size_t* _5, cl_uint _6, const cl_event* _7, cl_event* _8) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueNDRangeKernel;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8);
}

void* CL_API_CALL clEnqueueMapBuffer(cl_command_queue _0, cl_mem _1, cl_bool _2, cl_map_flags _3, size_t _4, size_t _5,
                                     cl_uint _6, const cl_event* _7, cl_event* _8, cl_int* _9) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueMapBuffer;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9);
}

void* CL_API_CALL clEnqueueMapImage(cl_command_queue _0, cl_mem _1, cl_bool _2, cl_map_flags _3, const size_t* _4,
                                    const size_t* _5, size_t* _6, size_t* _7, cl_uint _8, const cl_event* _9, cl_event* _10,
                                    cl_int* _11) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueMapImage;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);
}

cl_int CL_API_CALL clEnqueueUnmapMemObject(cl_command_queue _0, cl_mem _1, void* _2, cl_uint _3, const cl_event* _4,
                                           cl_event* _5) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueUnmapMemObject;
  return func(_0, _1, _2, _3, _4, _5);
}

cl_int CL_API_CALL clEnqueueCopyBuffer(cl_command_queue _0, cl_mem _1, cl_mem _2, size_t _3, size_t _4, size_t _5, cl_uint _6,
                                       const cl_event* _7, cl_event* _8) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueCopyBuffer;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8);
}

cl_int CL_API_CALL clEnqueueCopyImage(cl_command_queue _0, cl_mem _1, cl_mem _2, const size_t* _3, const size_t* _4,
                                      const size_t* _5, cl_uint _6, const cl_event* _7, cl_event* _8) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueCopyImage;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8);
}

cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue _0, cl_mem _1, cl_bool _2, size_t _3, size_t _4, void* _5, cl_uint _6,
                                       const cl_event* _7, cl_event* _8) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueReadBuffer;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8);
}

cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue _0, cl_mem _1, cl_bool _2, size_t _3, size_t _4, const void* _5,
                                        cl_uint _6, const cl_event* _7, cl_event* _8) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueWriteBuffer;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8);
}

cl_int CL_API_CALL clEnqueueReadImage(cl_command_queue _0, cl_mem _1, cl_bool _2, const size_t* _3, const size_t* _4, size_t _5,
                                      size_t _6, void* _7, cl_uint _8, const cl_event* _9, cl_event* _10) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueReadImage;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10);
}

cl_int CL_API_CALL clEnqueueWriteImage(cl_command_queue _0, cl_mem _1, cl_bool _2, const size_t* _3, const size_t* _4,
                                       size_t _5, size_t _6, const void* _7, cl_uint _8, const cl_event* _9, cl_event* _10) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clEnqueueWriteImage;
  return func(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10);
}

cl_int CL_API_CALL clFlush(cl_command_queue _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clFlush;
  return func(_0);
}

cl_int CL_API_CALL clFinish(cl_command_queue _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clFinish;
  return func(_0);
}

cl_int CL_API_CALL clRetainContext(cl_context _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainContext;
  return func(_0);
}

cl_int CL_API_CALL clGetContextInfo(cl_context _0, cl_context_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetContextInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clWaitForEvents(cl_uint _0, const cl_event* _1) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clWaitForEvents;
  return func(_0, _1);
}

cl_int CL_API_CALL clReleaseEvent(cl_event _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clReleaseEvent;
  return func(_0);
}

cl_int CL_API_CALL clRetainEvent(cl_event _0) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clRetainEvent;
  return func(_0);
}

cl_int CL_API_CALL clGetEventInfo(cl_event _0, cl_event_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetEventInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clGetEventProfilingInfo(cl_event _0, cl_profiling_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetEventProfilingInfo;
  return func(_0, _1, _2, _3, _4);
}

cl_int CL_API_CALL clGetMemObjectInfo(cl_mem _0, cl_mem_info _1, size_t _2, void* _3, size_t* _4) {
  auto func = ::mllm::opencl::OpenCLLoader::instance().clGetMemObjectInfo;
  return func(_0, _1, _2, _3, _4);
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