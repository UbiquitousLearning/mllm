#include "OpenCLBackend.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#if defined(MLLM_TARGET_ANDROID)
#include <unistd.h>
#include <libgen.h> // for dirname
#endif
#include <filesystem>   // C++17, for directory creation
#include <system_error> // for std::error_code

#include "Tensor.hpp"
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include "Module.hpp"
#include "utils/OpenCLTools.hpp"
#include "op/OpenCLAddOp.hpp"
#include "op/OpenCLAddTwoOp.hpp"
#include "op/OpenCLSubOp.hpp"
#include "op/OpenCLSubTwoOp.hpp"
#include "op/OpenCLMulOp.hpp"
#include "op/OpenCLMulTwoOp.hpp"
#include "op/OpenCLDivOp.hpp"
#include "op/OpenCLDivIntOp.hpp"
#include "op/OpenCLDivTwoOp.hpp"
#include "op/OpenCLMatmulOp.hpp"
#include "op/OpenCLLinearOp.hpp"
#include "op/OpenCLTransposeOp.hpp"
#include "op/OpenCLSoftMaxOp.hpp"
#include "op/OpenCLRMSNormOp.hpp"
#include "op/OpenCLEmbeddingOp.hpp"
#include "op/OpenCLSiLUOp.hpp"
#include "op/OpenCLViewOp.hpp"
#include "op/OpenCLKVCacheOp.hpp"
#include "op/OpenCLRoPEOp.hpp"
#include "op/OpenCLClipOp.hpp"
#include "op/OpenCLFlashAttentionOp.hpp"
#include "op/OpenCLSplitOp.hpp"
#include "op/OpenCLTopkOp.hpp"
#include "op/OpenCLSumOp.hpp"
#include "op/OpenCLLikeOp.hpp"
#include "op/OpenCLClipTensorOp.hpp"
#include "op/OpenCLScatterAddOp.hpp"
#include "op/OpenCLArgSortOp.hpp"
#include "op/OpenCLBinCountOp.hpp"

// 错误检查函数
void check_cl_error(cl_int err, const std::string &operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error during " << operation << " (" << err << ")" << std::endl;
        throw std::runtime_error("OpenCL Error: " + operation);
    }
}

// 从文件加载内核源码的辅助函数
std::string load_file_contents(const char *filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in) {
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    throw std::runtime_error(std::string("Could not open file: ") + filename);
}

#if defined(MLLM_TARGET_ANDROID)
std::string get_executable_dir() {
    char path_buf[1024] = {0};
    // 读取 /proc/self/exe 符号链接，获取可执行文件的完整路径
    ssize_t len = readlink("/proc/self/exe", path_buf, sizeof(path_buf) - 1);
    if (len != -1) {
        path_buf[len] = '\0';
        // 使用 dirname 获取路径的目录部分
        return std::string(dirname(path_buf));
    }
    // 如果失败，返回一个默认的相对路径作为后备
    return ".";
}
#endif

namespace mllm {

#if defined(MLLM_TARGET_ANDROID)
// 【关键修正】将 OpenCLSymbols 的完整定义放回到 .cpp 文件中
struct OpenCLSymbols {
    typedef cl_int (*clGetPlatformIDs_f_t)(cl_uint, cl_platform_id *, cl_uint *);
    typedef cl_int (*clGetDeviceIDs_f_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    typedef cl_int (*clGetDeviceInfo_f_t)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    typedef cl_context (*clCreateContext_f_t)(const cl_context_properties *, cl_uint, const cl_device_id *, void(CL_CALLBACK *)(const char *, const void *, size_t, void *), void *, cl_int *);
    typedef cl_command_queue (*clCreateCommandQueue_f_t)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
    typedef cl_int (*clReleaseCommandQueue_f_t)(cl_command_queue);
    typedef cl_int (*clReleaseContext_f_t)(cl_context);
    typedef cl_program (*clCreateProgramWithSource_f_t)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    typedef cl_int (*clBuildProgram_f_t)(cl_program, cl_uint, const cl_device_id *, const char *, void(CL_CALLBACK *)(cl_program, void *), void *);
    typedef cl_int (*clGetProgramBuildInfo_f_t)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
    typedef cl_program (*clCreateProgramWithBinary_f_t)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
    typedef cl_int (*clGetProgramInfo_f_t)(cl_program, cl_program_info, size_t, void *, size_t *);
    typedef cl_int (*clReleaseProgram_f_t)(cl_program);
    typedef cl_kernel (*clCreateKernel_f_t)(cl_program, const char *, cl_int *);
    typedef cl_int (*clReleaseKernel_f_t)(cl_kernel);
    typedef cl_int (*clSetKernelArg_f_t)(cl_kernel, cl_uint, size_t, const void *);
    typedef cl_int (*clEnqueueNDRangeKernel_f_t)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    typedef cl_mem (*clCreateBuffer_f_t)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    typedef cl_int (*clReleaseMemObject_f_t)(cl_mem);
    typedef cl_int (*clEnqueueWriteBuffer_f_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueReadBuffer_f_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clFinish_f_t)(cl_command_queue);
    typedef cl_sampler (*clCreateSampler_f_t)(cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *);
    typedef cl_int (*clReleaseSampler_f_t)(cl_sampler);
    typedef cl_mem (*clCreateImage_f_t)(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *);
    typedef cl_int (*clEnqueueWriteImage_f_t)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueReadImage_f_t)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueWriteBufferRect_f_t)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueReadBufferRect_f_t)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clReleaseDevice_f_t)(cl_device_id);
    typedef cl_int (*clRetainDevice_f_t)(cl_device_id);
    typedef cl_command_queue (*clCreateCommandQueueWithProperties_f_t)(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
    typedef cl_int (*clRetainCommandQueue_f_t)(cl_command_queue);
    typedef void *(*clSVMAlloc_f_t)(cl_context, cl_svm_mem_flags, size_t, cl_uint);
    typedef void (*clSVMFree_f_t)(cl_context, void *);
    typedef cl_int (*clEnqueueSVMMap_f_t)(cl_command_queue, cl_bool, cl_map_flags, void *, size_t, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueSVMUnmap_f_t)(cl_command_queue, void *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clSetKernelArgSVMPointer_f_t)(cl_kernel, cl_uint, const void *);
    typedef cl_mem (*clCreateSubBuffer_f_t)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *);
    typedef cl_int (*clEnqueueCopyBuffer_f_t)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueCopyBufferToImage_f_t)(cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueCopyBufferRect_f_t)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clWaitForEvents_f_t)(cl_uint, const cl_event *);
    typedef cl_int (*clGetEventProfilingInfo_f_t)(cl_event, cl_profiling_info, size_t, void *, size_t *);
    typedef cl_int (*clReleaseEvent_f_t)(cl_event);
    typedef cl_int (*clEnqueueCopyImageToBuffer_f_t)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, size_t, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clEnqueueCopyImage_f_t)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    typedef cl_int (*clGetMemObjectInfo_f_t)(cl_mem, cl_mem_info, size_t, void *, size_t *);
    typedef cl_int (*clEnqueueFillBuffer_f_t)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
    typedef void *(*clEnqueueMapBuffer_f_t)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
    typedef cl_int (*clEnqueueUnmapMemObject_f_t)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);

    clGetPlatformIDs_f_t clGetPlatformIDs = nullptr;
    clGetDeviceIDs_f_t clGetDeviceIDs = nullptr;
    clGetDeviceInfo_f_t clGetDeviceInfo = nullptr;
    clCreateContext_f_t clCreateContext = nullptr;
    clCreateCommandQueue_f_t clCreateCommandQueue = nullptr;
    clReleaseCommandQueue_f_t clReleaseCommandQueue = nullptr;
    clReleaseContext_f_t clReleaseContext = nullptr;
    clCreateProgramWithSource_f_t clCreateProgramWithSource = nullptr;
    clBuildProgram_f_t clBuildProgram = nullptr;
    clGetProgramBuildInfo_f_t clGetProgramBuildInfo = nullptr;
    clCreateProgramWithBinary_f_t clCreateProgramWithBinary = nullptr;
    clGetProgramInfo_f_t clGetProgramInfo = nullptr;
    clReleaseProgram_f_t clReleaseProgram = nullptr;
    clCreateKernel_f_t clCreateKernel = nullptr;
    clReleaseKernel_f_t clReleaseKernel = nullptr;
    clSetKernelArg_f_t clSetKernelArg = nullptr;
    clEnqueueNDRangeKernel_f_t clEnqueueNDRangeKernel = nullptr;
    clCreateBuffer_f_t clCreateBuffer = nullptr;
    clReleaseMemObject_f_t clReleaseMemObject = nullptr;
    clEnqueueWriteBuffer_f_t clEnqueueWriteBuffer = nullptr;
    clEnqueueReadBuffer_f_t clEnqueueReadBuffer = nullptr;
    clFinish_f_t clFinish = nullptr;
    clCreateSampler_f_t clCreateSampler = nullptr;
    clReleaseSampler_f_t clReleaseSampler = nullptr;
    clCreateImage_f_t clCreateImage = nullptr;
    clEnqueueWriteImage_f_t clEnqueueWriteImage = nullptr;
    clEnqueueReadImage_f_t clEnqueueReadImage = nullptr;
    clEnqueueWriteBufferRect_f_t clEnqueueWriteBufferRect = nullptr;
    clEnqueueReadBufferRect_f_t clEnqueueReadBufferRect = nullptr;
    clReleaseDevice_f_t clReleaseDevice = nullptr;
    clRetainDevice_f_t clRetainDevice = nullptr;
    clCreateCommandQueueWithProperties_f_t clCreateCommandQueueWithProperties = nullptr;
    clRetainCommandQueue_f_t clRetainCommandQueue = nullptr;
    clSVMAlloc_f_t clSVMAlloc = nullptr;
    clSVMFree_f_t clSVMFree = nullptr;
    clEnqueueSVMMap_f_t clEnqueueSVMMap = nullptr;
    clEnqueueSVMUnmap_f_t clEnqueueSVMUnmap = nullptr;
    clSetKernelArgSVMPointer_f_t clSetKernelArgSVMPointer = nullptr;
    clCreateSubBuffer_f_t clCreateSubBuffer = nullptr;
    clEnqueueCopyBuffer_f_t clEnqueueCopyBuffer = nullptr;
    clEnqueueCopyBufferToImage_f_t clEnqueueCopyBufferToImage = nullptr;
    clEnqueueCopyBufferRect_f_t clEnqueueCopyBufferRect = nullptr;
    clWaitForEvents_f_t clWaitForEvents = nullptr;
    clGetEventProfilingInfo_f_t clGetEventProfilingInfo = nullptr;
    clReleaseEvent_f_t clReleaseEvent = nullptr;
    clEnqueueCopyImageToBuffer_f_t clEnqueueCopyImageToBuffer = nullptr;
    clEnqueueCopyImage_f_t clEnqueueCopyImage = nullptr;
    clGetMemObjectInfo_f_t clGetMemObjectInfo = nullptr;
    clEnqueueFillBuffer_f_t clEnqueueFillBuffer = nullptr;
    clEnqueueMapBuffer_f_t clEnqueueMapBuffer = nullptr;
    clEnqueueUnmapMemObject_f_t clEnqueueUnmapMemObject = nullptr;

    void *handle = nullptr;
};

OpenCLSymbols OpenCLBackend::symbols_;
static std::once_flag opencl_symbols_load_flag;

// 实现 getSymbols 辅助函数
OpenCLSymbols *OpenCLBackend::getSymbols() {
    return &symbols_;
}

// extern "C" 包装函数
extern "C" {
cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetPlatformIDs;
    return func(num_entries, platforms, num_platforms);
}
cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetDeviceIDs;
    return func(platform, device_type, num_entries, devices, num_devices);
}
cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetDeviceInfo;
    return func(device, param_name, param_value_size, param_value, param_value_size_ret);
}
cl_context CL_API_CALL clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateContext;
    return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}
cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateCommandQueue;
    return func(context, device, properties, errcode_ret);
}
cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue command_queue) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseCommandQueue;
    return func(command_queue);
}
cl_int CL_API_CALL clReleaseContext(cl_context context) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseContext;
    return func(context);
}
cl_program CL_API_CALL clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateProgramWithSource;
    return func(context, count, strings, lengths, errcode_ret);
}
cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void(CL_CALLBACK *pfn_notify)(cl_program, void *), void *user_data) {
    auto func = mllm::OpenCLBackend::getSymbols()->clBuildProgram;
    return func(program, num_devices, device_list, options, pfn_notify, user_data);
}
cl_int CL_API_CALL clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetProgramBuildInfo;
    return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}
cl_int CL_API_CALL clReleaseProgram(cl_program program) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseProgram;
    return func(program);
}
cl_kernel CL_API_CALL clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateKernel;
    return func(program, kernel_name, errcode_ret);
}
cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseKernel;
    return func(kernel);
}
cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    auto func = mllm::OpenCLBackend::getSymbols()->clSetKernelArg;
    return func(kernel, arg_index, arg_size, arg_value);
}
cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueNDRangeKernel;
    return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}
cl_mem CL_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateBuffer;
    return func(context, flags, size, host_ptr, errcode_ret);
}
cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseMemObject;
    return func(memobj);
}
cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueWriteBuffer;
    return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueReadBuffer;
    return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
    auto func = mllm::OpenCLBackend::getSymbols()->clFinish;
    return func(command_queue);
}
cl_sampler CL_API_CALL clCreateSampler(cl_context context, cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateSampler;
    return func(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
}
cl_int CL_API_CALL clReleaseSampler(cl_sampler sampler) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseSampler;
    return func(sampler);
}
cl_mem CL_API_CALL clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateImage;
    return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}
cl_int CL_API_CALL clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t *origin, const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueWriteImage;
    return func(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t *origin, const size_t *region, size_t row_pitch, size_t slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueReadImage;
    return func(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clEnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, const size_t *buffer_origin, const size_t *host_origin, const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueWriteBufferRect;
    return func(command_queue, buffer, blocking_write, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clEnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, const size_t *buffer_origin, const size_t *host_origin, const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueReadBufferRect;
    return func(command_queue, buffer, blocking_read, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clReleaseDevice(cl_device_id device) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseDevice;
    return func(device);
}
cl_int CL_API_CALL clRetainDevice(cl_device_id device) {
    auto func = mllm::OpenCLBackend::getSymbols()->clRetainDevice;
    return func(device);
}
cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateCommandQueueWithProperties;
    return func(context, device, properties, errcode_ret);
}
cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue command_queue) {
    auto func = mllm::OpenCLBackend::getSymbols()->clRetainCommandQueue;
    return func(command_queue);
}
void *CL_API_CALL clSVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment) {
    auto func = mllm::OpenCLBackend::getSymbols()->clSVMAlloc;
    return func(context, flags, size, alignment);
}
void CL_API_CALL clSVMFree(cl_context context, void *svm_pointer) {
    auto func = mllm::OpenCLBackend::getSymbols()->clSVMFree;
    func(context, svm_pointer);
}
cl_int CL_API_CALL clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking_map, cl_map_flags map_flags, void *svm_ptr, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueSVMMap;
    return func(command_queue, blocking_map, map_flags, svm_ptr, size, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clEnqueueSVMUnmap(cl_command_queue command_queue, void *svm_ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueSVMUnmap;
    return func(command_queue, svm_ptr, num_events_in_wait_list, event_wait_list, event);
}
cl_int CL_API_CALL clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index, const void *arg_value) {
    auto func = mllm::OpenCLBackend::getSymbols()->clSetKernelArgSVMPointer;
    return func(kernel, arg_index, arg_value);
}
cl_mem CL_API_CALL clCreateSubBuffer(cl_mem mem, cl_mem_flags flags, cl_buffer_create_type type, const void *b_info, cl_int *err) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateSubBuffer;
    return func(mem, flags, type, b_info, err);
}
cl_int CL_API_CALL clEnqueueCopyBuffer(cl_command_queue q, cl_mem s, cl_mem d, size_t s_o, size_t d_o, size_t si, cl_uint e_l, const cl_event *e, cl_event *ev) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueCopyBuffer;
    return func(q, s, d, s_o, d_o, si, e_l, e, ev);
}
cl_int CL_API_CALL clEnqueueCopyBufferToImage(cl_command_queue q, cl_mem s, cl_mem d, size_t s_o, const size_t *d_o, const size_t *r, cl_uint el, const cl_event *e, cl_event *ev) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueCopyBufferToImage;
    return func(q, s, d, s_o, d_o, r, el, e, ev);
}
cl_int CL_API_CALL clEnqueueCopyBufferRect(cl_command_queue q, cl_mem s_b, cl_mem d_b, const size_t *s_o, const size_t *d_o, const size_t *r, size_t s_r_p, size_t s_s_p, size_t d_r_p, size_t d_s_p, cl_uint el, const cl_event *e, cl_event *ev) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueCopyBufferRect;
    return func(q, s_b, d_b, s_o, d_o, r, s_r_p, s_s_p, d_r_p, d_s_p, el, e, ev);
}
cl_int CL_API_CALL clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
    auto func = mllm::OpenCLBackend::getSymbols()->clWaitForEvents;
    return func(num_events, event_list);
}

cl_int CL_API_CALL clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetEventProfilingInfo;
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clReleaseEvent(cl_event event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clReleaseEvent;
    return func(event);
}

// 添加下面的两个新函数
cl_int CL_API_CALL clEnqueueCopyImageToBuffer(cl_command_queue q, cl_mem s_img, cl_mem d_buf, const size_t *s_o, const size_t *r, size_t d_o, cl_uint el, const cl_event *e, cl_event *ev) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueCopyImageToBuffer;
    if (func) {
        return func(q, s_img, d_buf, s_o, r, d_o, el, e, ev);
    }
    return CL_INVALID_OPERATION; // Or another appropriate error
}

cl_int CL_API_CALL clEnqueueCopyImage(cl_command_queue q, cl_mem s_img, cl_mem d_img, const size_t *s_o, const size_t *d_o, const size_t *r, cl_uint el, const cl_event *e, cl_event *ev) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueCopyImage;
    if (func) {
        return func(q, s_img, d_img, s_o, d_o, r, el, e, ev);
    }
    return CL_INVALID_OPERATION; // Or another appropriate error
}
cl_int CL_API_CALL clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetMemObjectInfo;
    if (func) {
        return func(memobj, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_OPERATION; // 或者返回其他合适的错误码
}
cl_program CL_API_CALL clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clCreateProgramWithBinary;
    return func(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

cl_int CL_API_CALL clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clGetProgramInfo;
    return func(program, param_name, param_value_size, param_value, param_value_size_ret);
}
cl_int CL_API_CALL clEnqueueFillBuffer(cl_command_queue command_queue,
                                       cl_mem buffer,
                                       const void *pattern,
                                       size_t pattern_size,
                                       size_t offset,
                                       size_t size,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event *event_wait_list,
                                       cl_event *event) {
    // 从 symbols_ 结构体中获取函数指针
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueFillBuffer;
    if (func) {
        // 如果成功加载，则调用真实的 OpenCL 函数
        return func(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
    }
    // 如果函数指针为空（例如在某些非常老的设备上不支持），返回一个错误码
    return CL_INVALID_OPERATION;
}
void *CL_API_CALL clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueMapBuffer;
    if (func) {
        return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
    }
    if (errcode_ret) {
        *errcode_ret = CL_INVALID_OPERATION;
    }
    return nullptr;
}

cl_int CL_API_CALL clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = mllm::OpenCLBackend::getSymbols()->clEnqueueUnmapMemObject;
    if (func) {
        return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_OPERATION;
}

} // extern "C"
#endif // MLLM_TARGET_ANDROID

std::shared_ptr<OpenCLMemoryManager> OpenCLBackend::createMemoryManager(cl_context &context, cl_device_id &device) {
#if defined(MLLM_TARGET_ANDROID)
    std::call_once(opencl_symbols_load_flag, [&]() {
        loadOpenCLSymbols();
    });
#endif
    cl_int err;
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    check_cl_error(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "No GPU found, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }
    check_cl_error(err, "clGetDeviceIDs");
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check_cl_error(err, "clCreateContext");
    return std::make_shared<OpenCLMemoryManager>(context);
}

OpenCLBackend::OpenCLBackend(const BackendConfig &config) :
    Backend() {
    mem_manager_ = createMemoryManager(context_, device_);
    cl_int err;
    // queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    queue_ = clCreateCommandQueue(context_, device_, properties, &err);
    check_cl_error(err, "clCreateCommandQueue");
    err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &this->max_image2d_width_, nullptr);
    check_cl_error(err, "clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_WIDTH");
    // 如果查询失败或返回0，可以设置一个保守的默认值
    if (this->max_image2d_width_ == 0) {
        this->max_image2d_width_ = 8192; // 一个非常保守的值
    }
    size_t extensions_size;
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensions_size);
    std::string extensions(extensions_size, ' ');
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, extensions_size, &extensions[0], nullptr);
    if (extensions.find("cl_khr_fp16") != std::string::npos) {
        this->has_fp16_support_ = true;
    } else {
        this->has_fp16_support_ = false;
    }
    if (extensions.find("cl_khr_image2d_from_buffer") != std::string::npos) {
        this->image_from_buffer_supported_ = true;
        clGetDeviceInfo(device_, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, sizeof(cl_uint), &this->image_pitch_alignment_bytes_, nullptr);
        if (this->image_pitch_alignment_bytes_ == 0) {
            this->image_pitch_alignment_bytes_ = 1;
        }
    } else {
        this->image_from_buffer_supported_ = false;
        this->image_pitch_alignment_bytes_ = 0;
    }
#if defined(MLLM_TARGET_ANDROID)
    kernel_root_path_ = get_executable_dir();
#else
    kernel_root_path_ = get_kernel_path(__FILE__, ".");
#endif
    const std::string convert_kernel_path = kernel_root_path_ + "/kernel/convert_fp.cl";
    std::string build_options = "";
    if (this->has_fp16_support_) {
        build_options += " -DSUPPORTS_FP16";
        // std::cout << "OpenCL supports cl_khr_fp16." << std::endl;
    }
    cl_program convert_program = getProgram(convert_kernel_path, build_options);
    if (this->has_fp16_support_) {
        kernel_fp32_to_fp16_buffer_ = clCreateKernel(convert_program, "convert_fp32_to_fp16_buffer_ext", &err);
        check_cl_error(err, "CreateKernel: convert_fp32_to_fp16_buffer_ext");
        kernel_fp16_to_fp32_buffer_ = clCreateKernel(convert_program, "convert_fp16_to_fp32_buffer_ext", &err);
        check_cl_error(err, "CreateKernel: convert_fp16_to_fp32_buffer_ext");
        kernel_fp32_to_fp16_image_ = clCreateKernel(convert_program, "convert_fp32_to_fp16_image2d", &err);
        check_cl_error(err, "CreateKernel: convert_fp32_to_fp16_image2d");
        kernel_fp16_to_fp32_image_ = clCreateKernel(convert_program, "convert_fp16_to_fp32_image2d", &err);
        check_cl_error(err, "CreateKernel: convert_fp16_to_fp32_image2d");
    } else {
        kernel_fp32_to_fp16_buffer_ = clCreateKernel(convert_program, "convert_fp32_to_fp16_buffer_compat", &err);
        check_cl_error(err, "CreateKernel: convert_fp32_to_fp16_buffer_compat");
        kernel_fp16_to_fp32_buffer_ = clCreateKernel(convert_program, "convert_fp16_to_fp32_buffer_compat", &err);
        check_cl_error(err, "CreateKernel: convert_fp16_to_fp32_buffer_compat");
    }
    sampler_ = clCreateSampler(context_, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler in Backend");
    this->type_ = MLLM_OPENCL;
    registerOps();
}

OpenCLBackend::~OpenCLBackend() {
    if (mem_manager_) {
        mem_manager_.reset();
    }
    if (kernel_fp32_to_fp16_buffer_) clReleaseKernel(kernel_fp32_to_fp16_buffer_);
    if (kernel_fp16_to_fp32_buffer_) clReleaseKernel(kernel_fp16_to_fp32_buffer_);
    if (kernel_fp32_to_fp16_image_) clReleaseKernel(kernel_fp32_to_fp16_image_);
    if (kernel_fp16_to_fp32_image_) clReleaseKernel(kernel_fp16_to_fp32_image_);
    if (sampler_) clReleaseSampler(sampler_);
    for (auto const &[key, program] : program_cache_) {
        if (program) {
            clReleaseProgram(program);
        }
    }
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
#if defined(MLLM_TARGET_ANDROID)
    if (symbols_.handle) {
        dlclose(symbols_.handle);
        symbols_.handle = nullptr;
    }
#endif
}

void OpenCLBackend::finishQueue() {
    if (queue_) {
        clFinish(queue_);
    }
}
cl_program OpenCLBackend::getProgram(const std::string &program_name, const std::string &build_options) {
    // 使用 program_name 和 build_options 创建唯一的键，用于在内存中缓存 program 对象
    std::string cache_key = program_name + build_options;
    auto it = program_cache_.find(cache_key);
    if (it != program_cache_.end()) {
        return it->second;
    }

    // 1. 构建源文件和缓存文件的完整路径
    std::filesystem::path source_path(kernel_root_path_);
    source_path /= program_name;

    // a. 创建缓存目录 (例如: opencl/kernel/cache)
    std::filesystem::path cache_dir = source_path.parent_path() / "cache";
    std::error_code ec;
    if (!std::filesystem::exists(cache_dir, ec)) {
        std::filesystem::create_directories(cache_dir, ec);
    }

    // b. 生成一个稳定且唯一的缓存文件名 (例如: kernel_add_cl_xxxxx.bin)
    //    将路径中的 '/' 和 '.' 替换为 '_'，并附上编译选项的哈希值
    std::string bin_file_name = program_name;
    std::replace(bin_file_name.begin(), bin_file_name.end(), '/', '_');
    std::replace(bin_file_name.begin(), bin_file_name.end(), '.', '_');
    std::hash<std::string> hasher;
    std::string options_hash = std::to_string(hasher(build_options));
    std::filesystem::path bin_path = cache_dir / (bin_file_name + "_" + options_hash + ".bin");

    cl_program program = nullptr;
    cl_int err;

    // 2. 尝试从二进制缓存文件加载程序
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (bin_file.is_open()) {
        bin_file.seekg(0, std::ios::end);
        size_t bin_size = bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);
        std::vector<unsigned char> bin_data(bin_size);
        bin_file.read(reinterpret_cast<char *>(bin_data.data()), bin_size);
        bin_file.close();

        const unsigned char *bin_ptr = bin_data.data();
        cl_int binary_status;
        program = clCreateProgramWithBinary(context_, 1, &device_, &bin_size, &bin_ptr, &binary_status, &err);

        if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
            // ===== [ 核心修正点 ] =====
            // 即使从二进制加载，也需要Build来使其对设备可执行
            err = clBuildProgram(program, 1, &device_, build_options.c_str(), nullptr, nullptr);
            if (err != CL_SUCCESS) {
                // 如果Build失败，说明缓存可能已损坏或不兼容，需要回退到从源码编译
                if (program) clReleaseProgram(program); // 释放无效的 program 对象
                program = nullptr;                      // 将 program 置空，以便后续逻辑能从源码重新编译
            }
            // ===== [ 修正结束 ] =====
        } else {
            // 如果加载失败，清空 program 对象
            if (program) clReleaseProgram(program);
            program = nullptr;
        }
    }

    // 3. 如果从缓存加载失败 (program == nullptr)，则从源码编译
    if (program == nullptr) {
        std::string kernel_source = load_file_contents(source_path.c_str());
        const char *source_ptr = kernel_source.c_str();
        size_t source_len = kernel_source.length();
        program = clCreateProgramWithSource(context_, 1, &source_ptr, &source_len, &err);
        check_cl_error(err, "clCreateProgramWithSource for " + program_name);

        err = clBuildProgram(program, 1, &device_, build_options.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::string error_msg = "Kernel build error for " + program_name + ":\n" + log.data();
            if (program) clReleaseProgram(program);
            throw std::runtime_error(error_msg);
        }

        // 4. 编译成功后，获取二进制码并保存到缓存文件
        size_t binary_size;
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
        if (binary_size > 0) {
            std::vector<unsigned char> binary_data(binary_size);
            unsigned char *bin_ptr = binary_data.data();
            // 注意：clGetProgramInfo的第三个参数应该是`sizeof(unsigned char*)`
            clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin_ptr, nullptr);

            std::ofstream out_bin_file(bin_path, std::ios::binary);
            if (out_bin_file.is_open()) {
                out_bin_file.write(reinterpret_cast<char *>(binary_data.data()), binary_size);
                out_bin_file.close();
            } else {
                std::cerr << "Warning: Could not write to kernel cache file: " << bin_path << std::endl;
            }
        }
    }

    // 将最终获取的 program 对象存入内存缓存
    program_cache_[cache_key] = program;
    return program;
}
void OpenCLBackend::alloc_device(DeviceMemory &mem, DataType dtype) {
    if (context_ == nullptr) throw std::runtime_error("OpenCL context is not initialized.");
    cl_int err;
    switch (mem.type) {
    case MEM_TYPE_BUFFER: {
        if (mem.image_width > 0 && mem.image_height > 0 && image_from_buffer_supported_) {
            const size_t pixel_width = mem.image_width;
            const cl_uint pitch_alignment = image_pitch_alignment_bytes_;
            size_t row_pitch = pixel_width * 4 * sizeof(float);
            if (pitch_alignment > 0 && row_pitch % pitch_alignment != 0) {
                row_pitch = (row_pitch + pitch_alignment - 1) / pitch_alignment * pitch_alignment;
            }
            const size_t padded_buffer_size = mem.image_height * row_pitch;
            mem.size_in_bytes = padded_buffer_size;
            mem.image_row_pitch_in_bytes = row_pitch;
            mem_manager_->alloc(&mem.handle, mem.size_in_bytes, 0);
            if (mem.handle == nullptr) {
                throw std::runtime_error("OpenCLMemoryManager failed to allocate buffer.");
            }
        } else if (mem.size_in_bytes > 0) {
            mem_manager_->alloc(&mem.handle, mem.size_in_bytes, 0);
            if (mem.handle == nullptr) {
                throw std::runtime_error("OpenCLMemoryManager failed to allocate buffer.");
            }
        }
        break;
    }
    case MEM_TYPE_IMAGE_2D: {
        cl_image_format format = {CL_RGBA};
        switch (dtype) {
        case MLLM_TYPE_F32:
            format.image_channel_data_type = CL_FLOAT;
            break;
        case MLLM_TYPE_F16:
            format.image_channel_data_type = CL_HALF_FLOAT;
            break;
        default:
            throw std::runtime_error("Unsupported data type for Image2D creation.");
        }
        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = mem.image_width;
        desc.image_height = mem.image_height;
        if (desc.image_width > 0) {
            mem.handle = clCreateImage(context_, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);
            check_cl_error(err, "clCreateImage");
        }
        break;
    }
    default: throw std::runtime_error("Unsupported device memory type for OpenCL.");
    }
}

void OpenCLBackend::free_device(DeviceMemory &mem) {
    if (mem.handle != nullptr) {
        switch (mem.type) {
        case MEM_TYPE_BUFFER:
            mem_manager_->free(mem.handle);
            break;
        case MEM_TYPE_IMAGE_2D:
            clReleaseMemObject(static_cast<cl_mem>(mem.handle));
            break;
        default:
            // 对于其他类型，也许也应该直接释放
            clReleaseMemObject(static_cast<cl_mem>(mem.handle));
            break;
        }
        mem.handle = nullptr;
    }
}

void OpenCLBackend::copy_from_host(const DeviceMemory &dest, const void *src) {
    if (dest.handle == nullptr || src == nullptr) return;
    cl_mem dest_handle = static_cast<cl_mem>(dest.handle);
    switch (dest.type) {
    case MEM_TYPE_BUFFER: {
        if (dest.image_row_pitch_in_bytes > 0 && dest.image_height > 0) {
            const size_t buffer_origin[3] = {0, 0, 0};
            const size_t host_origin[3] = {0, 0, 0};
            const size_t region_in_bytes[3] = {
                dest.image_width * 4 * sizeof(float),
                dest.image_height,
                1};
            clEnqueueWriteBufferRect(
                queue_, dest_handle, CL_TRUE, buffer_origin, host_origin,
                region_in_bytes, dest.image_row_pitch_in_bytes, 0,
                dest.image_width * 4 * sizeof(float), 0, src, 0, nullptr, nullptr);
        } else {
            clEnqueueWriteBuffer(queue_, dest_handle, CL_TRUE, 0, dest.size_in_bytes, src, 0, nullptr, nullptr);
        }
        break;
    }
    case MEM_TYPE_IMAGE_2D: {
        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {dest.image_width, dest.image_height, 1};
        clEnqueueWriteImage(queue_, dest_handle, CL_TRUE, origin, region, 0, 0, src, 0, nullptr, nullptr);
        break;
    }
    default: throw std::runtime_error("Unsupported copy for this memory type.");
    }
}

void OpenCLBackend::copy_to_host(void *dest, const DeviceMemory &src) {
    if (dest == nullptr || src.handle == nullptr) return;
    cl_mem src_handle = static_cast<cl_mem>(src.handle);
    switch (src.type) {
    case MEM_TYPE_BUFFER:
        clEnqueueReadBuffer(queue_, src_handle, CL_TRUE, 0, src.size_in_bytes, dest, 0, nullptr, nullptr);
        break;
    case MEM_TYPE_IMAGE_2D: {
        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {src.image_width, src.image_height, 1};
        clEnqueueReadImage(queue_, src_handle, CL_TRUE, origin, region, 0, 0, dest, 0, nullptr, nullptr);
        break;
    }
    default: throw std::runtime_error("Unsupported copy for this memory type.");
    }
}

cl_mem OpenCLBackend::get_cl_mem(const Tensor &tensor) const {
    if (tensor.backend() != this) throw std::runtime_error("Tensor is not on this backend.");
    const auto &mem = tensor.device_memory();
    if (mem.handle == nullptr) throw std::runtime_error("Tensor CL handle is null.");
    return static_cast<cl_mem>(mem.handle);
}

Op *OpenCLBackend::opCreate(const OpParam &op_param, std::string name, int threadCount) {
    OpType type = (OpType)op_param.find("type")->second;
    auto it = op_creator_map_.find(type);
    if (it == op_creator_map_.end()) {
        return nullptr;
    }
    return it->second->create(op_param, this, name, threadCount);
}

TensorFunction *OpenCLBackend::funcCreate(TensorFuncType type) {
    throw std::runtime_error("funcCreate not implemented for OpenCLBackend");
}

void OpenCLBackend::registerOps() {
    op_creator_map_[F_ADD] = std::make_shared<OpenCLAddOpCreator>();
    op_creator_map_[F_TTADD] = std::make_shared<OpenCLAddTwoOpCreator>();
    op_creator_map_[F_SUB] = std::make_shared<OpenCLSubOpCreator>();
    op_creator_map_[F_TTSUB] = std::make_shared<OpenCLSubTwoOpCreator>();
    op_creator_map_[F_MUL] = std::make_shared<OpenCLMulOpCreator>();
    op_creator_map_[F_TTMUL] = std::make_shared<OpenCLMulTwoOpCreator>();
    op_creator_map_[F_DIV] = std::make_shared<OpenCLDivOpCreator>();
    op_creator_map_[F_DIVINT] = std::make_shared<OpenCLDivIntOpCreator>();
    op_creator_map_[F_TTDIV] = std::make_shared<OpenCLDivTwoOpCreator>();
    op_creator_map_[F_MM] = std::make_shared<OpenCLMatmulOpCreator>();
    op_creator_map_[LINEAR] = std::make_shared<OpenCLLinearOpCreator>();
    op_creator_map_[F_TRANPOSE] = std::make_shared<OpenCLTransposeOpCreator>();
    op_creator_map_[SOFTMAX] = std::make_shared<OpenCLSoftMaxOpCreator>();
    op_creator_map_[RMSNORM] = std::make_shared<OpenCLRMSNormOpCreator>();
    op_creator_map_[EMBEDDING] = std::make_shared<OpenCLEmbeddingOpCreator>();
    op_creator_map_[SILU] = std::make_shared<OpenCLSiLUOpCreator>();
    op_creator_map_[F_VIEW] = std::make_shared<OpenCLViewOpCreator>();
    op_creator_map_[KVCACHE] = std::make_shared<OpenCLKVCacheOpCreator>();
    op_creator_map_[ROPE] = std::make_shared<OpenCLRoPEOpCreator>();
    op_creator_map_[F_CLIP] = std::make_shared<OpenCLClipOpCreator>();
    op_creator_map_[F_FA2] = std::make_shared<OpenCLFlashAttentionOpCreator>();
    op_creator_map_[F_SPLIT] = std::make_shared<OpenCLSplitOpCreator>();
    op_creator_map_[F_TOPK] = std::make_shared<OpenCLTopkOpCreator>();
    op_creator_map_[F_SUM] = std::make_shared<OpenCLSumOpCreator>();
    op_creator_map_[F_LIKE] = std::make_shared<OpenCLLikeOpCreator>();
    op_creator_map_[F_CLIPTENSOR] = std::make_shared<OpenCLClipTensorOpCreator>();
    op_creator_map_[F_SCATTERRADD] = std::make_shared<OpenCLScatterAddOpCreator>();
    op_creator_map_[F_ARGSORT] = std::make_shared<OpenCLArgSortOpCreator>();
    op_creator_map_[F_BINCOUNT] = std::make_shared<OpenCLBinCountOpCreator>();
}

void OpenCLBackend::registerFuncs() {
    std::cout << "OpenCLBackend funcs is abanded." << std::endl;
}

void OpenCLBackend::convert_fp_data(Tensor *src, Tensor *dest) {
    if (src->device() != MLLM_OPENCL || dest->device() != MLLM_OPENCL) {
        throw std::runtime_error("Type conversion on GPU requires both tensors to be on OpenCL backend.");
    }
    auto &src_mem = src->device_memory();
    auto &dest_mem = dest->device_memory();

    if (src_mem.type == MEM_TYPE_BUFFER) {
        if (dest_mem.type != MEM_TYPE_BUFFER) {
            throw std::runtime_error("Destination must be a Buffer for Buffer conversion.");
        }
        cl_kernel kernel_to_use = nullptr;

        // 根据转换类型选择内核
        if (src->dtype() == MLLM_TYPE_F32 && dest->dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp32_to_fp16_buffer_;
        } else if (src->dtype() == MLLM_TYPE_F16 && dest->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp16_to_fp32_buffer_;
        } else {
            if (src->dtype() == dest->dtype()) return;
            throw std::runtime_error("Unsupported Buffer conversion types.");
        }

        cl_mem src_buf = get_cl_mem(*src);
        cl_mem dest_buf = get_cl_mem(*dest);
        const int count = src->count();

        // ✨ **关键修正: 明确控制工作组大小**
        if (count > 0) {
            clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_buf);
            clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &dest_buf);
            clSetKernelArg(kernel_to_use, 2, sizeof(int), &count);

            // 1. 定义一个标准的工作组大小
            const size_t local_work_size = 256;

            // 2. 手动计算向上补齐的全局工作大小
            const size_t global_work_size = ((count + local_work_size - 1) / local_work_size) * local_work_size;

            // 3. 使用明确的 local 和 global size 启动内核
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(queue_, kernel_to_use, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, &event);
            this->addProfilingEvent("convert_fp_data", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for type conversion");
        }

    } else if (src_mem.type == MEM_TYPE_IMAGE_2D) {
        if (dest_mem.type != MEM_TYPE_IMAGE_2D) {
            throw std::runtime_error("Destination must be an Image for Image conversion.");
        }
        cl_kernel kernel_to_use = nullptr;
        if (src->dtype() == MLLM_TYPE_F32 && dest->dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp32_to_fp16_image_;
        } else if (src->dtype() == MLLM_TYPE_F16 && dest->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp16_to_fp32_image_;
        } else {
            if (src->dtype() == dest->dtype()) return;
            throw std::runtime_error("Unsupported Image conversion types.");
        }
        if (!kernel_to_use) {
            throw std::runtime_error("Image conversion kernel is not available. This may be due to lack of FP16 hardware support.");
        }

        cl_mem src_img = get_cl_mem(*src);
        cl_mem dest_img = get_cl_mem(*dest);
        const int width = src_mem.image_width;
        const int height = src_mem.image_height;

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &src_img);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &dest_img);
        clSetKernelArg(kernel_to_use, 3, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &height);

        // 对于2D图像，通常让驱动选择最佳工作组大小是安全的，但也可以明确指定
        const size_t local_ws[2] = {16, 16};
        const size_t global_ws[2] = {
            ((size_t)width + local_ws[0] - 1) / local_ws[0] * local_ws[0],
            ((size_t)height + local_ws[1] - 1) / local_ws[1] * local_ws[1]};

        clEnqueueNDRangeKernel(queue_, kernel_to_use, 2, nullptr, global_ws, local_ws, 0, nullptr, nullptr);
    }
}
bool OpenCLBackend::load_from_file(Tensor *tensor, ParamLoader *loader) {
    // 1. 从加载器获取张量的元数据和文件句柄
    ParamMetadata metadata = loader->getParamMetadata(tensor->name());
    FILE *fp = loader->getInputStream();
    if (metadata.size == 0) {
        return true; // 无需加载
    }

    // 2. 检查张量设备内存是否已就绪
    if (tensor->device_memory().handle == nullptr) {
        // 如果内存未分配，此快速路径无法工作。
        // 这通常意味着调用顺序有问题，load() 之前应先 alloc()。
        return false;
    }

    // 3. 获取OpenCL对象
    cl_command_queue queue = this->getQueue();
    cl_mem buffer = this->get_cl_mem(*tensor);

    // 4. 将GPU缓冲区映射到主机地址空间 (阻塞式，写入模式)
    cl_int err;
    void *mapped_ptr = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, metadata.size, 0, nullptr, nullptr, &err);
    check_cl_error(err, "OpenCLBackend::load_from_file clEnqueueMapBuffer");
    if (mapped_ptr == nullptr) {
        fprintf(stderr, "Error: Failed to map OpenCL buffer for tensor '%s'.\n", tensor->name().c_str());
        return false;
    }

    // 5. 将文件指针移动到权重数据的起始位置，并直接读入映射后的内存
    fseek(fp, metadata.offset, SEEK_SET);
    size_t read_size = fread(mapped_ptr, 1, metadata.size, fp);
    if (read_size != metadata.size) {
        fprintf(stderr, "Error: File read failed for tensor '%s'. Expected %llu, got %zu.\n", tensor->name().c_str(), metadata.size, read_size);
        // 出错也要确保解映射
        clEnqueueUnmapMemObject(queue, buffer, mapped_ptr, 0, nullptr, nullptr);
        clFinish(queue); // 等待命令完成
        return false;
    }

    // 6. 解除内存映射，并将控制权交还GPU
    cl_event unmap_event;
    err = clEnqueueUnmapMemObject(queue, buffer, mapped_ptr, 0, nullptr, &unmap_event);
    check_cl_error(err, "OpenCLBackend::load_from_file clEnqueueUnmapMemObject");

    // 7. 阻塞等待解映射操作完成，确保数据对GPU可见
    clWaitForEvents(1, &unmap_event);
    clReleaseEvent(unmap_event);

    // 8. 数据已在设备上，主机指针应失效，防止误用
    tensor->forceResetHostPointer(nullptr);

    return true; // 表示加载已由本函数成功处理
}

void registerOpenCLBackendCreator() {
    InsertBackendCreatorMap(MLLM_OPENCL, std::make_shared<OpenCLBackendCreator>());
}

std::vector<Tensor> OpenCLBackend::runLayer(Layer *layer, std::vector<Tensor> inputs, int N) {
    throw std::runtime_error("runLayer not implemented for OpenCLBackend");
}

std::vector<Tensor> OpenCLBackend::runOp(Op *op, std::vector<Tensor> inputs, std::vector<std::string> out_names, bool in_place) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    static map<string, shared_ptr<Tensor>> empty_activation_tensors;
    map<string, shared_ptr<Tensor>> &activation_tensors = module ? module->activation_tensors : empty_activation_tensors;
    if (module && module->doTrace) {
        if (module->tracedFlag) {
            vector<Tensor> results = {};
            for (auto &name : out_names) results.push_back(*activation_tensors[name]);
            return results;
        }
        for (auto &input : inputs) {
            if (input.shouldInGraphs() && activation_tensors.find(input.name()) == activation_tensors.end()) {
                activation_tensors[input.name()] = std::make_shared<Tensor>(op->backend());
                activation_tensors[input.name()]->setName(input.name());
                activation_tensors[input.name()]->setModule(module);
            }
        }
        for (const auto &out_name : out_names) {
            if (activation_tensors.find(out_name) == activation_tensors.end()) {
                activation_tensors[out_name] = std::make_shared<Tensor>(op->backend());
                activation_tensors[out_name]->setName(out_name);
                activation_tensors[out_name]->setModule(module);
            }
        }
        vector<shared_ptr<Tensor>> inPtrs;
        for (auto &input : inputs) {
            inPtrs.push_back(input.shouldInGraphs() ? activation_tensors[input.name()] :
                                                      std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
        }
        vector<shared_ptr<Tensor>> outPtrs = {};
        for (auto &name : out_names) outPtrs.push_back(activation_tensors[name]);
        op->setUp(inPtrs, outPtrs);
        vector<Tensor> results = {};
        for (auto &name : out_names) results.push_back(*activation_tensors[name]);
        return results;
    }
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
    }
    vector<shared_ptr<Tensor>> out_tensors;
    if (!in_place) {
        for (const auto &out_name : out_names) {
            auto out_tensor = std::make_shared<Tensor>(op->backend());
            out_tensor->setName(out_name);
            out_tensors.push_back(out_tensor);
        }
    } else {
        for (size_t i = 0; i < input_tensors.size() && i < out_names.size(); ++i) {
            input_tensors[i]->setName(out_names[i]);
            out_tensors.push_back(input_tensors[i]);
        }
    }
    op->reshape(input_tensors, out_tensors);
    op->setUp(input_tensors, out_tensors);
    op->execute(input_tensors, out_tensors);
    vector<Tensor> results;
    for (const auto &out_tensor : out_tensors) {
        results.push_back(*out_tensor);
#ifdef DEBUGSAVETENSOR
        out_tensor->cpu();
        if (out_tensor->dtype() == MLLM_TYPE_F32) {
            out_tensor->saveData<float>();
        }
        if (out_tensor->dtype() == MLLM_TYPE_F16) {
            out_tensor->saveData<mllm_fp16_t>();
        }
        out_tensor->cl();
#endif
    }
    return results;
}

std::vector<Tensor> OpenCLBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    if (Module::llm_model_ptr && (Module::llm_model_ptr->doLoad || Module::llm_model_ptr->doChangeBn)) {
        auto outputs = module->Forward(inputs, args);
        return outputs;
    }
    uint64_t time_start, time_end;
    bool ouilter_flag = (inputs[0].ttype() == TensorType::INPUT_TENSOR);
    if (ouilter_flag) {
        for (int i = 0; i < inputs.size(); i++) {
            auto &input = inputs[i];
            input.setModule(module);
            input.setTtype(TensorType::NORMAL_TENSOR);
        }
        Module::llm_model_ptr = module;
        if (module->prefilling_token_size_ == 0) {
            module->prefilling_token_size_ = inputs[0].sequence() * inputs[0].batch();
        } else if (module->decoding_token_size_ == 0) {
            module->decoding_token_size_ = inputs[0].sequence() * inputs[0].batch();
        }
        time_start = mllm_time_us();
        // exe_times.clear();
    }
    auto output = module->Forward(inputs, args);
    if (ouilter_flag) {
        this->finishQueue();
        time_end = mllm_time_us();
        double inference_time_ = (time_end - time_start) / 1000.0F;
#ifdef DEBUGOPTIME
        this->reportProfilingResult();
        std::cout << "One token total inference time: " << inference_time_ << " ms" << std::endl;
#endif
        module->inference_times_.push_back(inference_time_);
    }
    return output;
}

#if defined(MLLM_TARGET_ANDROID)
void OpenCLBackend::loadOpenCLSymbols() {
    static const std::vector<std::string> android_paths = {
        "libOpenCL.so",
        "libGLES_mali.so",
        "libmali.so",
        "libOpenCL-pixel.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/system/vendor/lib64/egl/libGLES_mali.so",
        "/system/lib64/egl/libGLES_mali.so"};

    for (const auto &path : android_paths) {
        symbols_.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (symbols_.handle) break;
    }

    if (!symbols_.handle) {
        throw std::runtime_error("Failed to load OpenCL library on Android");
    }

#define LOAD_FUNC(name)                                                                       \
    symbols_.name = reinterpret_cast<decltype(symbols_.name)>(dlsym(symbols_.handle, #name)); \
    if (!symbols_.name) {                                                                     \
        std::cerr << "Failed to load: " << #name << std::endl;                                \
    }

    LOAD_FUNC(clGetPlatformIDs);
    LOAD_FUNC(clGetDeviceIDs);
    LOAD_FUNC(clGetDeviceInfo);
    LOAD_FUNC(clCreateContext);
    LOAD_FUNC(clCreateCommandQueue);
    LOAD_FUNC(clReleaseCommandQueue);
    LOAD_FUNC(clReleaseContext);
    LOAD_FUNC(clCreateProgramWithSource);
    LOAD_FUNC(clBuildProgram);
    LOAD_FUNC(clGetProgramBuildInfo);
    LOAD_FUNC(clCreateProgramWithBinary);
    LOAD_FUNC(clGetProgramInfo);
    LOAD_FUNC(clReleaseProgram);
    LOAD_FUNC(clCreateKernel);
    LOAD_FUNC(clReleaseKernel);
    LOAD_FUNC(clSetKernelArg);
    LOAD_FUNC(clEnqueueNDRangeKernel);
    LOAD_FUNC(clCreateBuffer);
    LOAD_FUNC(clReleaseMemObject);
    LOAD_FUNC(clEnqueueWriteBuffer);
    LOAD_FUNC(clEnqueueReadBuffer);
    LOAD_FUNC(clFinish);
    LOAD_FUNC(clCreateSampler);
    LOAD_FUNC(clReleaseSampler);
    LOAD_FUNC(clCreateImage);
    LOAD_FUNC(clEnqueueWriteImage);
    LOAD_FUNC(clEnqueueReadImage);
    LOAD_FUNC(clEnqueueWriteBufferRect);
    LOAD_FUNC(clEnqueueReadBufferRect);
    LOAD_FUNC(clReleaseDevice);
    LOAD_FUNC(clRetainDevice);
    LOAD_FUNC(clCreateCommandQueueWithProperties);
    LOAD_FUNC(clRetainCommandQueue);
    LOAD_FUNC(clSVMAlloc);
    LOAD_FUNC(clSVMFree);
    LOAD_FUNC(clEnqueueSVMMap);
    LOAD_FUNC(clEnqueueSVMUnmap);
    LOAD_FUNC(clSetKernelArgSVMPointer);
    LOAD_FUNC(clCreateSubBuffer);
    LOAD_FUNC(clEnqueueCopyBuffer);
    LOAD_FUNC(clEnqueueCopyBufferToImage);
    LOAD_FUNC(clEnqueueCopyBufferRect);
    LOAD_FUNC(clWaitForEvents);
    LOAD_FUNC(clGetEventProfilingInfo);
    LOAD_FUNC(clReleaseEvent);
    LOAD_FUNC(clEnqueueCopyImageToBuffer);
    LOAD_FUNC(clEnqueueCopyImage);
    LOAD_FUNC(clEnqueueCopyBufferRect);
    LOAD_FUNC(clGetMemObjectInfo);
    LOAD_FUNC(clEnqueueMapBuffer);
    LOAD_FUNC(clEnqueueUnmapMemObject);

#undef LOAD_FUNC
}
#endif

void OpenCLBackend::addProfilingEvent(const std::string &op_name, cl_event event) {
    profiling_events_.push_back(event);
    event_op_names_[event] = op_name;
}

void OpenCLBackend::reportProfilingResult() {
    if (profiling_events_.empty()) {
        return;
    }

    clWaitForEvents(profiling_events_.size(), profiling_events_.data());

    std::cout << "--- OpenCL Kernel Profiling Report ---" << std::endl;
    double total_duration_ms = 0.0;
    for (cl_event event : profiling_events_) {
        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);

        double duration_ms = (end_time - start_time) / 1000000.0;
        std::cout << "OpenCL Operator [" << event_op_names_[event] << "] took " << duration_ms << " ms" << std::endl;
        total_duration_ms += duration_ms;
        clReleaseEvent(event);
    }
    std::cout << "---- Total Duration: " << total_duration_ms << " ms ---" << std::endl;

    // 清空
    profiling_events_.clear();
    event_op_names_.clear();
}
} // namespace mllm