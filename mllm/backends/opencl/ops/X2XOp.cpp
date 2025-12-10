// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "X2XOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLX2XOp::OpenCLX2XOp(const aops::X2XOpOptions& options) : aops::X2XOp(options) {}

void OpenCLX2XOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // FIXME: when input is a sliced tensor, the storage will be wrong
  return BaseOp::setup(inputs, outputs);
}

void OpenCLX2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto& output = outputs[0];

  if (input.device() == kOpenCL && output.device() == kOpenCL) {
    // If both input and output are on OpenCL device
    // Retain the buffer to ensure proper reference counting
    cl_mem src_buffer = static_cast<cl_mem>(input.impl()->storage()->ptr_);
    OpenCLLoader::instance().clRetainMemObject(src_buffer);
    output.impl()->storage()->ptr_ = src_buffer;
  } else if (input.device() == kOpenCL && output.device() == kCPU) {
    // Calculate data size in bytes
    size_t data_size = input.bytes();

    // Get OpenCL runtime
    auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

    // Get the OpenCL buffer from input tensor's storage
    auto cl_buffer = cl::Buffer(static_cast<cl_mem>(input.impl()->storage()->ptr_), true);

    // Get output data pointer
    void* dst_data = output.ptr<void>();

    // Read data from OpenCL buffer to host (CPU) memory
    auto error = runtime->commandQueue().enqueueReadBuffer(cl_buffer,
                                                           CL_TRUE,    // blocking read
                                                           0,          // offset
                                                           data_size,  // size
                                                           dst_data    // pointer to host memory
    );

    if (error != CL_SUCCESS) { MLLM_ERROR("Failed to read data from OpenCL buffer, error code: {}", error); }

    // Wait for the command to finish
    runtime->commandQueue().finish();
    return;
  } else if (input.device() == kCPU && output.device() == kOpenCL) {
    size_t data_size = input.bytes();
    auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
    void* src_data = input.ptr<void>();
    cl_mem cl_buffer = (cl_mem)output.ptr<void>();

    cl_int error = OpenCLLoader::instance().clEnqueueWriteBuffer(runtime->commandQueue()(), cl_buffer, CL_TRUE, 0, data_size,
                                                                 src_data, 0, nullptr, nullptr);

    if (error != CL_SUCCESS) { MLLM_ERROR("Failed to write data to OpenCL buffer, error code: {}", error); }
    return;
  }

  MLLM_ERROR("OpenCLX2XOp only supports transform between CPU and OpenCL.\n");
}

}  // namespace mllm::opencl