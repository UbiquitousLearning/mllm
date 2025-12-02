// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "CL/opencl.hpp"

namespace mllm::opencl {

enum GpuType { MALI = 0, ADRENO = 1, RADEON = 2, INTEL = 3, OTHER = 4 };

struct KernelPool {
  uint64_t max_work_group_size_;
  std::queue<std::shared_ptr<cl::Kernel>> recycle_;
};

class KernelWrap {
 public:
  KernelWrap(std::shared_ptr<cl::Kernel> k, KernelPool* recycle) : kernel_(std::move(k)), recycle_(recycle) {}
  ~KernelWrap() {
    if (nullptr != recycle_) { recycle_->recycle_.push(kernel_); }
  }
  cl::Kernel& get() { return *kernel_; }

 private:
  KernelPool* recycle_;
  std::shared_ptr<cl::Kernel> kernel_;
};

class OpenCLRuntime {
 public:
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime&) = delete;
  OpenCLRuntime& operator=(const OpenCLRuntime&) = delete;

  static OpenCLRuntime* get();

  cl::Context& context();
  cl::CommandQueue& commandQueue();
  [[nodiscard]] const std::vector<cl::Device>& getDevices() const;

  [[nodiscard]] bool isSupportedFP16() const;
  [[nodiscard]] bool isSupportedDotInt8() const;
  [[nodiscard]] bool isSupportedDotAccInt8() const;

  [[nodiscard]] uint64_t deviceGlobalMemeryCacheSize() const;
  [[nodiscard]] uint32_t deviceComputeUnits() const;
  [[nodiscard]] uint32_t maxWorkGroupSize() const;
  GpuType getGpuType();

  std::shared_ptr<KernelWrap> buildKernel(const std::string& programName, const std::string& kernelName,
                                          const std::set<std::string>& buildOptions);

 private:
  OpenCLRuntime();
  bool loadProgram(const std::string& programName, cl::Program* program);
  bool buildProgram(const std::string& buildOptionsStr, cl::Program* program);
  bool getDeviceSupportsExtension(const cl::Device& device, const char* extensionName);

 private:
  std::shared_ptr<cl::Context> context_;
  std::vector<cl::Device> devices_;
  std::shared_ptr<cl::CommandQueue> command_queue_;

  struct ProgramWithKernel {
    cl::Program program;
    std::map<std::string, KernelPool> kernels;
  };
  std::map<std::pair<std::string, std::string>, ProgramWithKernel> build_program_map_;
  std::mutex build_program_mutex_;

  uint64_t gpu_global_memery_cache_size_;
  uint32_t gpu_compute_units_;
  uint64_t max_mem_alloc_size_;
  size_t max_work_group_size_;

  bool is_supported_fp16_ = false;
  bool support_dot_int8_ = false;
  bool support_dot_acc_int8_ = false;
  GpuType gpu_type_;
  float cl_version_ = 1.0f;
  std::string device_name_;
  std::string default_build_params_;
};

}  // namespace mllm::opencl
