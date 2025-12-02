// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "OpenCLRuntime.hpp"
#include "OpenCLLoader.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"
#include <vector>
#include <string>
#include <fstream>

namespace mllm::opencl {

OpenCLRuntime* OpenCLRuntime::get() {
  static OpenCLRuntime runtime;
  return &runtime;
}

OpenCLRuntime::OpenCLRuntime() {
  if (!OpenCLLoader::instance().loadOpenCLDynLib()) {
    MLLM_ERROR_EXIT(LogLevel::kError, "Failed to load OpenCL dynamic library.");
  }
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    MLLM_ERROR("OpenCL platforms not found!\n");
    return;
  }

  cl::Platform platform = platforms[0];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.empty()) {
    MLLM_ERROR("OpenCL devices not found!\n");
    return;
  }

  devices_ = {devices[0]};
  context_ = std::make_shared<cl::Context>(devices_);
  command_queue_ = std::make_shared<cl::CommandQueue>(*context_, devices_[0]);

  devices_[0].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &gpu_global_memery_cache_size_);
  devices_[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &gpu_compute_units_);
  devices_[0].getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_alloc_size_);
  devices_[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size_);
  devices_[0].getInfo(CL_DEVICE_NAME, &device_name_);

  std::string ext;
  devices_[0].getInfo(CL_DEVICE_EXTENSIONS, &ext);

  is_supported_fp16_ = ext.find("cl_khr_fp16") != std::string::npos;
  support_dot_int8_ = ext.find("cl_arm_integer_dot_product_int8") != std::string::npos;
  support_dot_acc_int8_ = ext.find("cl_arm_integer_dot_product_accumulate_int8") != std::string::npos;

  if (device_name_.find("Mali") != std::string::npos) {
    gpu_type_ = MALI;
  } else if (device_name_.find("Adreno") != std::string::npos) {
    gpu_type_ = ADRENO;
  } else if (device_name_.find("Radeon") != std::string::npos || device_name_.find("AMD") != std::string::npos) {
    gpu_type_ = RADEON;
  } else if (device_name_.find("Intel") != std::string::npos) {
    gpu_type_ = INTEL;
  } else {
    gpu_type_ = OTHER;
  }

  std::string version_str;
  devices_[0].getInfo(CL_DEVICE_VERSION, &version_str);
  if (version_str.find("OpenCL 2.") != std::string::npos) {
    cl_version_ = 2.0f;
  } else if (version_str.find("OpenCL 3.") != std::string::npos) {
    cl_version_ = 3.0f;
  }

  default_build_params_ = " -cl-std=CL2.0";
}

OpenCLRuntime::~OpenCLRuntime() = default;

cl::Context& OpenCLRuntime::context() { return *context_; }

cl::CommandQueue& OpenCLRuntime::commandQueue() { return *command_queue_; }

const std::vector<cl::Device>& OpenCLRuntime::getDevices() const { return devices_; }

bool OpenCLRuntime::isSupportedFP16() const { return is_supported_fp16_; }

bool OpenCLRuntime::isSupportedDotInt8() const { return support_dot_int8_; }

bool OpenCLRuntime::isSupportedDotAccInt8() const { return support_dot_acc_int8_; }

uint64_t OpenCLRuntime::deviceGlobalMemeryCacheSize() const { return gpu_global_memery_cache_size_; }

uint32_t OpenCLRuntime::deviceComputeUnits() const { return gpu_compute_units_; }

uint32_t OpenCLRuntime::maxWorkGroupSize() const { return max_work_group_size_; }

GpuType OpenCLRuntime::getGpuType() { return gpu_type_; }

bool OpenCLRuntime::loadProgram(const std::string& programName, cl::Program* program) {
  const std::string& program_path = programName;
  std::ifstream file(program_path);
  if (!file.is_open()) {
    MLLM_ERROR("Failed to open program file: %s\n", program_path.c_str());
    return false;
  }
  std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
  *program = cl::Program(*context_, source);
  return true;
}

bool OpenCLRuntime::buildProgram(const std::string& buildOptionsStr, cl::Program* program) {
  cl_int ret = program->build(devices_, buildOptionsStr.c_str());
  if (ret != CL_SUCCESS) {
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
      std::string build_log;
      program->getBuildInfo(devices_[0], CL_PROGRAM_BUILD_LOG, &build_log);
      MLLM_INFO("Build log: %s\n", build_log.c_str());
    }
    MLLM_ERROR("Build program failed: %d\n", ret);
    return false;
  }
  return true;
}

std::shared_ptr<KernelWrap> OpenCLRuntime::buildKernel(const std::string& programName, const std::string& kernelName,
                                                       const std::set<std::string>& buildOptions) {
  std::string buildOptionsStr;
  for (const auto& option : buildOptions) { buildOptionsStr += " " + option; }
  buildOptionsStr += default_build_params_;

  std::lock_guard<std::mutex> lock(build_program_mutex_);

  auto key = std::make_pair(programName, buildOptionsStr);
  auto& program_with_kernel = build_program_map_[key];

  if (program_with_kernel.program() == nullptr) {
    if (!loadProgram(programName, &program_with_kernel.program)) { return nullptr; }
    if (!buildProgram(buildOptionsStr, &program_with_kernel.program)) { return nullptr; }
  }

  auto& kernel_pool = program_with_kernel.kernels[kernelName];
  if (kernel_pool.recycle_.empty()) {
    auto kernel = std::make_shared<cl::Kernel>(program_with_kernel.program, kernelName.c_str());
    kernel->getWorkGroupInfo(devices_[0], CL_KERNEL_WORK_GROUP_SIZE, &kernel_pool.max_work_group_size_);
    return std::make_shared<KernelWrap>(kernel, &kernel_pool);
  }

  auto kernel = kernel_pool.recycle_.front();
  kernel_pool.recycle_.pop();
  return std::make_shared<KernelWrap>(kernel, &kernel_pool);
}

bool OpenCLRuntime::getDeviceSupportsExtension(const cl::Device& device, const char* extensionName) {
  std::string extensions;
  device.getInfo(CL_DEVICE_EXTENSIONS, &extensions);
  return extensions.find(extensionName) != std::string::npos;
}

}  // namespace mllm::opencl
