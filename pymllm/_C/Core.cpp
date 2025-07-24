/**
 * @file Core.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "pymllm/_C/Core.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"

void registerCoreBinding(py::module_& m) {
  py::enum_<mllm::DeviceTypes>(m, "DeviceTypes")
      .value("CPU", mllm::DeviceTypes::kCPU)
      .value("CUDA", mllm::DeviceTypes::kCUDA)
      .value("OpenCL", mllm::DeviceTypes::kOpenCL);
}
