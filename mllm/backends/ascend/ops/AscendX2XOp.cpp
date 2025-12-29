// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendX2XOp.hpp"

#include <acl/acl.h>
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

AscendX2XOp::AscendX2XOp(const aops::X2XOpOptions& options) : aops::X2XOp(options) {}

void AscendX2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 1);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  const auto& input = inputs[0];
  auto& output = outputs[0];

  const DeviceTypes input_device = input.device();
  const DeviceTypes output_device = output.device();

  // Case 1: CPU -> Ascend
  if (input_device == kCPU && output_device == kAscend) {
    const size_t data_size = input.bytes();
    const void* src_data = input.ptr<void>();
    void* dst_data = output.ptr<void>();

    // Copy data from CPU to Ascend device
    auto ret = aclrtMemcpy(
        dst_data, data_size,
        src_data, data_size,
        ACL_MEMCPY_HOST_TO_DEVICE);

    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }

    syncGlobalAtbStream();
    return;
  }

  // Case 2: Ascend -> CPU
  if (input_device == kAscend && output_device == kCPU) {
    const size_t data_size = input.bytes();
    const void* src_data = input.ptr<void>();
    void* dst_data = output.ptr<void>();

    // Copy data from Ascend device to CPU
    auto ret = aclrtMemcpy(
        dst_data, data_size,
        src_data, data_size,
        ACL_MEMCPY_DEVICE_TO_HOST);

    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }

    syncGlobalAtbStream();
    return;
  }

  // Case 3: Ascend -> Ascend (same device, just copy pointer or do memcpy)
  if (input_device == kAscend && output_device == kAscend) {
    const size_t data_size = input.bytes();
    const void* src_data = input.ptr<void>();
    void* dst_data = output.ptr<void>();

    if (src_data != dst_data) {
      auto ret = aclrtMemcpy(
          dst_data, data_size,
          src_data, data_size,
          ACL_MEMCPY_DEVICE_TO_DEVICE);

      if (ret != ACL_SUCCESS) {
        MLLM_ACL_CHECK(ret);
      }

      syncGlobalAtbStream();
    }
    return;
  }

  MLLM_ERROR("AscendX2XOp only supports transform between CPU and Ascend devices. "
             "Input device: {}, Output device: {}", 
             static_cast<int>(input_device), static_cast<int>(output_device));
}

}  // namespace mllm::ascend

