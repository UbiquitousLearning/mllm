// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendCopyOp.hpp"

#include <acl/acl.h>
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

AscendCopyOp::AscendCopyOp(const aops::CopyOpOptions& options) : aops::CopyOp(options) {}

void AscendCopyOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // std::memcpy(i1.ptr<char>(), i0.ptr<char>(), i0.bytes());
  const auto& src = inputs[0];
  auto& dst = inputs[1];

  MLLM_RT_ASSERT(src.device() == kAscend && dst.device() == kAscend);

  const size_t data_size = src.bytes();
  const void* src_data = src.ptr<void>();
  void* dst_data = dst.ptr<void>();

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
}

}  // namespace mllm::ascend
