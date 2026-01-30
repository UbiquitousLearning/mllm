// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/AscendBackend.hpp"
#include "mllm/backends/ascend/AscendAllocator.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/core/DeviceTypes.hpp"

#include "mllm/backends/ascend/ops/AscendElewiseOps.hpp"
#include "mllm/backends/ascend/ops/AscendX2XOp.hpp"
#include "mllm/backends/ascend/ops/AscendSiLUOp.hpp"
#include "mllm/backends/ascend/ops/AscendLinearOp.hpp"
#include "mllm/backends/ascend/ops/AscendRMSNormOp.hpp"
#include "mllm/backends/ascend/ops/AscendViewOp.hpp"
#include "mllm/backends/ascend/ops/AscendMatMulOp.hpp"
#include "mllm/backends/ascend/ops/AscendSoftmaxOp.hpp"

namespace mllm::ascend {

AscendBackend::AscendBackend() : Backend(kAscend, createAscendAllocator()) {
  regOpFactory<AscendAddOpFactory,AscendSubOpFactory,AscendMulOpFactory,AscendX2XOpFactory,AscendSiLUOpFactory,
              AscendLinearOpFactory,AscendRMSNormOpFactory,AscendViewOpFactory,AscendMatMulOpFactory,AscendSoftmaxOpFactory>();
  auto& devices = AscendDeviceMetaInfo::instance().devices;
  for (const auto& device : devices) {
    const auto bytes_to_mb = [](size_t bytes) { return bytes / (1024.0 * 1024.0); };
    MLLM_INFO("Found Ascend device {} (ID: {}, SOC: {}, Memory: {:.2f} MB free / {:.2f} MB total)", device.name,
              device.id, device.soc_version, bytes_to_mb(device.free_memory), bytes_to_mb(device.total_memory));
  }
}

std::shared_ptr<AscendBackend> createAscendBackend() { return std::make_shared<AscendBackend>(); }

}  // namespace mllm::ascend
