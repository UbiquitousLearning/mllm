/**
 * @file TensorStorage.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/TensorStorage.hpp"

namespace mllm {

TensorStorage::~TensorStorage() {
  switch (mem_type_) {
    case kNormal:
    case kGlobal:
    case kParamsNormal:
      // TODO MllmEngineCtx::instance().mem()->free(this); break;
    case kExtraInput:
    case kExtraOutput:
    case kParamsMMAP:
    case kQnnAppRead:
    case kQnnAppWrite:
    case kQnnAppReadWrite:
    case kManual: break;
    case kReference: MLLM_WARN("mem_type_ kReference is not used anymore."); break;
    default:
      MLLM_WARN("When trying to free TensorStorage, found invalid mem_type_. Mllm will still trying "
                "to free this TensorStorage, but may lead to memory error.");
      // TODO MllmEngineCtx::instance().mem()->free(this);
      break;
  };
}

std::shared_ptr<TensorStorage> TensorStorage::create(const std::vector<int32_t>& shape, DataTypes dtype, const Device& device) {
  auto ret = std::make_shared<TensorStorage>();

  ret->dtype_ = dtype;
  ret->device_ = device;

  size_t cnt = 1;
  for (auto i : shape) { cnt *= i; }

  // Get total bytes
  MLLM_RT_ASSERT((cnt % lanesOfType(dtype)) == 0);
  ret->size_ = (cnt / lanesOfType(dtype)) * bytesOfType(dtype);

  // Set storage type
  ret->type_ = Storage::kTensor;

  // TODO ret->custom_32bit_uuid_ = MllmEngineCtx::instance().getUUID();

  return ret;
}

}  // namespace mllm