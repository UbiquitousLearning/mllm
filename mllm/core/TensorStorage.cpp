// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm {

TensorStorage::~TensorStorage() {
  switch (mem_type_) {
    case kNormal:
    case kGlobal:
    case kParamsNormal: {
      Context::instance().memoryManager()->free(this);
      break;
    }
    case kExtraInput:
    case kExtraOutput:
    case kParamsMMAP:
    case kQnnAppRead:
    case kQnnAppWrite:
    case kQnnAppReadWrite:
    case kManual: {
      MLLM_EMPTY_SCOPE
      break;
    }
    default:
      MLLM_WARN("When trying to free TensorStorage, found invalid mem_type_. Mllm will still trying "
                "to free this TensorStorage, but may lead to memory error.");
      Context::instance().memoryManager()->free(this);
      break;
  };
}

std::shared_ptr<TensorStorage> TensorStorage::create(const std::vector<int32_t>& shape, DataTypes dtype,
                                                     const DeviceTypes& device) {
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

  // Give it a unique uuid
  ret->custom_32bit_uuid_ = Context::instance().getUUID();

  return ret;
}

}  // namespace mllm