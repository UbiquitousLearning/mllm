/**
 * @file TensorStorage.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#include "mllm/core/Storage.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

enum TensorMemTypes : int32_t {  // NOLINT
  kTensorMemTypes_Start = 0,

  kNormal,
  kExtraInput,
  kExtraOutput,
  kManual,
  kGlobal,

  kParams_Start,
  kParamsMMAP,
  kParamsNormal,
  kParams_End,

  kReference,

  kQnnAppRead,
  kQnnAppWrite,
  kQnnAppReadWrite,

  kTensorMemTypes_End,
};

class TensorStorage final : public Storage {
 public:
  // TODO   ~TensorStorage() override;

  // TODO   static std::shared_ptr<TensorStorage> create(const std::vector<int32_t>& shape, DataTypes dtype, DeviceTypes
  // device);

  std::string name_;
  // TODO DataTypes dtype_ = kFp32;
  TensorMemTypes mem_type_ = kNormal;
};

}  // namespace mllm
