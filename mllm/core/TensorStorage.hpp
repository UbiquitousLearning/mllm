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

namespace mllm {

enum TensorMemTypes : int32_t {  // NOLINT
  kTensorMemTypes_Start = 0,

  // For MLLM Frame work to use
  kNormal,
  kExtraInput,
  kExtraOutput,
  kManual,
  kGlobal,
  kReference,

  // Framework need to judge if this tensor is mmap from disk.
  kParams_Start,
  kParamsMMAP,
  kParamsNormal,
  kParams_End,

  // For QNN Backend to use.
  kQnnAppRead,
  kQnnAppWrite,
  kQnnAppReadWrite,

  kTensorMemTypes_End,
};

class TensorStorage final : public Storage {
 public:
  ~TensorStorage() override;

  static std::shared_ptr<TensorStorage> create(const std::vector<int32_t>& shape, DataTypes dtype, const DeviceTypes& device);

  std::string name_;
  DataTypes dtype_ = kFloat32;
  TensorMemTypes mem_type_ = kNormal;
};

}  // namespace mllm
