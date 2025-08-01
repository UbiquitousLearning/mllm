// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>

#include "mllm/core/DeviceTypes.hpp"

namespace mllm {

class Storage : public std::enable_shared_from_this<Storage> {
 public:
  using ptr_t = std::shared_ptr<Storage>;

  virtual ~Storage() = default;

  enum StorageTypes : uint8_t {
    kStart = 0,
    kGeneral,
    kTensor,
    kEnd,
  };

  void* ptr_ = nullptr;
  size_t size_ = 0;
  DeviceTypes device_ = kCPU;
  StorageTypes type_ = kGeneral;
  uint32_t custom_32bit_uuid_ = -1;
};

}  // namespace mllm
