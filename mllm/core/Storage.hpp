/**
 * @file Storage.hpp
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
#include <cstddef>
#include <memory>

#include "mllm/core/DeviceTypes.hpp"

namespace mllm {

class Storage : public std::enable_shared_from_this<Storage> {
 public:
  virtual ~Storage() = default;

  enum StorageTypes : uint8_t {
    kStart = 0,
    kGeneral,
    kTensor,
    kEnd,
  };

  void* ptr_ = nullptr;
  size_t size_ = 0;
  Device device_ = Device::CPU;
  StorageTypes type_ = kGeneral;
  uint32_t custom_32bit_uuid_ = -1;
};

}  // namespace mllm
