/**
 * @file DeviceTypes.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/core/DeviceTypes.hpp"

namespace mllm {

Device::Device(const std::string& device_binary) {
  auto [type_tt, id_tt] = parse(device_binary);
  type_ = type_tt;
  id_ = id_tt;
}

Device::Device(DeviceTypes device_type, int32_t id) : type_(device_type), id_(id) {}

}  // namespace mllm
