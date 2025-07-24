/**
 * @file ModelFileV2.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-24
 *
 */
#pragma once

#include <cstdint>

#define MLLM_MODEL_FILE_V2_MAGIC_NUMBER 0x519A

namespace mllm {

struct __attribute__((packed)) ModelFileV2Descriptor {
  int32_t magic_number;  // 4B
};

}  // namespace mllm