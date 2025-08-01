// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// After 2025-07-31:
// Developer should not use ModelFileV1 Anymore.

// clang-format off
/*
 * ┌───────┬──────┬───────┬────────┬───────────┬─────────┬─────────┬──────┬──────────────────────┬─────────────────────────┐
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │Index │       │        │           │         │         │      │                      │                         │
 * │       │ Len  │       │        │           │         │         │      │                      │                         │
 * │ Magic │ INT  │ Name  │Name    │ Weights   │ Offset  │ DataType│....  │   Weights Contents   │   Weights Contents      │
 * │       │      │ Length│String  │ Length    │  INT    │  INT    │      │                      │                         │
 * │       │      │ INT   │        │  INT      │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * │       │      │       │        │           │         │         │      │                      │                         │
 * └───────┴──────┴───────┴────────┴───────────┴─────────┴─────────┴──────┴──────────────────────┴─────────────────────────┘
 * Weights File Structure
 */
// clang-format on

#include <string>
#include <cstdint>

#define MLLM_MODEL_FILE_V1_MAGIC_NUMBER 20012

namespace mllm {

// Pack to 1B if possible
struct __attribute__((packed)) ModelFileV1Descriptor {
  int32_t magic_number;            // 4B
  uint64_t parameter_desc_offset;  // 8B
};
static_assert(sizeof(ModelFileV1Descriptor) == 12, "ModelFileV1Descriptor size must be 12 bytes");

// Pack to 1B if possible
struct __attribute__((packed)) ModelFileV1ParamsDescriptor {
  int32_t name_len;   // 4B
  uint64_t data_len;  // 8B
  uint64_t offset;    // 8B
  int32_t dtype;      // 4B
};
static_assert(sizeof(ModelFileV1ParamsDescriptor) == 24, "ModelFileV1ParamsDescriptor size must be 20 bytes");

struct ModelFileV1ParamsDescriptorHelper {
  void* ptr_ = nullptr;
  std::string name_;
  ModelFileV1ParamsDescriptor descriptor_;
};

}  // namespace mllm
