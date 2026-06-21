// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#define MLLM_MODEL_FILE_V2_MAGIC_NUMBER 0x519A
#define MLLM_MODEL_FILE_V2_VERSION 2
#define MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH 512
#define MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH 256
#define MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH 16

namespace mllm {

// File structure:
// ---------------------------------------------
// ModelFileV2Descriptor (532B)
// ---------------------------------------------
// ExtraData
// ---------------------------------------------
// <- params_desc_offset
// ModelFileV2ParamsDescriptor-0 (416B)
// ModelFileV2ParamsDescriptor-1 (416B)
// ModelFileV2ParamsDescriptor-2 (416B)
// ...
// ModelFileV2ParamsDescriptor-(num_params-1) (416B)
// ---------------------------------------------
// Tensor Data-0
// Tensor Data-1
// Tensor Data-2
// ...
// Tensor Data-(num_params-1)
// ---------------------------------------------

struct __attribute__((packed)) ModelFileV2Descriptor {
  int32_t magic_number;                                   // 4B
  int32_t version;                                        // 4B
  char model_name[MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH];  // 512B
  uint32_t num_params;                                    // 4B
  size_t params_desc_offset;                              // 8B
};
static_assert(sizeof(ModelFileV2Descriptor) == 532, "ModelFileV2Descriptor size mismatch");

struct __attribute__((packed)) ModelFileV2ParamsDescriptor {
  uint32_t parameter_id;                                  // 4B
  uint32_t parameter_type;                                // 4B
  size_t parameter_size;                                  // 8B
  size_t parameter_offset;                                // 8B data_ptr = file_begin + parameter_offset
  size_t shape_len;                                       // 8B
  int32_t shape[MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH];  // 128B
  char name[MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH];       // 256B

  [[nodiscard]] std::string_view _param_name_view() const noexcept { return {name, strnlen(name, sizeof(name))}; }
};
static_assert(sizeof(ModelFileV2ParamsDescriptor) == 352, "ModelFileV2ParamsDescriptor size mismatch");

}  // namespace mllm