
// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class MultimodalRoPEOpOptionsType : uint8_t {
  kStart = 0,
  kDefault,
  kQwen2VL,
  kEnd,
};

enum class MultimodalRoPEOpOptionsInputType : uint8_t {
  kBHSD = 0,
  kBSHD = 1,
};

struct DefaultMultimodalRoPEOpOptions {};

struct Qwen2VLMultimodalRoPEOpOptions {
  float rope_theta;
  int32_t max_position_embeddings;
  std::vector<int32_t> mrope_section;
};

struct MultimodalRoPEOpOptions : public BaseOpOptions<MultimodalRoPEOpOptions> {
  MultimodalRoPEOpOptionsType type;
  union {
    DefaultMultimodalRoPEOpOptions default_options;
    Qwen2VLMultimodalRoPEOpOptions qwen2vl_options;
  };

  MultimodalRoPEOpOptionsInputType input_type = MultimodalRoPEOpOptionsInputType::kBHSD;

  MultimodalRoPEOpOptions(MultimodalRoPEOpOptionsType t, const Qwen2VLMultimodalRoPEOpOptions& o,
                          MultimodalRoPEOpOptionsInputType i_type)
      : type(t), input_type(i_type) {
    new (&qwen2vl_options) Qwen2VLMultimodalRoPEOpOptions(o);
  }

  explicit MultimodalRoPEOpOptions(MultimodalRoPEOpOptionsType t = MultimodalRoPEOpOptionsType::kDefault) : type(t) {
    if (type == MultimodalRoPEOpOptionsType::kQwen2VL) {
      new (&qwen2vl_options) Qwen2VLMultimodalRoPEOpOptions();
    } else {
      new (&default_options) DefaultMultimodalRoPEOpOptions();
    }
  }

  ~MultimodalRoPEOpOptions() {
    if (type == MultimodalRoPEOpOptionsType::kQwen2VL) { qwen2vl_options.~Qwen2VLMultimodalRoPEOpOptions(); }
  }

  MultimodalRoPEOpOptions(const MultimodalRoPEOpOptions& other) : type(other.type), input_type(other.input_type) {
    // placement new.
    switch (type) {
      case MultimodalRoPEOpOptionsType::kQwen2VL:
        new (&qwen2vl_options) Qwen2VLMultimodalRoPEOpOptions(other.qwen2vl_options);
        break;
      case MultimodalRoPEOpOptionsType::kDefault:
        new (&default_options) DefaultMultimodalRoPEOpOptions(other.default_options);
        break;
      default: break;
    }
  }

  MultimodalRoPEOpOptions& operator=(const MultimodalRoPEOpOptions& other) {
    if (this == &other) { return *this; }
    this->~MultimodalRoPEOpOptions();
    new (this) MultimodalRoPEOpOptions(other);
    return *this;
  }
};

class MultimodalRoPEOp : public BaseOp {
 public:
  explicit MultimodalRoPEOp(const MultimodalRoPEOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const MultimodalRoPEOpOptions& options() const { return options_; }

 protected:
  MultimodalRoPEOpOptions options_;
};
}  // namespace mllm::aops
