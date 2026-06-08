// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include <atb/infer_op_params.h>

#include "mllm/mllm.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::models::qwen_ascend {

inline atb::Operation* createLinearGraphOp(bool has_bias) {
  atb::infer::LinearParam param;
  param.transposeA = false;
  param.transposeB = true;
  param.hasBias = has_bias;
  param.outDataType = ACL_DT_UNDEFINED;
  param.enAccum = false;
  param.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  param.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Linear) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createRmsNormGraphOp(float epsilon) {
  atb::infer::RmsNormParam param;
  param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  param.normParam.quantType = atb::infer::QuantType::QUANT_UNQUANT;
  param.normParam.epsilon = epsilon;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(RMS_NORM) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createRoPEGraphOp() {
  atb::infer::RopeParam param;
  param.rotaryCoeff = 2;
  param.cosFormat = 0;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(RoPE) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createTransposeGraphOp(int rank, int dim0, int dim1) {
  atb::infer::TransposeParam param;
  for (int i = 0; i < rank; ++i) {
    param.perm.push_back(i);
  }
  std::swap(param.perm[dim0], param.perm[dim1]);

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Transpose) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createSoftmaxGraphOp(int axis) {
  atb::infer::SoftmaxParam param;
  param.axes.push_back(axis);

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(Softmax) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createAddGraphOp() {
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(ELEWISE_ADD) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createMulGraphOp() {
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ELEWISE_MUL;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError, "ATB CreateOperation(ELEWISE_MUL) failed, status={}", static_cast<int>(st));
  }
  return op;
}

inline atb::Operation* createSiLUGraphOp() {
  atb::infer::ActivationParam param;
  param.activationType = atb::infer::ACTIVATION_SWISH;

  atb::Operation* op = nullptr;
  auto st = atb::CreateOperation(param, &op);
  if (st != atb::NO_ERROR || op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "ATB CreateOperation(ACTIVATION_SWISH) failed, status={}",
                    static_cast<int>(st));
  }
  return op;
}

inline bool isQwenAscendDecoderGraphEnabled() {
  const char* env = std::getenv("MLLM_ASCEND_QWEN_DECODER_GRAPH");
  return env == nullptr || env[0] != '0';
}

inline int32_t getQwenAscendDecoderGraphSetupBucketSize() {
  const char* env = std::getenv("MLLM_ASCEND_QWEN_DECODER_GRAPH_SETUP_BUCKET");
  if (env == nullptr || env[0] == '\0') return 0;
  char* end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed <= 0) return 0;
  return static_cast<int32_t>(parsed);
}

class QwenAscendDecoderGraphRunner final {
 public:
  QwenAscendDecoderGraphRunner() = default;

  void configure(int32_t max_cache_length) {
    max_cache_length_ = max_cache_length;
    setup_bucket_size_ = getQwenAscendDecoderGraphSetupBucketSize();
    if (setup_bucket_size_ > max_cache_length_) {
      setup_bucket_size_ = max_cache_length_;
    }
  }

  [[nodiscard]] bool hasExecutor() const { return executor_ != nullptr; }
  [[nodiscard]] int32_t setupBucketSize() const { return setup_bucket_size_; }

  void setExecutor(std::unique_ptr<mllm::ascend::AscendGraphExecutor> executor) {
    executor_ = std::move(executor);
  }

  Tensor attentionScaleTensor(int32_t head_dim, DataTypes dtype, DeviceTypes device) {
    if (attention_scale_tensor_.isNil()
        || attention_scale_tensor_.dtype() != dtype
        || attention_scale_tensor_.device() != device) {
      attention_scale_tensor_ =
          (Tensor::ones({1, 1, 1, 1}, dtype, kCPU) * (1.f / sqrtf(head_dim))).to(device);
    }
    return attention_scale_tensor_;
  }

  Tensor currentSeqLenTensor(int32_t seq_len) {
    if (current_seq_len_tensor_.isNil()) {
      current_seq_len_tensor_ = Tensor::empty({1}, kInt32, kAscend).alloc();
    }
    int32_t host_seq_len = seq_len;
    auto ret = aclrtMemcpy(current_seq_len_tensor_.ptr<void>(),
                           sizeof(int32_t),
                           &host_seq_len,
                           sizeof(int32_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MLLM_ACL_CHECK(ret);
    }
    return current_seq_len_tensor_;
  }

  void execute(const std::vector<Tensor>& inputs,
               std::vector<Tensor>& outputs) {
    MLLM_RT_ASSERT(executor_ != nullptr);
    executor_->execute(inputs, outputs);
  }

 private:
  std::unique_ptr<mllm::ascend::AscendGraphExecutor> executor_;
  Tensor attention_scale_tensor_;
  Tensor current_seq_len_tensor_;
  int32_t max_cache_length_{0};
  int32_t setup_bucket_size_{0};
};

}  // namespace mllm::models::qwen_ascend
