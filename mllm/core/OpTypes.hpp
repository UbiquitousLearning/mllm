/**
 * @file OpTypes.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 */
#pragma once

#include <cstdint>

namespace mllm {

enum class OpTypes : int32_t {
  kOpType_Start = 0,

  // Op set V1.
  kFill = 1,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMatMul,
  kEmbedding,
  kLinear,
  kRoPE,
  kSoftmax,
  kTranspose,
  kRMSNorm,
  kSiLU,
  kKVCache,
  kCausalMask,
  kCastType,
  kD2H,
  kH2D,
  kSplit,
  kView,
  kFlashAttention2,
  kRepeat,
  kPermute,
  kConv3D,
  kConv2D,
  kConv1D,
  kGELU,
  kLayerNorm,
  kMultimodalRoPE,
  kVisionRoPE,
  kQuickGELU,
  kCopy,
  kClone,
  kNeg,
  kConcat,
  kReLU,
  kReLU2,

  kOpType_End,
};

}  // namespace mllm
