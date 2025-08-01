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
#include <string>

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
  kReduceMax,
  kReduceMin,
  kReduceSum,

  // Graph Control Ops
  kGraphBegin,
  kGraphEnd,

  kOpType_End,
};

inline std::string optype2Str(OpTypes type) {
  switch (type) {
    case OpTypes::kOpType_Start: return "OpType_Start";
    case OpTypes::kFill: return "Fill";
    case OpTypes::kAdd: return "Add";
    case OpTypes::kSub: return "Sub";
    case OpTypes::kMul: return "Mul";
    case OpTypes::kDiv: return "Div";
    case OpTypes::kMatMul: return "MatMul";
    case OpTypes::kEmbedding: return "Embedding";
    case OpTypes::kLinear: return "Linear";
    case OpTypes::kRoPE: return "RoPE";
    case OpTypes::kSoftmax: return "Softmax";
    case OpTypes::kTranspose: return "Transpose";
    case OpTypes::kRMSNorm: return "RMSNorm";
    case OpTypes::kSiLU: return "SiLU";
    case OpTypes::kKVCache: return "KVCache";
    case OpTypes::kCausalMask: return "CausalMask";
    case OpTypes::kCastType: return "CastType";
    case OpTypes::kD2H: return "D2H";
    case OpTypes::kH2D: return "H2D";
    case OpTypes::kSplit: return "Split";
    case OpTypes::kView: return "View";
    case OpTypes::kFlashAttention2: return "FlashAttention2";
    case OpTypes::kRepeat: return "Repeat";
    case OpTypes::kPermute: return "Permute";
    case OpTypes::kConv3D: return "Conv3D";
    case OpTypes::kConv2D: return "Conv2D";
    case OpTypes::kConv1D: return "Conv1D";
    case OpTypes::kGELU: return "GELU";
    case OpTypes::kLayerNorm: return "LayerNorm";
    case OpTypes::kMultimodalRoPE: return "MultimodalRoPE";
    case OpTypes::kVisionRoPE: return "VisionRoPE";
    case OpTypes::kQuickGELU: return "QuickGELU";
    case OpTypes::kCopy: return "Copy";
    case OpTypes::kClone: return "Clone";
    case OpTypes::kNeg: return "Neg";
    case OpTypes::kConcat: return "Concat";
    case OpTypes::kReLU: return "ReLU";
    case OpTypes::kReLU2: return "ReLU2";
    case OpTypes::kGraphBegin: return "GraphBegin";
    case OpTypes::kGraphEnd: return "GraphEnd";
    case OpTypes::kOpType_End: return "OpType_End";
    default: return "Unknown";
  }
}

}  // namespace mllm
