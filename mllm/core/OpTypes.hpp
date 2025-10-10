// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

namespace mllm {

enum class OpTypes : int32_t {
  kOpType_Start = 0,

  // Op set V1.
  kFill = 1,
  kAdd = 2,
  kSub = 3,
  kMul = 4,
  kDiv = 5,
  kMatMul = 6,
  kEmbedding = 7,
  kLinear = 8,
  kRoPE = 9,
  kSoftmax = 10,
  kSTFT = 11,
  kISTFT = 12,
  kTranspose = 13,
  kRMSNorm = 14,
  kSiLU = 15,
  kKVCache = 16,
  kCausalMask = 17,
  kCastType = 18,
  kX2X = 19,
  kSplit = 20,
  kView = 21,
  kFlashAttention2 = 22,
  kRepeat = 23,
  kPermute = 24,
  kConv3D = 25,
  kConv2D = 26,
  kConv1D = 27,
  kGELU = 28,
  kLayerNorm = 29,
  kMultimodalRoPE = 30,
  kVisionRoPE = 31,
  kQuickGELU = 32,
  kCopy = 33,
  kClone = 34,
  kNeg = 35,
  kConcat = 36,
  kReLU = 37,
  kReLU2 = 38,
  kReduceMax = 39,
  kReduceMin = 40,
  kReduceSum = 41,
  kContiguous = 42,
  kReshape = 43,
  kSlice = 44,
  kParam = 45,
  kIndex = 46,
  kAbs = 47,
  kLog = 48,
  kTopK = 49,
  kMean = 50,
  kClip = 51,
  kExp = 52,
  kSin = 53,
  kCos = 54,

  // Graph Control Ops
  kGraphBegin = 55,
  kGraphEnd = 56,

  // High-level Op or Fused Op
  kPagedAttn = 57,
  kRadixAttn = 58,
  kScatter2Shards = 59,

  // Dynamic Op Start for user to register there own ops.
  kDynamicOp_Start = 4096,

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
    case OpTypes::kAbs: return "Abs";
    case OpTypes::kLog: return "Log";
    case OpTypes::kMatMul: return "MatMul";
    case OpTypes::kEmbedding: return "Embedding";
    case OpTypes::kLinear: return "Linear";
    case OpTypes::kRoPE: return "RoPE";
    case OpTypes::kSoftmax: return "Softmax";
    case OpTypes::kSTFT: return "STFT";
    case OpTypes::kISTFT: return "ISTFT";
    case OpTypes::kTranspose: return "Transpose";
    case OpTypes::kRMSNorm: return "RMSNorm";
    case OpTypes::kSiLU: return "SiLU";
    case OpTypes::kKVCache: return "KVCache";
    case OpTypes::kCausalMask: return "CausalMask";
    case OpTypes::kCastType: return "CastType";
    case OpTypes::kX2X: return "X2X";
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
    case OpTypes::kReduceMax: return "ReduceMax";
    case OpTypes::kReduceMin: return "ReduceMin";
    case OpTypes::kReduceSum: return "ReduceSum";
    case OpTypes::kContiguous: return "Contiguous";
    case OpTypes::kReshape: return "Reshape";
    case OpTypes::kSlice: return "Slice";
    case OpTypes::kIndex: return "Index";
    case OpTypes::kTopK: return "TopK";
    case OpTypes::kMean: return "Mean";
    case OpTypes::kClip: return "Clip";
    case OpTypes::kExp: return "Exp";
    case OpTypes::kSin: return "Sin";
    case OpTypes::kCos: return "Cos";
    case OpTypes::kParam: return "Param";
    case OpTypes::kGraphBegin: return "GraphBegin";
    case OpTypes::kGraphEnd: return "GraphEnd";
    case OpTypes::kPagedAttn: return "PagedAttn";
    case OpTypes::kScatter2Shards: return "Scatter2Shards";
    case OpTypes::kOpType_End: return "OpType_End";
    default: return "Unknown";
  }
}

struct OpTypesSymbolTableFormatter {
  std::string operator()(const mllm::OpTypes& optype) const { return optype2Str(optype); }
};

}  // namespace mllm
