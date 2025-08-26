// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/BaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::ir::linalg {

LinalgIROp::LinalgIROp() : Op(RK_Op_LinalgIROp) {}

LinalgIROp::LinalgIROp(const NodeKind& kind) : Op(kind) {}

LINALG_AOPS_DECL(OpTypes::kFill, FillOp)
LINALG_AOPS_DECL(OpTypes::kAdd, AddOp);
LINALG_AOPS_DECL(OpTypes::kSub, SubOp);
LINALG_AOPS_DECL(OpTypes::kMul, MulOp);
LINALG_AOPS_DECL(OpTypes::kDiv, DivOp);
LINALG_AOPS_DECL(OpTypes::kAbs, AbsOp);
LINALG_AOPS_DECL(OpTypes::kLog, LogOp);

LINALG_AOPS_DECL(OpTypes::kMatMul, MatMulOp);

LINALG_AOPS_DECL(OpTypes::kEmbedding, EmbeddingOp);
LINALG_AOPS_DECL(OpTypes::kLinear, LinearOp);
LINALG_AOPS_DECL(OpTypes::kRoPE, RoPEOp);
LINALG_AOPS_DECL(OpTypes::kKVCache, KVCacheOp);
LINALG_AOPS_DECL(OpTypes::kCausalMask, CausalMaskOp);

LINALG_AOPS_DECL(OpTypes::kSoftmax, SoftmaxOp);
LINALG_AOPS_DECL(OpTypes::kTranspose, TransposeOp);
LINALG_AOPS_DECL(OpTypes::kRMSNorm, RMSNormOp);
LINALG_AOPS_DECL(OpTypes::kSiLU, SiLUOp);

LINALG_AOPS_DECL(OpTypes::kCastType, CastTypeOp);

LINALG_AOPS_DECL(OpTypes::kX2X, X2XOp);

LINALG_AOPS_DECL(OpTypes::kView, ViewOp);
LINALG_AOPS_DECL(OpTypes::kSplit, SplitOp);
LINALG_AOPS_DECL(OpTypes::kSTFT, STFTOp);

LINALG_AOPS_DECL(OpTypes::kFlashAttention2, FlashAttention2Op);
LINALG_AOPS_DECL(OpTypes::kRepeat, RepeatOp);
LINALG_AOPS_DECL(OpTypes::kPermute, PermuteOp);

LINALG_AOPS_DECL(OpTypes::kConv1D, Conv1DOp);
LINALG_AOPS_DECL(OpTypes::kConv2D, Conv2DOp);
LINALG_AOPS_DECL(OpTypes::kConv3D, Conv3DOp);

LINALG_AOPS_DECL(OpTypes::kGELU, GELUOp);
LINALG_AOPS_DECL(OpTypes::kLayerNorm, LayerNormOp);

LINALG_AOPS_DECL(OpTypes::kMultimodalRoPE, MultimodalRoPEOp);
LINALG_AOPS_DECL(OpTypes::kVisionRoPE, VisionRoPEOp);

LINALG_AOPS_DECL(OpTypes::kQuickGELU, QuickGELUOp);

LINALG_AOPS_DECL(OpTypes::kCopy, CopyOp);
LINALG_AOPS_DECL(OpTypes::kClone, CloneOp);

LINALG_AOPS_DECL(OpTypes::kNeg, NegOp);
LINALG_AOPS_DECL(OpTypes::kConcat, ConcatOp);

LINALG_AOPS_DECL(OpTypes::kReduceMax, ReduceMaxOp);
LINALG_AOPS_DECL(OpTypes::kReduceMin, ReduceMinOp);
LINALG_AOPS_DECL(OpTypes::kReduceSum, ReduceSumOp);

LINALG_AOPS_DECL(OpTypes::kContiguous, ContiguousOp);
LINALG_AOPS_DECL(OpTypes::kReLU, ReLUOp);
LINALG_AOPS_DECL(OpTypes::kReshape, ReshapeOp);

LINALG_AOPS_DECL(OpTypes::kSlice, SliceOp);
LINALG_AOPS_DECL(OpTypes::kParam, ParamOp);

LINALG_AOPS_DECL(OpTypes::kIndex, IndexOp);

}  // namespace mllm::ir::linalg
