// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/base/PluginInterface.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::ir::linalg {

LinalgIROp::LinalgIROp() : Op(RK_Op_LinalgIROp) {}

LinalgIROp::LinalgIROp(const NodeKind& kind) : Op(kind) {}

RegisterOp::RegisterOp() : LinalgIROp(RK_Op_LinalgIROp_RegisterOp) { MLLM_EMPTY_SCOPE; }

RegisterOp::ptr_t RegisterOp::build(IRContext* ctx, BaseOp* aop, const std::string& symbol_name) {
  auto ret = std::make_shared<RegisterOp>();
  auto symbol_attr = ctx->create<SymbolAttr>(symbol_name);
  ret->setSymbolAttr(symbol_attr);
  ret->bare_op_ptr_ = aop;
  ret->op_type_ = aop->getOpType();
  ctx->addToSymbolTable(ret, symbol_attr->str());
  return ret;
}

void RegisterOp::dump(IRPrinter& p) {
  p.print("linalg.{}.register<{}>", deviceTypes2Str(getDevice()), optype2Str(op_type_));
  Op::dump(p);
  dumpAttributes(p);
}

LINALG_AOPS_DECL(OpTypes::kFill, FillOp)
LINALG_AOPS_DECL(OpTypes::kAdd, AddOp);
LINALG_AOPS_DECL(OpTypes::kSub, SubOp);
LINALG_AOPS_DECL(OpTypes::kMul, MulOp);
LINALG_AOPS_DECL(OpTypes::kDiv, DivOp);
LINALG_AOPS_DECL(OpTypes::kAbs, AbsOp);
LINALG_AOPS_DECL(OpTypes::kLog, LogOp);
LINALG_AOPS_DECL(OpTypes::kExp, ExpOp)
LINALG_AOPS_DECL(OpTypes::kSin, SinOp)
LINALG_AOPS_DECL(OpTypes::kCos, CosOp)

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
LINALG_AOPS_DECL(OpTypes::kISTFT, ISTFTOp);

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
LINALG_AOPS_DECL(OpTypes::kTopK, TopKOp);
LINALG_AOPS_DECL(OpTypes::kMean, MeanOp);
LINALG_AOPS_DECL(OpTypes::kClip, ClipOp);
LINALG_AOPS_DECL(OpTypes::kPagedAttn, PagedAttnOp);


LINALG_AOPS_DECL(OpTypes::kLayerNorm2D, LayerNorm2DOp);
LINALG_AOPS_DECL(OpTypes::kPad, PadOp);
LINALG_AOPS_DECL(OpTypes::kInterpolate, InterpolateOp);
LINALG_AOPS_DECL(OpTypes::kEinsum, EinsumOp);
LINALG_AOPS_DECL(OpTypes::kStack, StackOp);
LINALG_AOPS_DECL(OpTypes::kMaskedScatter, MaskedScatterOp);

LINALG_AOPS_DECL(OpTypes::kScatter, ScatterOp);
LINALG_AOPS_DECL(OpTypes::kGather, GatherOp);
LINALG_AOPS_DECL(OpTypes::kArgsort, ArgsortOp);

// special implementation for CustomizedOp
CustomizedOp ::~CustomizedOp() = default;
CustomizedOp ::CustomizedOp(const BaseOp ::ptr_t& aop) : LinalgIROp(RK_Op_LinalgIROp_CustomizedOp) {
  setAOp(aop->getOpType(), aop);
}
::mllm ::ir ::linalg ::CustomizedOp ::ptr_t CustomizedOp ::build(
    IRContext* ctx, const BaseOp ::ptr_t& aop, const std ::vector<::mllm ::ir ::tensor ::TensorValue ::ptr_t>& ins,
    const std ::vector<::mllm ::ir ::tensor ::TensorValue ::ptr_t>& ous) {
  auto op = std ::make_shared<::mllm ::ir ::linalg ::CustomizedOp>(aop);
  for (auto& i : ins) { (*i)-- > op; }
  for (auto& o : ous) { (*op)-- > o; }
  op->setDevice(aop->getDevice());
  return op;
}
void CustomizedOp ::dump(IRPrinter& p) {
  p.print("linalg.{}.{}", deviceTypes2Str(getDevice()),
          std::static_pointer_cast<mllm::plugin::interface::CustomizedOp>(op_)->getCustomOpTypeName());
  if (!getAOp()->getName().empty()) { p.print(" [name=\"{}\"]", getAOp()->getName()); }
  Op ::dump(p);
}

}  // namespace mllm::ir::linalg
