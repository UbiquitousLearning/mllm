// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/builtin/Interface.hpp"

namespace mllm {
class BaseOp;
class AddOp;
class SubOp;
class MulOp;
class DivOp;
class AbsOp;
class LogOp;
class FillOp;
class MatMulOp;
class EmbeddingOp;
class LinearOp;
class RoPEOp;
class KVCacheOp;
class SoftmaxOp;
class STFTOp;
class ISTFTOp;
class TransposeOp;
class RMSNormOp;
class SiLUOp;
class CausalMaskOp;
class CastTypeOp;
class X2XOp;
class ViewOp;
class SplitOp;
class FlashAttention2Op;
class RepeatOp;
class PermuteOp;
class Conv1DOp;
class Conv2DOp;
class Conv3DOp;
class GELUOp;
class LayerNormOp;
class MultimodalRoPEOp;
class VisionRoPEOp;
class QuickGELUOp;
class CopyOp;
class CloneOp;
class NegOp;
class ConcatOp;
class ReduceMaxOp;
class ReduceMinOp;
class ReduceSumOp;
class ReLUOp;
class ContiguousOp;
class ReshapeOp;
class SliceOp;
class ParamOp;
class IndexOp;
class TopKOp;
class MeanOp;
class ClipOp;
class ExpOp;
class SinOp;
class CosOp;
class PagedAttnOp;
}  // namespace mllm

#define LINALG_AOPS_DEFINE(class_name, rtti_name)                                                                       \
  class class_name final : public LinalgIROp {                                                                          \
   public:                                                                                                              \
    DEFINE_SPECIFIC_IR_CLASS(class_name);                                                                               \
    ~class_name() override;                                                                                             \
    class_name();                                                                                                       \
    explicit class_name(const BaseOp::ptr_t& aop);                                                                      \
    ::mllm::class_name* getOp() { return (::mllm::class_name*)(op_.get()); }                                            \
    static inline bool classof(const Node* node) { RTTI_RK_OP_LINALGIROP_##rtti_name##_IMPL(node); }                    \
    static ::mllm::ir::linalg::class_name::ptr_t build(IRContext* ctx, const BaseOp::ptr_t& aop,                        \
                                                       const std::vector<::mllm::ir::tensor::TensorValue::ptr_t>& ins,  \
                                                       const std::vector<::mllm::ir::tensor::TensorValue::ptr_t>& ous); \
    void dump(IRPrinter& p) override;                                                                                   \
  }

#define LINALG_AOPS_DECL(op_type, class_name)                                                                               \
  class_name::~class_name() = default;                                                                                      \
  class_name::class_name(const BaseOp::ptr_t& aop) : LinalgIROp(RK_Op_LinalgIROp_##class_name) { setAOp(op_type, aop); }    \
  ::mllm::ir::linalg::class_name::ptr_t class_name::build(IRContext* ctx, const BaseOp::ptr_t& aop,                         \
                                                          const std::vector<::mllm::ir::tensor::TensorValue::ptr_t>& ins,   \
                                                          const std::vector<::mllm::ir::tensor::TensorValue::ptr_t>& ous) { \
    auto op = std::make_shared<::mllm::ir::linalg::class_name>(aop);                                                        \
    for (auto& i : ins) { (*i)-- > op; }                                                                                    \
    for (auto& o : ous) { (*op)-- > o; }                                                                                    \
    op->setDevice(aop->getDevice());                                                                                        \
    return op;                                                                                                              \
  }                                                                                                                         \
  void class_name::dump(IRPrinter& p) {                                                                                     \
    p.print("linalg.{}.{}", deviceTypes2Str(getDevice()), #class_name);                                                     \
    Op::dump(p);                                                                                                            \
  }

namespace mllm::ir::linalg {
class LinalgIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(LinalgIROp);

  ~LinalgIROp() override = default;
  LinalgIROp();
  explicit LinalgIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_LINALGIROP_IMPL(node); }

  inline void setAOp(OpTypes op_type, const BaseOp::ptr_t& op) {
    op_type_ = op_type;
    op_ = op;
  }

  inline OpTypes getAOpTypes() const { return op_type_; }

  inline BaseOp* getAOp() const { return op_.get(); }

 protected:
  OpTypes op_type_;
  BaseOp::ptr_t op_;
};

class RegisterOp : public LinalgIROp, public SymbolInterface<RegisterOp> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(RegisterOp);

  RegisterOp();

  static inline bool classof(const Node* node) { RTTI_RK_OP_LINALGIROP_REGISTEROP_IMPL(node); }

  static ptr_t build(IRContext* ctx, BaseOp* aop, const std::string& symbol_name);

  void dump(IRPrinter& p) override;

  inline BaseOp* getOp() const { return bare_op_ptr_; }

 private:
  BaseOp* bare_op_ptr_ = nullptr;
};

LINALG_AOPS_DEFINE(FillOp, FILLOP);
LINALG_AOPS_DEFINE(AddOp, ADDOP);
LINALG_AOPS_DEFINE(SubOp, SUBOP);
LINALG_AOPS_DEFINE(MulOp, MULOP);
LINALG_AOPS_DEFINE(DivOp, DIVOP);
LINALG_AOPS_DEFINE(AbsOp, ABSOP);
LINALG_AOPS_DEFINE(LogOp, LOGOP);
LINALG_AOPS_DEFINE(ExpOp, EXPOP);
LINALG_AOPS_DEFINE(SinOp, SINOP);
LINALG_AOPS_DEFINE(CosOp, COSOP);

LINALG_AOPS_DEFINE(MatMulOp, MATMULOP);

LINALG_AOPS_DEFINE(EmbeddingOp, EMBEDDINGOP);
LINALG_AOPS_DEFINE(LinearOp, LINEAROP);
LINALG_AOPS_DEFINE(RoPEOp, ROPEOP);
LINALG_AOPS_DEFINE(KVCacheOp, KVCACHEOP);
LINALG_AOPS_DEFINE(CausalMaskOp, CAUSALMASKOP);

LINALG_AOPS_DEFINE(SoftmaxOp, SOFTMAXOP);
LINALG_AOPS_DEFINE(TransposeOp, TRANSPOSEOP);
LINALG_AOPS_DEFINE(RMSNormOp, RMSNORMOP);
LINALG_AOPS_DEFINE(SiLUOp, SILUOP);

LINALG_AOPS_DEFINE(CastTypeOp, CASTTYPEOP);

LINALG_AOPS_DEFINE(X2XOp, X2XOP);

LINALG_AOPS_DEFINE(ViewOp, VIEWOP);
LINALG_AOPS_DEFINE(SplitOp, SPLITOP);

LINALG_AOPS_DEFINE(FlashAttention2Op, FLASHATTENTION2OP);
LINALG_AOPS_DEFINE(RepeatOp, REPEATOP);
LINALG_AOPS_DEFINE(PermuteOp, PERMUTEOP);

LINALG_AOPS_DEFINE(Conv1DOp, CONV1DOP);
LINALG_AOPS_DEFINE(Conv2DOp, CONV2DOP);
LINALG_AOPS_DEFINE(Conv3DOp, CONV3DOP);

LINALG_AOPS_DEFINE(GELUOp, GELUOP);
LINALG_AOPS_DEFINE(LayerNormOp, LAYERNORMOP);

LINALG_AOPS_DEFINE(MultimodalRoPEOp, MULTIMODALROPEOP);
LINALG_AOPS_DEFINE(VisionRoPEOp, VISIONROPEOP);

LINALG_AOPS_DEFINE(QuickGELUOp, QUICKGELUOP);

LINALG_AOPS_DEFINE(CopyOp, COPYOP);
LINALG_AOPS_DEFINE(CloneOp, CLONEOP);

LINALG_AOPS_DEFINE(NegOp, NEGOP);
LINALG_AOPS_DEFINE(ConcatOp, CONCATOP);

LINALG_AOPS_DEFINE(ReduceMaxOp, REDUCEMAXOP);
LINALG_AOPS_DEFINE(ReduceMinOp, REDUCEMINOP);
LINALG_AOPS_DEFINE(ReduceSumOp, REDUCESUMOP);

LINALG_AOPS_DEFINE(ReLUOp, RELUOP);
LINALG_AOPS_DEFINE(ContiguousOp, CONTIGUOUSOP);
LINALG_AOPS_DEFINE(ReshapeOp, RESHAPEOP);

LINALG_AOPS_DEFINE(SliceOp, SLICEOP);
LINALG_AOPS_DEFINE(STFTOp, STFTOP);
LINALG_AOPS_DEFINE(ISTFTOp, ISTFTOP);
LINALG_AOPS_DEFINE(ParamOp, PARAMOP);

LINALG_AOPS_DEFINE(IndexOp, INDEXOP);
LINALG_AOPS_DEFINE(TopKOp, TOPKOP);
LINALG_AOPS_DEFINE(MeanOp, MEANOP);
LINALG_AOPS_DEFINE(ClipOp, CLIPOP);
LINALG_AOPS_DEFINE(PagedAttnOp, PAGEDATTNOP);

}  // namespace mllm::ir::linalg
