// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once
#include "mllm/core/BaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

#define __MLLM_ELEWISE_OP_DEFINE(name, option_name)                                                            \
  class name : public BaseOp {                                                                                 \
   public:                                                                                                     \
    explicit name(const name##Options& options);                                                               \
    void load(const ParameterFile::ptr_t& ploader) override;                                                   \
    void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override; \
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;                    \
    void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;                    \
    void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;                      \
    inline const option_name& options() const { return options_; }                                             \
                                                                                                               \
   protected:                                                                                                  \
    name##Options options_;                                                                                    \
  };

namespace mllm::aops {

struct AddOpOptions : public BaseOpOptions<AddOpOptions> {};

struct SubOpOptions : public BaseOpOptions<SubOpOptions> {};

struct MulOpOptions : public BaseOpOptions<MulOpOptions> {};

struct DivOpOptions : public BaseOpOptions<DivOpOptions> {};

struct NegOpOptions : public BaseOpOptions<NegOpOptions> {};

struct AbsOpOptions : public BaseOpOptions<AbsOpOptions> {};

struct LogOpOptions : public BaseOpOptions<LogOpOptions> {};

struct ExpOpOptions : public BaseOpOptions<ExpOpOptions> {};

struct SinOpOptions : public BaseOpOptions<SinOpOptions> {};

struct CosOpOptions : public BaseOpOptions<CosOpOptions> {};

struct ClipOpOptions : public BaseOpOptions<ClipOpOptions> {
  float min_val = -1.0f;
  float max_val = 1.0f;

  ClipOpOptions& min(float min) {
    min_val = min;
    return *this;
  }

  ClipOpOptions& max(float max) {
    max_val = max;
    return *this;
  }
};

__MLLM_ELEWISE_OP_DEFINE(AddOp, AddOpOptions);
__MLLM_ELEWISE_OP_DEFINE(SubOp, SubOpOptions);
__MLLM_ELEWISE_OP_DEFINE(MulOp, MulOpOptions);
__MLLM_ELEWISE_OP_DEFINE(DivOp, DivOpOptions);
__MLLM_ELEWISE_OP_DEFINE(NegOp, NegOpOptions);

// Unary Ops
__MLLM_ELEWISE_OP_DEFINE(AbsOp, AbsOpOptions);
__MLLM_ELEWISE_OP_DEFINE(LogOp, LogOpOptions);
__MLLM_ELEWISE_OP_DEFINE(ClipOp, ClipOpOptions);
__MLLM_ELEWISE_OP_DEFINE(ExpOp, ExpOpOptions);
__MLLM_ELEWISE_OP_DEFINE(SinOp, SinOpOptions);
__MLLM_ELEWISE_OP_DEFINE(CosOp, CosOpOptions);

}  // namespace mllm::aops

#undef __MLLM_ELEWISE_OP_DEFINE
