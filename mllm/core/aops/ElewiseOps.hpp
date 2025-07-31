/**
 * @file ElewiseOps.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 */
#pragma once
#include "mllm/core/BaseOp.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

#define __MLLM_ELEWISE_OP_DEFINE(name)                                                                         \
  class name : public BaseOp {                                                                                 \
   public:                                                                                                     \
    explicit name(const name##Options& options);                                                               \
    void load(const ParameterFile::ptr_t& ploader) override;                                                   \
    void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override; \
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;                    \
    void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;                    \
    void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;                      \
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

__MLLM_ELEWISE_OP_DEFINE(AddOp);
__MLLM_ELEWISE_OP_DEFINE(SubOp);
__MLLM_ELEWISE_OP_DEFINE(MulOp);
__MLLM_ELEWISE_OP_DEFINE(DivOp);
__MLLM_ELEWISE_OP_DEFINE(NegOp);

}  // namespace mllm::aops

#undef __MLLM_ELEWISE_OP_DEFINE
