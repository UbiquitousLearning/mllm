// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/CmpOp.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace MLLM_ANONYMOUS_NAMESPACE {
static std::vector<int> broadcastShapes(const std::vector<std::vector<int>>& shapes) {
  if (shapes.empty()) return {};
  int max_dims = 0;
  for (const auto& shape : shapes) { max_dims = std::max(max_dims, static_cast<int>(shape.size())); }
  std::vector<int> output_shape(max_dims, 1);
  for (int i = 0; i < max_dims; ++i) {
    for (const auto& shape : shapes) {
      int dim = static_cast<int>(shape.size()) - max_dims + i;
      int size = dim >= 0 ? shape[dim] : 1;
      if (size != 1) {
        if (output_shape[i] == 1) {
          output_shape[i] = size;
        } else if (output_shape[i] != size) {
          MLLM_ERROR_EXIT(mllm::ExitCode::kShapeError, "Broadcast shape mismatch");
          return {};
        }
      }
    }
  }

  return output_shape;
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

namespace mllm::aops {

EqualOp::EqualOp(const EqualOpOptions& options) : BaseOp(OpTypes::kEqual), options_(options) {}

void EqualOp::load(const ParameterFile::ptr_t& ploader) {}

void EqualOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  if (options_.getInputsConstant(1)) {
    i_irs[1]->setAttr("constant", ctx->create<ir::VectorFP32Attr>(inputs[1].toVector<float>()));
  }
  ctx->create<ir::linalg::EqualOp>(shared_from_this(), i_irs, o_irs);
}

void EqualOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_WARN("EqualOp::forward is not implemented");
}

void EqualOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  std::vector<std::vector<int>> input_shapes;
  input_shapes.reserve(inputs.size());
  for (const auto& input : inputs) { input_shapes.push_back(input.shape()); }
  std::vector<int> output_shape = broadcastShapes(input_shapes);
  if (output_shape.empty()) { output_shape = inputs[0].shape(); }
  Tensor output_0 = Tensor::empty(output_shape, kUInt8, inputs[0].device());
  outputs.emplace_back(output_0);
}

void EqualOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
