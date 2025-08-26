/**
 * @file ReduceOps.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-01
 *
 */
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

ReduceMaxOp::ReduceMaxOp(const ReduceMaxOpOptions& options) : BaseOp(OpTypes::kReduceMax), options_(options) {}

void ReduceMaxOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ReduceMaxOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ReduceMaxOp>(shared_from_this(), i_irs, o_irs);
}

void ReduceMaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ReduceMaxOp::forward not implemented in aops base.");
}

void ReduceMaxOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];

  if (!input.isNil()) {
    auto input_shape = input.shape();
    std::vector<int32_t> output_shape;

    if (options_.dim == 0x7fffffff) {
      // Reduce over all dimensions, result is a scalar
      if (options_.keep_dim) {
        output_shape.resize(input_shape.size(), 1);
      } else {
        output_shape = {1};
      }
    } else {
      int32_t dim = options_.dim;
      // Handle negative dimension index
      if (dim < 0) { dim += input_shape.size(); }

      if (options_.keep_dim) {
        // Keep dimensions, so the reduced dimension becomes 1
        output_shape = input_shape;
        output_shape[dim] = 1;
      } else {
        // Don't keep dimensions, so remove the reduced dimension
        for (int i = 0; i < input_shape.size(); ++i) {
          if (i != dim) { output_shape.push_back(input_shape[i]); }
        }
      }
    }

    outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
  }
}

void ReduceMaxOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ReduceMinOp::ReduceMinOp(const ReduceMinOpOptions& options) : BaseOp(OpTypes::kReduceMin), options_(options) {}

void ReduceMinOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ReduceMinOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ReduceMinOp>(shared_from_this(), i_irs, o_irs);
}

void ReduceMinOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ReduceMinOp::forward not implemented in aops base.");
}

void ReduceMinOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];

  if (!input.isNil()) {
    auto input_shape = input.shape();
    std::vector<int32_t> output_shape;

    if (options_.dim == 0x7fffffff) {
      // Reduce over all dimensions, result is a scalar
      if (options_.keep_dim) {
        output_shape.resize(input_shape.size(), 1);
      } else {
        output_shape = {1};
      }
    } else {
      int32_t dim = options_.dim;
      // Handle negative dimension index
      if (dim < 0) { dim += input_shape.size(); }

      if (options_.keep_dim) {
        // Keep dimensions, so the reduced dimension becomes 1
        output_shape = input_shape;
        output_shape[dim] = 1;
      } else {
        // Don't keep dimensions, so remove the reduced dimension
        for (int i = 0; i < input_shape.size(); ++i) {
          if (i != dim) { output_shape.push_back(input_shape[i]); }
        }
      }
    }

    outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
  }
}

void ReduceMinOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ReduceSumOp::ReduceSumOp(const ReduceSumOpOptions& options) : BaseOp(OpTypes::kReduceSum), options_(options) {}

void ReduceSumOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ReduceSumOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ReduceSumOp>(shared_from_this(), i_irs, o_irs);
}

void ReduceSumOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ReduceSumOp::forward not implemented in aops base.");
}

void ReduceSumOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];

  if (!input.isNil()) {
    auto input_shape = input.shape();
    std::vector<int32_t> output_shape;

    if (options_.dim == 0x7fffffff) {
      // Reduce over all dimensions, result is a scalar
      if (options_.keep_dim) {
        output_shape.resize(input_shape.size(), 1);
      } else {
        output_shape = {1};
      }
    } else {
      int32_t dim = options_.dim;
      // Handle negative dimension index
      if (dim < 0) { dim += input_shape.size(); }

      if (options_.keep_dim) {
        // Keep dimensions, so the reduced dimension becomes 1
        output_shape = input_shape;
        output_shape[dim] = 1;
      } else {
        // Don't keep dimensions, so remove the reduced dimension
        for (int i = 0; i < input_shape.size(); ++i) {
          if (i != dim) { output_shape.push_back(input_shape[i]); }
        }
      }
    }

    outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
  }
}

void ReduceSumOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

MeanOp::MeanOp(const MeanOpOptions& options) : BaseOp(OpTypes::kMean), options_(options) {}

void MeanOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void MeanOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::MeanOp>(shared_from_this(), i_irs, o_irs);
}

void MeanOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("MeanOp::forward not implemented in aops base.");
}

void MeanOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];

  if (!input.isNil()) {
    auto input_shape = input.shape();
    std::vector<int32_t> output_shape;

    if (options_.dim == 0x7fffffff) {
      // Reduce over all dimensions, result is a scalar
      if (options_.keep_dim) {
        output_shape.resize(input_shape.size(), 1);
      } else {
        output_shape = {1};
      }
    } else {
      int32_t dim = options_.dim;
      // Handle negative dimension index
      if (dim < 0) { dim += input_shape.size(); }

      if (options_.keep_dim) {
        // Keep dimensions, so the reduced dimension becomes 1
        output_shape = input_shape;
        output_shape[dim] = 1;
      } else {
        // Don't keep dimensions, so remove the reduced dimension
        for (int i = 0; i < input_shape.size(); ++i) {
          if (i != dim) { output_shape.push_back(input_shape[i]); }
        }
      }
    }

    outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
  }
}

void MeanOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
