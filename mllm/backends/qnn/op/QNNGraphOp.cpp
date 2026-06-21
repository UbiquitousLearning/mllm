// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNGraphOp.hpp"

namespace mllm::qnn {

QNNGraphBeginOp::QNNGraphBeginOp(const aops::GraphBeginOpOptions& options) : aops::GraphBeginOp(options) {}

QNNGraphEndOp::QNNGraphEndOp(const aops::GraphEndOpOptions& options) : aops::GraphEndOp(options) {}

}  // namespace mllm::qnn
