/**
 * @file GraphOps.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-29
 *
 */

#include "mllm/backends/cpu/ops/GraphOps.hpp"

namespace mllm::cpu {

CPUGraphBeginOp::CPUGraphBeginOp(const aops::GraphBeginOpOptions& options) : aops::GraphBeginOp(options) {}

CPUGraphEndOp::CPUGraphEndOp(const aops::GraphEndOpOptions& options) : aops::GraphEndOp(options) {}

}  // namespace mllm::cpu
