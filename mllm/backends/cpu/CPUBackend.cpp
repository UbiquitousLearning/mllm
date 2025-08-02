// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/CPUBackend.hpp"
#include "mllm/backends/cpu/CPUAllocator.hpp"

// Ops
#include "mllm/backends/cpu/ops/CastTypeOp.hpp"
#include "mllm/backends/cpu/ops/ConcatOp.hpp"
#include "mllm/backends/cpu/ops/ContiguousOp.hpp"
#include "mllm/backends/cpu/ops/CopyOp.hpp"
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/backends/cpu/ops/EmbeddingOp.hpp"
#include "mllm/backends/cpu/ops/FillOp.hpp"
#include "mllm/backends/cpu/ops/GraphOps.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/ops/PermuteOp.hpp"
#include "mllm/backends/cpu/ops/ReduceOps.hpp"
#include "mllm/backends/cpu/ops/TransposeOp.hpp"

namespace mllm::cpu {

CPUBackend::CPUBackend() : Backend(kCPU, createCPUAllocator()) {
  regOpFactory<CPULinearOpFactory, CPUFillOpFactory, CPUGraphBeginOpFactory, CPUGraphEndOpFactory, CPUAddOpFactory,
               CPUSubOpFactory, CPUMulOpFactory, CPUDivOpFactory, CPUNegOpFactory, CPUReduceMaxOpFactory, CPUReduceMinOpFactory,
               CPUReduceSumOpFactory, CPUTransposeOpFactory, CPUPermuteOpFactory, CPUCastTypeOpFactory, CPUConcatOpFactory,
               CPUContiguousOpFactory, CPUCopyOpFactory, CPUEmbeddingOpFactory>();
}

std::shared_ptr<CPUBackend> createCPUBackend() { return std::make_shared<CPUBackend>(); }

}  // namespace mllm::cpu
