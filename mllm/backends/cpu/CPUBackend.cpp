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
#include "mllm/backends/cpu/ops/GELUOp.hpp"
#include "mllm/backends/cpu/ops/GraphOps.hpp"
#include "mllm/backends/cpu/ops/LayerNormOp.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/ops/MatMulOp.hpp"
#include "mllm/backends/cpu/ops/PermuteOp.hpp"
#include "mllm/backends/cpu/ops/QuickGELUOp.hpp"
#include "mllm/backends/cpu/ops/RMSNormOp.hpp"
#include "mllm/backends/cpu/ops/ReduceOps.hpp"
#include "mllm/backends/cpu/ops/RepeatOp.hpp"
#include "mllm/backends/cpu/ops/SiLUOp.hpp"
#include "mllm/backends/cpu/ops/SoftmaxOp.hpp"
#include "mllm/backends/cpu/ops/SplitOp.hpp"
#include "mllm/backends/cpu/ops/TransposeOp.hpp"
#include "mllm/backends/cpu/ops/ViewOp.hpp"
#include "mllm/backends/cpu/ops/X2XOp.hpp"

namespace mllm::cpu {

CPUBackend::CPUBackend() : Backend(kCPU, createCPUAllocator()) {
  regOpFactory<CPULinearOpFactory, CPUFillOpFactory, CPUGraphBeginOpFactory, CPUGraphEndOpFactory, CPUAddOpFactory,
               CPUSubOpFactory, CPUMulOpFactory, CPUDivOpFactory, CPUNegOpFactory, CPUReduceMaxOpFactory, CPUReduceMinOpFactory,
               CPUReduceSumOpFactory, CPUTransposeOpFactory, CPUPermuteOpFactory, CPUCastTypeOpFactory, CPUConcatOpFactory,
               CPUContiguousOpFactory, CPUCopyOpFactory, CPUEmbeddingOpFactory, CPUSplitOpFactory, CPUViewOpFactory,
               CPULayerNormOpFactory, CPURepeatOpFactory, CPUX2XOpFactory, CPUSoftmaxOpFactory, CPUSiLUOpFactory,
               CPURMSNormOpFactory, CPUGELUOpFactory, CPUQuickGELUOpFactory, CPUMatMulOpFactory>();
}

std::shared_ptr<CPUBackend> createCPUBackend() { return std::make_shared<CPUBackend>(); }

}  // namespace mllm::cpu
