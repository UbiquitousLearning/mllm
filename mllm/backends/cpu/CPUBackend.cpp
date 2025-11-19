// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/CPUBackend.hpp"
#include "mllm/backends/cpu/CPUAllocator.hpp"

// Ops
#include "mllm/backends/cpu/ops/ArgsortOp.hpp"
#include "mllm/backends/cpu/ops/CastTypeOp.hpp"
#include "mllm/backends/cpu/ops/CausalMaskOp.hpp"
#include "mllm/backends/cpu/ops/CloneOp.hpp"
#include "mllm/backends/cpu/ops/ConcatOp.hpp"
#include "mllm/backends/cpu/ops/ContiguousOp.hpp"
#include "mllm/backends/cpu/ops/Conv1DOp.hpp"
#include "mllm/backends/cpu/ops/Conv2DOp.hpp"
#include "mllm/backends/cpu/ops/Conv3DOp.hpp"
#include "mllm/backends/cpu/ops/CopyOp.hpp"
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/backends/cpu/ops/EmbeddingOp.hpp"
#include "mllm/backends/cpu/ops/FillOp.hpp"
#include "mllm/backends/cpu/ops/FlashAttention2Op.hpp"
#include "mllm/backends/cpu/ops/GELUOp.hpp"
#include "mllm/backends/cpu/ops/InterpolateOp.hpp"
#include "mllm/backends/cpu/ops/LayerNorm2DOp.hpp"
#include "mllm/backends/cpu/ops/MaskedScatterOp.hpp"
#include "mllm/backends/cpu/ops/PadOp.hpp"
#include "mllm/backends/cpu/ops/RadixAttnOp.hpp"
#include "mllm/backends/cpu/ops/ReLUOp.hpp"
#include "mllm/backends/cpu/ops/GraphOps.hpp"
#include "mllm/backends/cpu/ops/ISTFTOp.hpp"
#include "mllm/backends/cpu/ops/IndexOp.hpp"
#include "mllm/backends/cpu/ops/KVCacheOp.hpp"
#include "mllm/backends/cpu/ops/LayerNormOp.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/ops/MatMulOp.hpp"
#include "mllm/backends/cpu/ops/MultimodalRoPEOp.hpp"
#include "mllm/backends/cpu/ops/PagedAttnOp.hpp"
#include "mllm/backends/cpu/ops/ParamOp.hpp"
#include "mllm/backends/cpu/ops/PermuteOp.hpp"
#include "mllm/backends/cpu/ops/QuickGELUOp.hpp"
#include "mllm/backends/cpu/ops/RMSNormOp.hpp"
#include "mllm/backends/cpu/ops/ReduceOps.hpp"
#include "mllm/backends/cpu/ops/RepeatOp.hpp"
#include "mllm/backends/cpu/ops/RoPEOp.hpp"
#include "mllm/backends/cpu/ops/STFTOp.hpp"
#include "mllm/backends/cpu/ops/Scatter2ShardsOp.hpp"
#include "mllm/backends/cpu/ops/SiLUOp.hpp"
#include "mllm/backends/cpu/ops/SliceOp.hpp"
#include "mllm/backends/cpu/ops/SoftmaxOp.hpp"
#include "mllm/backends/cpu/ops/SplitOp.hpp"
#include "mllm/backends/cpu/ops/TopKOp.hpp"
#include "mllm/backends/cpu/ops/TransposeOp.hpp"
#include "mllm/backends/cpu/ops/ViewOp.hpp"
#include "mllm/backends/cpu/ops/VisionRoPEOp.hpp"
#include "mllm/backends/cpu/ops/X2XOp.hpp"
#include "mllm/backends/cpu/ops/StackOp.hpp"

// Context
#include "mllm/engine/Context.hpp"

namespace mllm::cpu {

CPUBackend::CPUBackend() : Backend(kCPU, createCPUAllocator()) {
  regOpFactory<CPULinearOpFactory, CPUFillOpFactory, CPUGraphBeginOpFactory, CPUGraphEndOpFactory, CPUAddOpFactory,
               CPUSubOpFactory, CPUMulOpFactory, CPUDivOpFactory, CPUNegOpFactory, CPUAbsOpFactory, CPULogOpFactory,
               CPUExpOpFactory, CPUSinOpFactory, CPUCosOpFactory, CPUReduceMaxOpFactory, CPUReduceMinOpFactory,
               CPUReduceSumOpFactory, CPUTransposeOpFactory, CPUPermuteOpFactory, CPUCastTypeOpFactory, CPUConcatOpFactory,
               CPUStackOpFactory, CPUContiguousOpFactory, CPUCopyOpFactory, CPUEmbeddingOpFactory, CPUSplitOpFactory,
               CPUViewOpFactory, CPULayerNormOpFactory, CPURepeatOpFactory, CPUX2XOpFactory, CPUSoftmaxOpFactory,
               CPUSiLUOpFactory, CPURMSNormOpFactory, CPUGELUOpFactory, CPUQuickGELUOpFactory, CPUReLUOpFactory,
               CPUMatMulOpFactory, CPUFlashAttention2OpFactory, CPUSliceOpFactory, CPUVisionRoPEOpFactory, CPUParamOpFactory,
               CPUMultimodalRoPEOpFactory, CPURoPEOpFactory, CPUCausalMaskOpFactory, CPUConv1DOpFactory, CPUConv3DOpFactory,
               CPUSTFTOpFactory, CPUISTFTOpFactory, CPUIndexOpFactory, CPUTopKOpFactory, CPUClipOpFactory, CPUMeanOpFactory,
               CPUKVCacheOpFactory, CPUPagedAttnOpFactory, CPUScatter2ShardsOpFactory, CPURadixAttnOpFactory,
               CPUConv2DOpFactory, CPULayerNorm2DOpFactory, CPUInterpolateOpFactory, CPUPadOpFactory, CPUMaskedScatterOpFactory,
               CPUArgsortOpFactory, CPUCloneOpFactory>();
}

CPUBackend::~CPUBackend() {
  if (thread_pool_) { thread_pool_->__threadPoolDestroy(); }
}

HpcThreadPool::ptr_t CPUBackend::getThreadPool() { return thread_pool_; }

void CPUBackend::initThreadPool(int32_t num_threads) {
  thread_pool_ = std::make_shared<HpcThreadPool>(num_threads);
  thread_pool_->activate();
  task_index_ = thread_pool_->acquireTaskSlot();
}

int32_t CPUBackend::taskIndex() { return task_index_; }

std::shared_ptr<CPUBackend> createCPUBackend() { return std::make_shared<CPUBackend>(); }

void idleHpcThreadPool() {
  auto& ctx = Context::instance();
  auto host_bk = std::static_pointer_cast<::mllm::cpu::CPUBackend>(ctx.getBackend(kCPU));
  auto tp = host_bk->getThreadPool();
  if (tp) { tp->idle(); }
}

void wakeupHpcThreadPool() {
  auto& ctx = Context::instance();
  auto host_bk = std::static_pointer_cast<::mllm::cpu::CPUBackend>(ctx.getBackend(kCPU));
  auto tp = host_bk->getThreadPool();
  if (tp) { tp->wakeup(); }
}

}  // namespace mllm::cpu
