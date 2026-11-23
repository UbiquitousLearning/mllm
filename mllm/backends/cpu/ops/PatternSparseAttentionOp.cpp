// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/PatternSparseAttentionOp.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "mllm/backends/cpu/kernels/common/shadownpu/HMXInt8Selector.hpp"
#include "mllm/backends/cpu/kernels/common/shadownpu/AttentionProfiler.hpp"
#include "mllm/backends/cpu/kernels/common/shadownpu/PatternSparseAttention.hpp"
#include "mllm/backends/cpu/kernels/common/shadownpu/PipelineTaskExecutor.hpp"

namespace mllm::cpu {
namespace {

std::array<int32_t, 4> toBhsdStride(const Tensor& tensor) {
  const auto stride = tensor.stride();
  return {stride[0], stride[1], stride[2], stride[3]};
}

struct SharedHmxIndexScratch {
  std::mutex mutex;
  std::unique_ptr<int32_t[]> data;
  size_t capacity = 0;

  int32_t* prepare(size_t required) {
    if (required > capacity) {
      data = std::make_unique<int32_t[]>(required);
      std::memset(data.get(), 0, required * sizeof(int32_t));
      capacity = required;
    }
    return data.get();
  }
};

SharedHmxIndexScratch& sharedHmxIndexScratch() {
  static auto* scratch = new SharedHmxIndexScratch();
  return *scratch;
}

bool sparseAttentionWorkerEnabled() {
  const char* value = std::getenv("MLLM_CPU_ATTENTION_WORKER");
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

shadownpu::PipelineTaskExecutor& sparseAttentionWorker() {
  static auto* executor = new shadownpu::PipelineTaskExecutor("MLLM_CPU_ATTENTION_WORKER_CPU", "MLLM_SPARSE_ATTENTION_THREADS",
                                                              "PatternSparseAttention worker", 1);
  return *executor;
}

int hmxPipelineGroupHeads(int heads) {
  const char* value = std::getenv("MLLM_HMX_PIPELINE_GROUP_HEADS");
  if (value == nullptr || value[0] == '\0') { return 0; }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > heads) {
    throw std::runtime_error("MLLM_HMX_PIPELINE_GROUP_HEADS must be 0 or between 1 and heads");
  }
  if (parsed != 0 && heads % parsed != 0) {
    throw std::runtime_error("MLLM_HMX_PIPELINE_GROUP_HEADS must divide the configured head count");
  }
  return static_cast<int>(parsed);
}

void reportHmxF16SparseOnce() {
  static std::once_flag once;
  std::call_once(once, []() { std::fprintf(stderr, "[SHADOWNPU_HMX_F16_SPARSE] enabled=1\n"); });
}

}  // namespace

CPUPatternSparseAttentionOp::CPUPatternSparseAttentionOp(const aops::PatternSparseAttentionOpOptions& options)
    : aops::PatternSparseAttentionOp(options) {
  if (options_.hmx_selector) {
    hmx_selector_ = std::make_unique<shadownpu::HMXInt8Selector>(static_cast<size_t>(options_.pack_reserve_tokens));
  }
}

CPUPatternSparseAttentionOp::~CPUPatternSparseAttentionOp() = default;

void CPUPatternSparseAttentionOp::prepareHmxRowPlan(int32_t query_heads, int32_t query_len, int32_t key_len) {
  hmx_row_offsets_.assign(static_cast<size_t>(query_heads) * (query_len + 1), 0);
  hmx_topk_head_stride_ = 0;
  for (int32_t head = 0; head < query_heads; ++head) {
    const float density = options_.head_retentions.empty() ? 1.0F - options_.sparsity : options_.head_retentions[head];
    int32_t* offsets = hmx_row_offsets_.data() + static_cast<size_t>(head) * (query_len + 1);
    for (int32_t query_index = 0; query_index < query_len; ++query_index) {
      const int32_t valid =
          options_.causal_mask && query_len > 1 ? std::min(key_len, key_len - query_len + query_index + 1) : key_len;
      const int32_t keep =
          valid <= options_.dense_tokens
              ? valid
              : std::max(1, std::min(valid, static_cast<int32_t>(std::ceil(density * static_cast<float>(valid)))));
      offsets[query_index + 1] = offsets[query_index] + keep;
    }
    hmx_topk_head_stride_ = std::max(hmx_topk_head_stride_, static_cast<size_t>(offsets[query_len]));
  }
}

void CPUPatternSparseAttentionOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  aops::PatternSparseAttentionOp::setup(inputs, outputs);
  if (!options_.hmx_selector) { return; }
  const auto& selector_key = inputs[3];
  if (inputs[0].dtype() != kFloat32 || selector_key.dtype() != kFloat16) {
    throw std::runtime_error("PatternSparseAttention HMX selector requires F32 query and a separate F16 selector key");
  }
  const auto q_shape = inputs[0].shape();
  const auto k_shape = selector_key.shape();
  const int32_t reserve_tokens = std::max(k_shape[2], options_.pack_reserve_tokens);
  prepareHmxRowPlan(q_shape[1], q_shape[2], reserve_tokens);
  const size_t matrix_count = static_cast<size_t>(q_shape[0]) * q_shape[1];
  {
    auto& scratch = sharedHmxIndexScratch();
    std::lock_guard<std::mutex> lock(scratch.mutex);
    (void)scratch.prepare(matrix_count * hmx_topk_head_stride_);
  }
  hmx_query_heads_.resize(matrix_count);
  hmx_key_heads_.resize(matrix_count);
  std::string error;
  if (!hmx_selector_->reserve(q_shape[0], q_shape[1], q_shape[2], reserve_tokens, q_shape[3], &error)) {
    throw std::runtime_error("failed to reserve PatternSparseAttention HMX selector: " + error);
  }
}

void CPUPatternSparseAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  shadownpu::ScopedAttentionProfile sparse_total_profile(shadownpu::AttentionProfileStage::SPARSE_TOTAL,
                                                         shadownpu::CPUAttentionProfiler::enabled());
  if (inputs[0].dtype() != kFloat32) { throw std::runtime_error("PatternSparseAttention CPU reference requires an F32 query"); }

  const auto q_shape = inputs[0].shape();
  const auto k_shape = inputs[1].shape();
  const auto v_shape = inputs[2].shape();
  shadownpu::PatternSparseAttentionF32Params params;
  params.batch = q_shape[0];
  params.query_heads = q_shape[1];
  params.kv_heads = k_shape[1];
  params.query_len = q_shape[2];
  params.key_len = k_shape[2];
  params.query_key_dim = q_shape[3];
  params.value_dim = v_shape[3];
  params.query_stride = toBhsdStride(inputs[0]);
  params.key_stride = toBhsdStride(inputs[1]);
  params.value_stride = toBhsdStride(inputs[2]);
  params.output_stride = toBhsdStride(outputs[0]);
  params.sparsity = options_.sparsity;
  params.causal_mask = options_.causal_mask;
  params.local_ratio = options_.local_ratio;
  params.prefix_tokens = options_.prefix_tokens;
  params.dense_tokens = options_.dense_tokens;
  params.random_pattern = options_.random_pattern;
  params.random_seed = options_.random_seed;
  params.head_retentions = options_.head_retentions.empty() ? nullptr : options_.head_retentions.data();

  uint64_t eligible = 0;
  uint64_t retained = 0;
  for (int32_t batch = 0; batch < params.batch; ++batch) {
    for (int32_t head = 0; head < params.query_heads; ++head) {
      const float density = params.head_retentions == nullptr ? 1.0F - params.sparsity : params.head_retentions[head];
      for (int32_t query_index = 0; query_index < params.query_len; ++query_index) {
        const int32_t valid = params.causal_mask && params.query_len > 1
                                  ? std::min(params.key_len, params.key_len - params.query_len + query_index + 1)
                                  : params.key_len;
        const int32_t keep =
            valid <= params.dense_tokens
                ? valid
                : std::max(1, std::min(valid, static_cast<int32_t>(std::ceil(density * static_cast<float>(valid)))));
        eligible += static_cast<uint64_t>(valid);
        retained += static_cast<uint64_t>(keep);
      }
    }
  }
  shadownpu::CPUSparseSelectionStats::add(eligible, retained);

  const int32_t* external_indices = nullptr;
  const int32_t* external_row_offsets = nullptr;
  size_t external_head_stride = 0;
  size_t external_row_offsets_head_stride = 0;
  const auto execute_sparse_range = [&](size_t matrix_begin, size_t matrix_count, int32_t query_row_begin,
                                        int32_t query_row_count) {
    if (options_.hmx_selector && inputs[1].dtype() == kFloat16 && inputs[2].dtype() == kFloat16) {
      reportHmxF16SparseOnce();
      return shadownpu::patternSparseAttentionHmxF16Range(
          params, inputs[0].ptr<float>(), inputs[1].ptr<mllm_fp16_t>(), inputs[2].ptr<mllm_fp16_t>(), outputs[0].ptr<float>(),
          matrix_begin, matrix_count, query_row_begin, query_row_count, external_indices, external_head_stride,
          external_row_offsets, external_row_offsets_head_stride);
    }
    if (inputs[1].dtype() == kFloat32 && inputs[2].dtype() == kFloat32) {
      return shadownpu::patternSparseAttentionRangeTyped(
          params, inputs[0].ptr<float>(), inputs[1].ptr<float>(), inputs[2].ptr<float>(), outputs[0].ptr<float>(), matrix_begin,
          matrix_count, query_row_begin, query_row_count, external_indices, external_head_stride, external_row_offsets,
          external_row_offsets_head_stride);
    }
    if (inputs[1].dtype() == kFloat32 && inputs[2].dtype() == kFloat16) {
      return shadownpu::patternSparseAttentionRangeTyped(
          params, inputs[0].ptr<float>(), inputs[1].ptr<float>(), inputs[2].ptr<mllm_fp16_t>(), outputs[0].ptr<float>(),
          matrix_begin, matrix_count, query_row_begin, query_row_count, external_indices, external_head_stride,
          external_row_offsets, external_row_offsets_head_stride);
    }
    if (inputs[1].dtype() == kFloat16 && inputs[2].dtype() == kFloat32) {
      return shadownpu::patternSparseAttentionRangeTyped(
          params, inputs[0].ptr<float>(), inputs[1].ptr<mllm_fp16_t>(), inputs[2].ptr<float>(), outputs[0].ptr<float>(),
          matrix_begin, matrix_count, query_row_begin, query_row_count, external_indices, external_head_stride,
          external_row_offsets, external_row_offsets_head_stride);
    }
    if (inputs[1].dtype() == kFloat16 && inputs[2].dtype() == kFloat16) {
      return shadownpu::patternSparseAttentionRangeTyped(
          params, inputs[0].ptr<float>(), inputs[1].ptr<mllm_fp16_t>(), inputs[2].ptr<mllm_fp16_t>(), outputs[0].ptr<float>(),
          matrix_begin, matrix_count, query_row_begin, query_row_count, external_indices, external_head_stride,
          external_row_offsets, external_row_offsets_head_stride);
    }
    throw std::runtime_error("PatternSparseAttention CPU reference requires F32 or F16 key and value tensors");
  };
  std::unique_lock<std::mutex> hmx_scratch_lock;
  bool sparse_pipeline_complete = false;
  if (options_.hmx_selector) {
    const auto& selector_key = inputs[3];
    if (selector_key.dtype() != kFloat16 || hmx_selector_ == nullptr) {
      throw std::runtime_error("PatternSparseAttention HMX selector requires a separate F16 selector key");
    }
    const auto selector_key_stride = toBhsdStride(selector_key);
    if (selector_key_stride[2] != params.query_key_dim || selector_key_stride[3] != 1) {
      throw std::runtime_error("PatternSparseAttention HMX selector requires contiguous F16 selector-key rows");
    }
    prepareHmxRowPlan(params.query_heads, params.query_len, params.key_len);
    const size_t matrix_count = static_cast<size_t>(params.batch) * params.query_heads;
    auto& scratch = sharedHmxIndexScratch();
    hmx_scratch_lock = std::unique_lock<std::mutex>(scratch.mutex);
    int32_t* hmx_indices = scratch.prepare(matrix_count * hmx_topk_head_stride_);
    hmx_query_heads_.resize(matrix_count);
    hmx_key_heads_.resize(matrix_count);
    const int32_t groups = params.query_heads / params.kv_heads;
    for (int32_t batch = 0; batch < params.batch; ++batch) {
      for (int32_t head = 0; head < params.query_heads; ++head) {
        const size_t matrix = static_cast<size_t>(batch) * params.query_heads + head;
        const int32_t kv_head = head / groups;
        hmx_query_heads_[matrix] = inputs[0].cptrAt<float>({batch, head, 0, 0});
        hmx_key_heads_[matrix] = reinterpret_cast<const uint16_t*>(selector_key.cptrAt<mllm_fp16_t>({batch, kv_head, 0, 0}));
      }
    }

    shadownpu::HMXInt8Selector::Request request;
    request.query_f32_heads = hmx_query_heads_.data();
    request.query_f32_row_stride = static_cast<size_t>(params.query_stride[2]);
    request.key_heads = hmx_key_heads_.data();
    request.batch = params.batch;
    request.heads = params.query_heads;
    request.query_len = params.query_len;
    request.key_len = params.key_len;
    request.head_dim = params.query_key_dim;
    request.row_offsets = hmx_row_offsets_.data();
    request.row_offsets_head_stride = static_cast<size_t>(params.query_len + 1);
    request.causal_prefix_tokens = options_.causal_mask ? params.key_len - params.query_len : params.key_len - 1;
    request.topk_indices = hmx_indices;
    request.topk_head_stride = hmx_topk_head_stride_;
    request.thread_count = options_.getThreads();
    request.layer_id = options_.layer_id;
    request.head_retentions = params.head_retentions;
    request.head_retention_count = options_.head_retentions.size();
    request.diagnostic_label = "PatternSparseAttention[layer=" + std::to_string(options_.layer_id) + "]";
    external_indices = hmx_indices;
    external_row_offsets = hmx_row_offsets_.data();
    external_head_stride = hmx_topk_head_stride_;
    external_row_offsets_head_stride = static_cast<size_t>(params.query_len + 1);
    std::atomic<bool> pipeline_succeeded{true};
    if (!options_.head_retentions.empty()) {
      request.pipeline_group_heads = hmxPipelineGroupHeads(params.query_heads);
      request.topk_group_ready = [&](size_t matrix_begin, size_t matrix_count) {
        if (!execute_sparse_range(matrix_begin, matrix_count, 0, params.query_len)) {
          pipeline_succeeded.store(false, std::memory_order_relaxed);
        }
      };
      request.topk_group_rows_ready = [&](size_t matrix_begin, size_t matrix_count, int query_row_begin, int query_row_count) {
        if (!execute_sparse_range(matrix_begin, matrix_count, query_row_begin, query_row_count)) {
          pipeline_succeeded.store(false, std::memory_order_relaxed);
        }
      };
    }
    std::string error;
    shadownpu::ScopedAttentionProfile hmx_select_profile(shadownpu::AttentionProfileStage::HMX_SELECT,
                                                         shadownpu::CPUAttentionProfiler::enabled());
    if (!hmx_selector_->select(request, &error)) {
      if (error.find("non-finite F32 Q value") != std::string::npos) {
        const auto isNonFiniteBits = [](const float* value) noexcept {
          uint32_t bits = 0;
          std::memcpy(&bits, value, sizeof(bits));
          return (bits & UINT32_C(0x7fffffff)) >= UINT32_C(0x7f800000);
        };
        size_t stride_aware_nonfinite = 0;
        size_t selector_linear_nonfinite = 0;
        int32_t first_batch = -1;
        int32_t first_head = -1;
        int32_t first_query = -1;
        int32_t first_dim = -1;
        uint32_t first_bits = 0;
        for (int32_t batch = 0; batch < params.batch; ++batch) {
          for (int32_t head = 0; head < params.query_heads; ++head) {
            const size_t matrix = static_cast<size_t>(batch) * params.query_heads + head;
            const float* query_base = hmx_query_heads_[matrix];
            for (int32_t query_index = 0; query_index < params.query_len; ++query_index) {
              const float* selector_row = query_base + static_cast<size_t>(query_index) * params.query_stride[2];
              for (int32_t dim = 0; dim < params.query_key_dim; ++dim) {
                const float* stride_aware_value = inputs[0].cptrAt<float>({batch, head, query_index, dim});
                if (isNonFiniteBits(stride_aware_value)) {
                  ++stride_aware_nonfinite;
                  if (first_batch < 0) {
                    first_batch = batch;
                    first_head = head;
                    first_query = query_index;
                    first_dim = dim;
                    std::memcpy(&first_bits, stride_aware_value, sizeof(first_bits));
                  }
                }
                if (isNonFiniteBits(selector_row + dim)) { ++selector_linear_nonfinite; }
              }
            }
          }
        }
        error += " query_stride=[" + std::to_string(params.query_stride[0]) + "," + std::to_string(params.query_stride[1]) + ","
                 + std::to_string(params.query_stride[2]) + "," + std::to_string(params.query_stride[3])
                 + "] stride_aware_nonfinite=" + std::to_string(stride_aware_nonfinite)
                 + " selector_linear_nonfinite=" + std::to_string(selector_linear_nonfinite) + " first_nonfinite=["
                 + std::to_string(first_batch) + "," + std::to_string(first_head) + "," + std::to_string(first_query) + ","
                 + std::to_string(first_dim) + "] first_bits=" + std::to_string(first_bits);
      }
      throw std::runtime_error("PatternSparseAttention HMX selection failed: " + error);
    }
    if (!pipeline_succeeded.load(std::memory_order_relaxed)) {
      throw std::runtime_error("PatternSparseAttention pipelined sparse stage rejected its runtime range");
    }
    sparse_pipeline_complete = request.pipeline_group_heads > 0;
  }

  if (!sparse_pipeline_complete) {
    bool succeeded = false;
    const auto execute_sparse = [&]() {
      succeeded = execute_sparse_range(0, static_cast<size_t>(params.batch) * params.query_heads, 0, params.query_len);
    };
    if (sparseAttentionWorkerEnabled()) {
      sparseAttentionWorker().submit(execute_sparse).get();
    } else {
      execute_sparse();
    }
    if (!succeeded) {
      throw std::runtime_error("PatternSparseAttention CPU reference rejected its runtime geometry or options");
    }
  }
}

}  // namespace mllm::cpu
