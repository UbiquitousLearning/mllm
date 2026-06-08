// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendLinearW8A8PluginOperation.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <aclnnop/aclnn_cast.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/graph/AscendClampPluginOperation.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/graph/AscendRoundPluginOperation.hpp"

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

constexpr uint32_t INPUT_NUM = 4;  // x_fp16, weight_int8, bias_i32, deq_scale
constexpr uint32_t OUTPUT_NUM = 1;  // y_fp16
constexpr uint32_t X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_INPUT_INDEX = 1;
constexpr uint32_t BIAS_INPUT_INDEX = 2;
constexpr uint32_t DEQ_SCALE_INPUT_INDEX = 3;
constexpr uint32_t Y_OUTPUT_INDEX = 0;

constexpr uint64_t ALIGNMENT = 512;

using Clock = std::chrono::high_resolution_clock;

inline uint64_t alignUp(uint64_t v) {
  return ((v + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
}

inline double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool isEnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '0';
}

int getEnvInt(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) return default_value;
  return static_cast<int>(parsed);
}

bool shouldProfileW8A8Linear() {
  return isEnvEnabled("MLLM_PROFILE_W8A8_LINEAR");
}

int profileW8A8LinearEvery() {
  return getEnvInt("MLLM_PROFILE_W8A8_LINEAR_EVERY", 50);
}

struct W8A8LinearProfileStats {
  uint64_t setup_calls{0};
  uint64_t exec_calls{0};
  double setup_total_ms{0.0};
  double setup_quant_ms{0.0};
  double setup_linear_ms{0.0};
  double setup_cast_query_ms{0.0};
  double exec_total_ms{0.0};
  double exec_quant_setup_ms{0.0};
  double exec_quant_ms{0.0};
  double exec_cast_query_ms{0.0};
  double exec_cast_ms{0.0};
  double exec_linear_setup_ms{0.0};
  double exec_linear_ms{0.0};
};

std::mutex& w8a8ProfileMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, W8A8LinearProfileStats>& w8a8ProfileMap() {
  static std::unordered_map<std::string, W8A8LinearProfileStats> stats_map;
  return stats_map;
}

void printW8A8ProfileSummary(const std::string& op_name, const W8A8LinearProfileStats& stats) {
  auto avg = [](double total, uint64_t count) { return count == 0 ? 0.0 : total / static_cast<double>(count); };
  std::cout << std::fixed << std::setprecision(3)
            << "[AscendW8A8Profile] op=" << op_name
            << " setup_avg_ms(total=" << avg(stats.setup_total_ms, stats.setup_calls)
            << ", quant_setup=" << avg(stats.setup_quant_ms, stats.setup_calls)
            << ", linear_setup=" << avg(stats.setup_linear_ms, stats.setup_calls)
            << ", cast_query=" << avg(stats.setup_cast_query_ms, stats.setup_calls)
            << ") exec_avg_ms(total=" << avg(stats.exec_total_ms, stats.exec_calls)
            << ", quant_setup=" << avg(stats.exec_quant_setup_ms, stats.exec_calls)
            << ", quant_exec=" << avg(stats.exec_quant_ms, stats.exec_calls)
            << ", cast_query=" << avg(stats.exec_cast_query_ms, stats.exec_calls)
            << ", cast_exec=" << avg(stats.exec_cast_ms, stats.exec_calls)
            << ", linear_setup=" << avg(stats.exec_linear_setup_ms, stats.exec_calls)
            << ", linear_exec=" << avg(stats.exec_linear_ms, stats.exec_calls)
            << ") calls(setup=" << stats.setup_calls
            << ", exec=" << stats.exec_calls << ")\n";
}

void recordW8A8SetupProfile(const std::string& op_name,
                                   double total_ms,
                                   double quant_setup_ms,
                                   double linear_setup_ms,
                                   double cast_query_ms) {
  std::lock_guard<std::mutex> lock(w8a8ProfileMutex());
  auto& stats = w8a8ProfileMap()[op_name];
  stats.setup_calls += 1;
  stats.setup_total_ms += total_ms;
  stats.setup_quant_ms += quant_setup_ms;
  stats.setup_linear_ms += linear_setup_ms;
  stats.setup_cast_query_ms += cast_query_ms;

  const int every = profileW8A8LinearEvery();
  if (stats.setup_calls == 1 || (every > 0 && stats.setup_calls % static_cast<uint64_t>(every) == 0)) {
    printW8A8ProfileSummary(op_name, stats);
  }
}

void recordW8A8ExecProfile(const std::string& op_name,
                                  double total_ms,
                                  double quant_setup_ms,
                                  double quant_exec_ms,
                                  double cast_query_ms,
                                  double cast_exec_ms,
                                  double linear_setup_ms,
                                  double linear_exec_ms) {
  std::lock_guard<std::mutex> lock(w8a8ProfileMutex());
  auto& stats = w8a8ProfileMap()[op_name];
  stats.exec_calls += 1;
  stats.exec_total_ms += total_ms;
  stats.exec_quant_setup_ms += quant_setup_ms;
  stats.exec_quant_ms += quant_exec_ms;
  stats.exec_cast_query_ms += cast_query_ms;
  stats.exec_cast_ms += cast_exec_ms;
  stats.exec_linear_setup_ms += linear_setup_ms;
  stats.exec_linear_ms += linear_exec_ms;

  const int every = profileW8A8LinearEvery();
  if (stats.exec_calls == 1 || (every > 0 && stats.exec_calls % static_cast<uint64_t>(every) == 0)) {
    printW8A8ProfileSummary(op_name, stats);
  }
}

// Build a zero-pointer atb::Tensor descriptor (for Setup workspace queries).
inline atb::Tensor makeTensorDesc(aclDataType dtype,
                                  const atb::Dims& shape,
                                  uint8_t* device_data = nullptr) {
  atb::Tensor t;
  t.desc.dtype  = dtype;
  t.desc.format = ACL_FORMAT_ND;
  t.desc.shape  = shape;
  t.deviceData  = device_data;
  t.dataSize    = atb::Utils::GetTensorSize(t);
  return t;
}

struct CastCacheState {
  aclTensor* src_tensor{nullptr};
  aclTensor* dst_tensor{nullptr};
  aclOpExecutor* executor{nullptr};
  uint64_t workspace_size{0};
  atb::TensorDesc src_desc{};
  atb::TensorDesc dst_desc{};
  void* bound_src_ptr{nullptr};
  void* bound_dst_ptr{nullptr};
  bool repeatable{false};
  std::vector<int64_t> src_dims;
  std::vector<int64_t> src_strides;
  std::vector<int64_t> dst_dims;
  std::vector<int64_t> dst_strides;
};

bool sameTensorDesc(const atb::TensorDesc& lhs, const atb::TensorDesc& rhs) {
  if (lhs.dtype != rhs.dtype || lhs.format != rhs.format || lhs.shape.dimNum != rhs.shape.dimNum) {
    return false;
  }
  for (uint32_t i = 0; i < lhs.shape.dimNum; ++i) {
    if (lhs.shape.dims[i] != rhs.shape.dims[i]) return false;
  }
  return true;
}

aclTensor* createAclTensor(const atb::Tensor& t,
                                  std::vector<int64_t>& dims,
                                  std::vector<int64_t>& strides) {
  const int nd = static_cast<int>(t.desc.shape.dimNum);
  dims.resize(nd);
  strides.resize(nd);
  int64_t stride = 1;
  for (int i = nd - 1; i >= 0; --i) {
    dims[i] = static_cast<int64_t>(t.desc.shape.dims[i]);
    strides[i] = stride;
    stride *= dims[i];
  }
  return aclCreateTensor(dims.data(),
                         nd,
                         t.desc.dtype,
                         strides.data(),
                         /*storageOffset=*/0,
                         ACL_FORMAT_ND,
                         dims.data(),
                         nd,
                         t.deviceData);
}

void destroyCastCacheState(CastCacheState*& cache) {
  if (cache == nullptr) return;
  if (cache->executor != nullptr) {
    aclDestroyAclOpExecutor(cache->executor);
    cache->executor = nullptr;
  }
  if (cache->src_tensor != nullptr) {
    aclDestroyTensor(cache->src_tensor);
    cache->src_tensor = nullptr;
  }
  if (cache->dst_tensor != nullptr) {
    aclDestroyTensor(cache->dst_tensor);
    cache->dst_tensor = nullptr;
  }
  delete cache;
  cache = nullptr;
}

atb::Status updateCastTensorAddrs(CastCacheState* cache, const atb::Tensor& src, const atb::Tensor& dst) {
  if (cache == nullptr || !cache->repeatable) return atb::ERROR_INTERNAL_ERROR;
  if (src.deviceData == nullptr || dst.deviceData == nullptr) return atb::ERROR_INTERNAL_ERROR;

  if (cache->bound_src_ptr != src.deviceData) {
    aclnnStatus ret = aclSetInputTensorAddr(cache->executor, 0, cache->src_tensor, src.deviceData);
    if (ret != ACL_SUCCESS) return atb::ERROR_INTERNAL_ERROR;
    cache->bound_src_ptr = src.deviceData;
  }
  if (cache->bound_dst_ptr != dst.deviceData) {
    aclnnStatus ret = aclSetOutputTensorAddr(cache->executor, 0, cache->dst_tensor, dst.deviceData);
    if (ret != ACL_SUCCESS) return atb::ERROR_INTERNAL_ERROR;
    cache->bound_dst_ptr = dst.deviceData;
  }
  return atb::NO_ERROR;
}

// Create ATB ELEWISE_MULS op with a fixed scalar multiplier.
atb::Operation* createMulsOp(float scalar) {
  atb::infer::ElewiseParam ep;
  ep.elewiseType       = atb::infer::ElewiseParam::ELEWISE_MULS;
  ep.mulsParam.varAttr = scalar;
  atb::Operation* op   = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(ep, &op));
  return op;
}

// Create ATB Linear W8A8 (PER_CHANNEL dequant, transposeB=true, hasBias=true).
atb::Operation* createLinearW8A8Op() {
  atb::infer::LinearParam lp;
  lp.transposeA  = false;
  lp.transposeB  = true;
  lp.hasBias     = true;
  lp.outDataType = ACL_FLOAT16;
  lp.enAccum     = false;
  lp.matmulType  = atb::infer::LinearParam::MATMUL_UNDEFINED;
  lp.quantMode   = atb::infer::LinearParam::PER_CHANNEL;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(lp, &op));
  return op;
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

AscendLinearW8A8PluginOperation::AscendLinearW8A8PluginOperation(float inv_scale_x,
                                                                   std::string name_suffix)
    : inv_scale_x_(inv_scale_x), name_suffix_(std::move(name_suffix)) {
  buildOps();
}

AscendLinearW8A8PluginOperation::~AscendLinearW8A8PluginOperation() {
  auto* cast_cache = static_cast<CastCacheState*>(cast_cache_state_);
  destroyCastCacheState(cast_cache);
  cast_cache_state_ = nullptr;
  // quant_subgraph_op_ owns its child ops (muls/round/clamp); destroy it first.
  if (quant_subgraph_op_ != nullptr) {
    atb::DestroyOperation(quant_subgraph_op_);
    quant_subgraph_op_ = nullptr;
  }
  if (linear_op_ != nullptr) {
    atb::DestroyOperation(linear_op_);
    linear_op_ = nullptr;
  }
}

void AscendLinearW8A8PluginOperation::buildOps() {
  // --- Quant subgraph: x_fp16 → MULS → Round → Clamp → x_clamped_fp16 ---
  AscendGraphBuilder builder;
  builder.beginGraph(
      "AscendLinearW8A8QuantSubGraph" + name_suffix_,
      {"x_fp16"},
      {"x_clamped_fp16"},
      [](const atb::SVector<atb::TensorDesc>& in,
         atb::SVector<atb::TensorDesc>& out) -> atb::Status {
        if (!in.empty() && !out.empty()) {
          out.at(0) = in.at(0);  // same shape and FP16 dtype
        }
        return atb::NO_ERROR;
      });

  // Child ops are passed to builder; graph takes ownership after build().
  atb::Operation* muls_op  = createMulsOp(inv_scale_x_);
  atb::Operation* round_op = createRoundPluginGraphOp();
  atb::Operation* clamp_op = createClampPluginGraphOp(-128.f, 127.f);

  builder.addOperation(muls_op,  {"x_fp16"},    {"x_scaled_fp16"});
  builder.addOperation(round_op, {"x_scaled_fp16"}, {"x_round_fp16"});
  builder.addOperation(clamp_op, {"x_round_fp16"},  {"x_clamped_fp16"});

  quant_subgraph_op_ = builder.build();

  // --- Standalone ATB Linear W8A8 ---
  linear_op_ = createLinearW8A8Op();
}

// ---------------------------------------------------------------------------
// OperationInfra interface
// ---------------------------------------------------------------------------

std::string AscendLinearW8A8PluginOperation::GetName() const {
  return "AscendLinearW8A8PluginOperation" + name_suffix_;
}

uint32_t AscendLinearW8A8PluginOperation::GetInputNum() const  { return INPUT_NUM; }
uint32_t AscendLinearW8A8PluginOperation::GetOutputNum() const { return OUTPUT_NUM; }

atb::Status AscendLinearW8A8PluginOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  if (inTensorDescs.size() != INPUT_NUM || outTensorDescs.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  // Output: same shape as x but last dim replaced by N.
  // e.g. x=[1,8,K] weight=[N,K] → y=[1,8,N]; x=[M,K] → y=[M,N].
  const auto& x_desc  = inTensorDescs.at(X_INPUT_INDEX);
  const auto& w_desc  = inTensorDescs.at(WEIGHT_INPUT_INDEX);
  auto& y_desc        = outTensorDescs.at(Y_OUTPUT_INDEX);
  y_desc.dtype        = ACL_FLOAT16;
  y_desc.format       = ACL_FORMAT_ND;
  y_desc.shape        = x_desc.shape;
  y_desc.shape.dims[y_desc.shape.dimNum - 1] = w_desc.shape.dims[0];  // N
  return atb::NO_ERROR;
}

// ---------------------------------------------------------------------------
// Setup: compute workspace layout, report total workspace size.
// ---------------------------------------------------------------------------

atb::Status AscendLinearW8A8PluginOperation::Setup(const atb::VariantPack& variantPack,
                                                    uint64_t& workspace_size,
                                                    atb::Context* context) {
  if (variantPack.inTensors.size() != INPUT_NUM || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  if (quant_subgraph_op_ == nullptr || linear_op_ == nullptr) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const bool profile_enabled = shouldProfileW8A8Linear();
  const auto total_start = profile_enabled ? Clock::now() : Clock::time_point{};
  double quant_setup_ms = 0.0;
  double linear_setup_ms = 0.0;
  double cast_query_ms = 0.0;

  const auto& x_desc  = variantPack.inTensors.at(X_INPUT_INDEX).desc;
  const auto& w_desc  = variantPack.inTensors.at(WEIGHT_INPUT_INDEX).desc;

  // x_clamped_fp16: same shape as x, FP16
  atb::Tensor x_clamped_fake = makeTensorDesc(ACL_FLOAT16, x_desc.shape);
  ws_layout_.x_clamped_bytes = alignUp(x_clamped_fake.dataSize);

  // x_int8: same shape as x, INT8
  atb::Tensor x_int8_fake = makeTensorDesc(ACL_INT8, x_desc.shape);
  ws_layout_.x_int8_bytes = alignUp(x_int8_fake.dataSize);

  // Query quant_subgraph workspace (manages x_scaled, x_round internally)
  {
    const auto start = profile_enabled ? Clock::now() : Clock::time_point{};
    atb::VariantPack qvp;
    qvp.inTensors.push_back(variantPack.inTensors.at(X_INPUT_INDEX));  // x_fp16 descriptor is fine
    qvp.outTensors.push_back(x_clamped_fake);
    uint64_t q_ws = 0;
    auto st = quant_subgraph_op_->Setup(qvp, q_ws, context);
    if (st != atb::NO_ERROR) return st;
    if (profile_enabled) quant_setup_ms = elapsedMs(start, Clock::now());
    ws_layout_.quant_ws_bytes = alignUp(q_ws);
  }

  // Query linear workspace
  {
    const auto start = profile_enabled ? Clock::now() : Clock::time_point{};
    // Build fake x_int8 tensor with INT8 dtype, same shape as x
    atb::Tensor x_int8_t = makeTensorDesc(ACL_INT8, x_desc.shape);
    // Output y: same ndim as x, last dim replaced by N
    atb::Dims y_shape  = x_desc.shape;
    y_shape.dims[y_shape.dimNum - 1] = w_desc.shape.dims[0];  // N
    atb::Tensor y_fake = makeTensorDesc(ACL_FLOAT16, y_shape);

    atb::VariantPack lvp;
    lvp.inTensors.push_back(x_int8_t);
    lvp.inTensors.push_back(variantPack.inTensors.at(WEIGHT_INPUT_INDEX));  // weight_int8
    lvp.inTensors.push_back(variantPack.inTensors.at(BIAS_INPUT_INDEX));  // bias_i32
    lvp.inTensors.push_back(variantPack.inTensors.at(DEQ_SCALE_INPUT_INDEX));  // deq_scale
    lvp.outTensors.push_back(y_fake);
    uint64_t l_ws = 0;
    auto st = linear_op_->Setup(lvp, l_ws, context);
    if (st != atb::NO_ERROR) return st;
    if (profile_enabled) linear_setup_ms = elapsedMs(start, Clock::now());
    ws_layout_.linear_ws_bytes = alignUp(l_ws);
  }

  // Query aclnnCast workspace (FP16 → INT8, shapes same as x).
  {
    const auto start = profile_enabled ? Clock::now() : Clock::time_point{};
    atb::Tensor x_clamped_q = makeTensorDesc(ACL_FLOAT16, x_desc.shape);
    atb::Tensor x_int8_q    = makeTensorDesc(ACL_INT8,    x_desc.shape);
    auto* cast_cache = static_cast<CastCacheState*>(cast_cache_state_);
    if (cast_cache != nullptr && cast_cache->repeatable &&
        sameTensorDesc(cast_cache->src_desc, x_clamped_q.desc) &&
        sameTensorDesc(cast_cache->dst_desc, x_int8_q.desc)) {
      ws_layout_.cast_ws_bytes = alignUp(cast_cache->workspace_size);
    } else {
      destroyCastCacheState(cast_cache);
      cast_cache_state_ = nullptr;
      auto* new_cache = new CastCacheState();
      new_cache->src_desc = x_clamped_q.desc;
      new_cache->dst_desc = x_int8_q.desc;
      new_cache->bound_src_ptr = x_clamped_q.deviceData;
      new_cache->bound_dst_ptr = x_int8_q.deviceData;
      new_cache->src_tensor = createAclTensor(x_clamped_q,
                                              new_cache->src_dims,
                                              new_cache->src_strides);
      new_cache->dst_tensor = createAclTensor(x_int8_q,
                                              new_cache->dst_dims,
                                              new_cache->dst_strides);
      if (new_cache->src_tensor == nullptr || new_cache->dst_tensor == nullptr) {
        destroyCastCacheState(new_cache);
        return atb::ERROR_INTERNAL_ERROR;
      }
      aclError ret = aclnnCastGetWorkspaceSize(new_cache->src_tensor,
                                               ACL_INT8,
                                               new_cache->dst_tensor,
                                               &new_cache->workspace_size,
                                               &new_cache->executor);
      if (ret != ACL_SUCCESS || new_cache->executor == nullptr) {
        destroyCastCacheState(new_cache);
        return atb::ERROR_INTERNAL_ERROR;
      }
      if (aclSetAclOpExecutorRepeatable(new_cache->executor) != ACL_SUCCESS) {
        destroyCastCacheState(new_cache);
        return atb::ERROR_INTERNAL_ERROR;
      }
      new_cache->repeatable = true;
      ws_layout_.cast_ws_bytes = alignUp(new_cache->workspace_size);
      cast_cache_state_ = new_cache;
    }
    if (profile_enabled) cast_query_ms = elapsedMs(start, Clock::now());
  }

  // Compute layout offsets
  ws_layout_.x_clamped_off = 0;
  ws_layout_.x_int8_off    = ws_layout_.x_clamped_off + ws_layout_.x_clamped_bytes;
  ws_layout_.quant_ws_off  = ws_layout_.x_int8_off    + ws_layout_.x_int8_bytes;
  ws_layout_.cast_ws_off   = ws_layout_.quant_ws_off  + ws_layout_.quant_ws_bytes;
  ws_layout_.linear_ws_off = ws_layout_.cast_ws_off   + ws_layout_.cast_ws_bytes;
  ws_layout_.total         = ws_layout_.linear_ws_off + ws_layout_.linear_ws_bytes;

  workspace_size = ws_layout_.total;
  if (profile_enabled) {
    recordW8A8SetupProfile(GetName(),
                           elapsedMs(total_start, Clock::now()),
                           quant_setup_ms,
                           linear_setup_ms,
                           cast_query_ms);
  }
  return atb::NO_ERROR;
}

// ---------------------------------------------------------------------------
// Execute: run quant subgraph → aclnnCast → linear W8A8.
// ---------------------------------------------------------------------------

atb::Status AscendLinearW8A8PluginOperation::Execute(const atb::VariantPack& variantPack,
                                                      uint8_t* workspace,
                                                      uint64_t workspace_size,
                                                      atb::Context* context) {
  if (variantPack.inTensors.size() != INPUT_NUM || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  if (quant_subgraph_op_ == nullptr || linear_op_ == nullptr) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  if (workspace == nullptr && workspace_size > 0) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const bool profile_enabled = shouldProfileW8A8Linear();
  const auto total_start = profile_enabled ? Clock::now() : Clock::time_point{};
  double quant_setup_ms = 0.0;
  double quant_exec_ms = 0.0;
  double cast_query_ms = 0.0;
  double cast_exec_ms = 0.0;
  double linear_setup_ms = 0.0;
  double linear_exec_ms = 0.0;

  const auto& x_desc = variantPack.inTensors.at(X_INPUT_INDEX).desc;

  // Carve workspace into sub-regions
  uint8_t* x_clamped_ptr = workspace + ws_layout_.x_clamped_off;
  uint8_t* x_int8_ptr    = workspace + ws_layout_.x_int8_off;
  uint8_t* quant_ws_ptr  = workspace + ws_layout_.quant_ws_off;
  uint8_t* cast_ws_ptr   = workspace + ws_layout_.cast_ws_off;
  uint8_t* linear_ws_ptr = workspace + ws_layout_.linear_ws_off;

  // ---- Step A/B/C: quant subgraph (MULS + Round + Clamp) ----
  atb::Tensor x_clamped_t = makeTensorDesc(ACL_FLOAT16, x_desc.shape, x_clamped_ptr);
  {
    atb::VariantPack qvp;
    qvp.inTensors.push_back(variantPack.inTensors.at(X_INPUT_INDEX));
    qvp.outTensors.push_back(x_clamped_t);
    // Re-Setup with actual tensors to bind executor to real pointers
    uint64_t q_ws_size = 0;
    const auto quant_setup_start = profile_enabled ? Clock::now() : Clock::time_point{};
    auto st = quant_subgraph_op_->Setup(qvp, q_ws_size, context);
    if (st != atb::NO_ERROR) return st;
    if (profile_enabled) quant_setup_ms = elapsedMs(quant_setup_start, Clock::now());
    const auto quant_exec_start = profile_enabled ? Clock::now() : Clock::time_point{};
    st = quant_subgraph_op_->Execute(qvp, quant_ws_ptr, ws_layout_.quant_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
    if (profile_enabled) {
      syncGlobalAtbStream();
      quant_exec_ms = elapsedMs(quant_exec_start, Clock::now());
    }
  }

  // ---- Step D: aclnnCast FP16 → INT8 (executor cached when pointers are stable) ----
  atb::Tensor x_int8_t = makeTensorDesc(ACL_INT8, x_desc.shape, x_int8_ptr);
  {
    const auto cast_query_start = profile_enabled ? Clock::now() : Clock::time_point{};
    auto* cast_cache = static_cast<CastCacheState*>(cast_cache_state_);
    if (cast_cache == nullptr || cast_cache->executor == nullptr ||
        !sameTensorDesc(cast_cache->src_desc, x_clamped_t.desc) ||
        !sameTensorDesc(cast_cache->dst_desc, x_int8_t.desc)) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    auto cast_update_st = updateCastTensorAddrs(cast_cache, x_clamped_t, x_int8_t);
    if (profile_enabled) cast_query_ms = elapsedMs(cast_query_start, Clock::now());
    const auto cast_exec_start = profile_enabled ? Clock::now() : Clock::time_point{};
    aclError ret = (cast_update_st == atb::NO_ERROR)
        ? aclnnCast(cast_cache->workspace_size > 0 ? cast_ws_ptr : nullptr,
                    cast_cache->workspace_size,
                    cast_cache->executor,
                    getGlobalAtbStream())
        : ACL_ERROR_INTERNAL_ERROR;
    if (profile_enabled && ret == ACL_SUCCESS) {
      syncGlobalAtbStream();
      cast_exec_ms = elapsedMs(cast_exec_start, Clock::now());
    }
    if (ret != ACL_SUCCESS) return atb::ERROR_INTERNAL_ERROR;
  }

  // ---- Step E: ATB Linear W8A8 ----
  {
    atb::VariantPack lvp;
    lvp.inTensors.push_back(x_int8_t);
    lvp.inTensors.push_back(variantPack.inTensors.at(WEIGHT_INPUT_INDEX));  // weight_int8
    lvp.inTensors.push_back(variantPack.inTensors.at(BIAS_INPUT_INDEX));  // bias_i32
    lvp.inTensors.push_back(variantPack.inTensors.at(DEQ_SCALE_INPUT_INDEX));  // deq_scale
    lvp.outTensors.push_back(variantPack.outTensors.at(Y_OUTPUT_INDEX));
    // Re-Setup to bind executor to current tensor pointers
    uint64_t l_ws_size = 0;
    const auto linear_setup_start = profile_enabled ? Clock::now() : Clock::time_point{};
    auto st = linear_op_->Setup(lvp, l_ws_size, context);
    if (st != atb::NO_ERROR) return st;
    if (profile_enabled) linear_setup_ms = elapsedMs(linear_setup_start, Clock::now());
    const auto linear_exec_start = profile_enabled ? Clock::now() : Clock::time_point{};
    st = linear_op_->Execute(lvp, linear_ws_ptr, ws_layout_.linear_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
    if (profile_enabled) {
      syncGlobalAtbStream();
      linear_exec_ms = elapsedMs(linear_exec_start, Clock::now());
    }
  }

  if (profile_enabled) {
    recordW8A8ExecProfile(GetName(),
                          elapsedMs(total_start, Clock::now()),
                          quant_setup_ms,
                          quant_exec_ms,
                          cast_query_ms,
                          cast_exec_ms,
                          linear_setup_ms,
                          linear_exec_ms);
  }

  return atb::NO_ERROR;
}

atb::Operation* createLinearW8A8PluginGraphOp(float inv_scale_x, std::string name_suffix) {
  return new AscendLinearW8A8PluginOperation(inv_scale_x, std::move(name_suffix));
}

}  // namespace mllm::ascend
