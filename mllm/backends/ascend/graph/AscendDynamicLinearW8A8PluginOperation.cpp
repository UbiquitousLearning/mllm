// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendDynamicLinearW8A8PluginOperation.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_max_dim.h>
#include <aclnnop/aclnn_round.h>
#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>
#include <atb/types.h>
#include <atb/utils.h>

#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

constexpr uint32_t INPUT_NUM = 4;
constexpr uint32_t OUTPUT_NUM = 1;
constexpr uint32_t X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_INPUT_INDEX = 1;
constexpr uint32_t BIAS_INPUT_INDEX = 2;
constexpr uint32_t DEQ_SCALE_INPUT_INDEX = 3;
constexpr uint32_t Y_OUTPUT_INDEX = 0;
constexpr uint64_t ALIGNMENT = 512;

inline uint64_t alignUp(uint64_t v) {
  return ((v + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
}

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

inline atb::Dims scaleShape(const atb::Dims& x) {
  atb::Dims scale_shape = x;
  scale_shape.dims[scale_shape.dimNum - 1] = 1;
  return scale_shape;
}

inline atb::Dims outputShape(const atb::Dims& x, int64_t output_channels) {
  atb::Dims output_shape = x;
  output_shape.dims[output_shape.dimNum - 1] = output_channels;
  return output_shape;
}

bool sameTensorDesc(const atb::TensorDesc& a, const atb::TensorDesc& b) {
  if (a.dtype != b.dtype || a.format != b.format || a.shape.dimNum != b.shape.dimNum) {
    return false;
  }
  for (uint32_t i = 0; i < a.shape.dimNum; ++i) {
    if (a.shape.dims[i] != b.shape.dims[i]) {
      return false;
    }
  }
  return true;
}

// Build a row-major aclTensor; out_dims/out_strides are owned by caller.
aclTensor* buildAclTensor(aclDataType dtype,
                                 const atb::Dims& dims,
                                 void* ptr,
                                 std::vector<int64_t>& out_dims,
                                 std::vector<int64_t>& out_strides) {
  const int nd = static_cast<int>(dims.dimNum);
  out_dims.resize(nd);
  out_strides.resize(nd);
  int64_t stride = 1;
  for (int i = nd - 1; i >= 0; --i) {
    out_dims[i] = static_cast<int64_t>(dims.dims[i]);
    out_strides[i] = stride;
    stride *= out_dims[i];
  }
  return aclCreateTensor(out_dims.data(),
                         nd,
                         dtype,
                         out_strides.data(),
                         0,
                         ACL_FORMAT_ND,
                         out_dims.data(),
                         nd,
                         ptr);
}

// ---- 1-in / 1-out ACL op cache ----
struct AclUnaryCache {
  aclTensor* src{nullptr};
  aclTensor* dst{nullptr};
  aclOpExecutor* exec{nullptr};
  uint64_t ws_size{0};
  atb::TensorDesc src_desc{};
  atb::TensorDesc dst_desc{};
  void* bound_src{nullptr};
  void* bound_dst{nullptr};
  bool repeatable{false};
  std::vector<int64_t> src_dims;
  std::vector<int64_t> src_strides;
  std::vector<int64_t> dst_dims;
  std::vector<int64_t> dst_strides;
};

void destroyAclUnaryCache(AclUnaryCache*& c) {
  if (c == nullptr) return;
  if (c->exec != nullptr) {
    aclDestroyAclOpExecutor(c->exec);
    c->exec = nullptr;
  }
  if (c->src != nullptr) {
    aclDestroyTensor(c->src);
    c->src = nullptr;
  }
  if (c->dst != nullptr) {
    aclDestroyTensor(c->dst);
    c->dst = nullptr;
  }
  delete c;
  c = nullptr;
}

atb::Status updateUnaryAddrs(AclUnaryCache* c, void* src_ptr, void* dst_ptr) {
  if (c == nullptr || !c->repeatable) return atb::ERROR_INTERNAL_ERROR;
  if (src_ptr == nullptr || dst_ptr == nullptr) return atb::ERROR_INTERNAL_ERROR;
  if (c->bound_src != src_ptr) {
    if (aclSetInputTensorAddr(c->exec, 0, c->src, src_ptr) != ACL_SUCCESS) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    c->bound_src = src_ptr;
  }
  if (c->bound_dst != dst_ptr) {
    if (aclSetOutputTensorAddr(c->exec, 0, c->dst, dst_ptr) != ACL_SUCCESS) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    c->bound_dst = dst_ptr;
  }
  return atb::NO_ERROR;
}

// Setup abs cache: aclnnAbsGetWorkspaceSize
atb::Status setupAbsCache(void*& cache_ptr,
                                  aclDataType dtype,
                                  const atb::Dims& shape,
                                  uint64_t& ws_bytes) {
  auto* c = static_cast<AclUnaryCache*>(cache_ptr);
  atb::TensorDesc d;
  d.dtype = dtype;
  d.format = ACL_FORMAT_ND;
  d.shape = shape;
  if (c && c->repeatable && sameTensorDesc(c->src_desc, d) && sameTensorDesc(c->dst_desc, d)) {
    ws_bytes = alignUp(c->ws_size);
    return atb::NO_ERROR;
  }
  destroyAclUnaryCache(c);
  cache_ptr = nullptr;
  auto* nc = new AclUnaryCache();
  nc->src_desc = d;
  nc->dst_desc = d;
  nc->src = buildAclTensor(dtype, shape, nullptr, nc->src_dims, nc->src_strides);
  nc->dst = buildAclTensor(dtype, shape, nullptr, nc->dst_dims, nc->dst_strides);
  if (nc->src == nullptr || nc->dst == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  aclError ret = aclnnAbsGetWorkspaceSize(nc->src, nc->dst, &nc->ws_size, &nc->exec);
  if (ret != ACL_SUCCESS || nc->exec == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (aclSetAclOpExecutorRepeatable(nc->exec) != ACL_SUCCESS) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  nc->repeatable = true;
  ws_bytes = alignUp(nc->ws_size);
  cache_ptr = nc;
  return atb::NO_ERROR;
}

// Setup round cache: aclnnRoundGetWorkspaceSize
atb::Status setupRoundCache(void*& cache_ptr,
                                    aclDataType dtype,
                                    const atb::Dims& shape,
                                    uint64_t& ws_bytes) {
  auto* c = static_cast<AclUnaryCache*>(cache_ptr);
  atb::TensorDesc d;
  d.dtype = dtype;
  d.format = ACL_FORMAT_ND;
  d.shape = shape;
  if (c && c->repeatable && sameTensorDesc(c->src_desc, d) && sameTensorDesc(c->dst_desc, d)) {
    ws_bytes = alignUp(c->ws_size);
    return atb::NO_ERROR;
  }
  destroyAclUnaryCache(c);
  cache_ptr = nullptr;
  auto* nc = new AclUnaryCache();
  nc->src_desc = d;
  nc->dst_desc = d;
  nc->src = buildAclTensor(dtype, shape, nullptr, nc->src_dims, nc->src_strides);
  nc->dst = buildAclTensor(dtype, shape, nullptr, nc->dst_dims, nc->dst_strides);
  if (nc->src == nullptr || nc->dst == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  aclError ret = aclnnRoundGetWorkspaceSize(nc->src, nc->dst, &nc->ws_size, &nc->exec);
  if (ret != ACL_SUCCESS || nc->exec == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (aclSetAclOpExecutorRepeatable(nc->exec) != ACL_SUCCESS) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  nc->repeatable = true;
  ws_bytes = alignUp(nc->ws_size);
  cache_ptr = nc;
  return atb::NO_ERROR;
}

// Setup clamp cache: aclnnClampGetWorkspaceSize (scalars fixed at -128 / 127)
atb::Status setupClampCache(void*& cache_ptr,
                                    aclDataType dtype,
                                    const atb::Dims& shape,
                                    uint64_t& ws_bytes) {
  auto* c = static_cast<AclUnaryCache*>(cache_ptr);
  atb::TensorDesc d;
  d.dtype = dtype;
  d.format = ACL_FORMAT_ND;
  d.shape = shape;
  if (c && c->repeatable && sameTensorDesc(c->src_desc, d) && sameTensorDesc(c->dst_desc, d)) {
    ws_bytes = alignUp(c->ws_size);
    return atb::NO_ERROR;
  }
  destroyAclUnaryCache(c);
  cache_ptr = nullptr;
  auto* nc = new AclUnaryCache();
  nc->src_desc = d;
  nc->dst_desc = d;
  nc->src = buildAclTensor(dtype, shape, nullptr, nc->src_dims, nc->src_strides);
  nc->dst = buildAclTensor(dtype, shape, nullptr, nc->dst_dims, nc->dst_strides);
  if (nc->src == nullptr || nc->dst == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  float mn = -128.0f, mx = 127.0f;
  aclScalar* acl_min = aclCreateScalar(&mn, ACL_FLOAT);
  aclScalar* acl_max = aclCreateScalar(&mx, ACL_FLOAT);
  aclError ret = aclnnClampGetWorkspaceSize(nc->src, acl_min, acl_max, nc->dst, &nc->ws_size, &nc->exec);
  aclDestroyScalar(acl_min);
  aclDestroyScalar(acl_max);
  if (ret != ACL_SUCCESS || nc->exec == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (aclSetAclOpExecutorRepeatable(nc->exec) != ACL_SUCCESS) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  nc->repeatable = true;
  ws_bytes = alignUp(nc->ws_size);
  cache_ptr = nc;
  return atb::NO_ERROR;
}

// Setup cast cache: aclnnCastGetWorkspaceSize
atb::Status setupCastCache(void*& cache_ptr,
                                   aclDataType src_dtype,
                                   aclDataType dst_dtype,
                                   const atb::Dims& shape,
                                   uint64_t& ws_bytes) {
  auto* c = static_cast<AclUnaryCache*>(cache_ptr);
  atb::TensorDesc sd;
  sd.dtype = src_dtype;
  sd.format = ACL_FORMAT_ND;
  sd.shape = shape;
  atb::TensorDesc dd;
  dd.dtype = dst_dtype;
  dd.format = ACL_FORMAT_ND;
  dd.shape = shape;
  if (c && c->repeatable && sameTensorDesc(c->src_desc, sd) && sameTensorDesc(c->dst_desc, dd)) {
    ws_bytes = alignUp(c->ws_size);
    return atb::NO_ERROR;
  }
  destroyAclUnaryCache(c);
  cache_ptr = nullptr;
  auto* nc = new AclUnaryCache();
  nc->src_desc = sd;
  nc->dst_desc = dd;
  nc->src = buildAclTensor(src_dtype, shape, nullptr, nc->src_dims, nc->src_strides);
  nc->dst = buildAclTensor(dst_dtype, shape, nullptr, nc->dst_dims, nc->dst_strides);
  if (nc->src == nullptr || nc->dst == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  aclError ret = aclnnCastGetWorkspaceSize(nc->src, dst_dtype, nc->dst, &nc->ws_size, &nc->exec);
  if (ret != ACL_SUCCESS || nc->exec == nullptr) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (aclSetAclOpExecutorRepeatable(nc->exec) != ACL_SUCCESS) {
    destroyAclUnaryCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  nc->repeatable = true;
  ws_bytes = alignUp(nc->ws_size);
  cache_ptr = nc;
  return atb::NO_ERROR;
}

// ---- MaxDim cache (1-in, 2-out: val + idx) ----
struct MaxDimCache {
  aclTensor* src{nullptr};
  aclTensor* val{nullptr};
  aclTensor* idx{nullptr};
  aclOpExecutor* exec{nullptr};
  uint64_t ws_size{0};
  atb::TensorDesc src_desc{};
  void* bound_src{nullptr};
  void* bound_val{nullptr};
  void* bound_idx{nullptr};
  bool repeatable{false};
  std::vector<int64_t> src_dims;
  std::vector<int64_t> src_strides;
  std::vector<int64_t> val_dims;
  std::vector<int64_t> val_strides;
  std::vector<int64_t> idx_dims;
  std::vector<int64_t> idx_strides;
};

void destroyMaxDimCache(MaxDimCache*& c) {
  if (c == nullptr) return;
  if (c->exec != nullptr) {
    aclDestroyAclOpExecutor(c->exec);
    c->exec = nullptr;
  }
  if (c->src != nullptr) {
    aclDestroyTensor(c->src);
    c->src = nullptr;
  }
  if (c->val != nullptr) {
    aclDestroyTensor(c->val);
    c->val = nullptr;
  }
  if (c->idx != nullptr) {
    aclDestroyTensor(c->idx);
    c->idx = nullptr;
  }
  delete c;
  c = nullptr;
}

atb::Status setupMaxDimCache(void*& cache_ptr,
                                     const atb::Dims& x_shape,
                                     const atb::Dims& s_shape,
                                     uint64_t& ws_bytes) {
  auto* c = static_cast<MaxDimCache*>(cache_ptr);
  atb::TensorDesc xd;
  xd.dtype = ACL_FLOAT16;
  xd.format = ACL_FORMAT_ND;
  xd.shape = x_shape;
  if (c && c->repeatable && sameTensorDesc(c->src_desc, xd)) {
    ws_bytes = alignUp(c->ws_size);
    return atb::NO_ERROR;
  }
  auto* old = c;
  destroyMaxDimCache(old);
  cache_ptr = nullptr;
  auto* nc = new MaxDimCache();
  nc->src_desc = xd;
  nc->src = buildAclTensor(ACL_FLOAT16, x_shape, nullptr, nc->src_dims, nc->src_strides);
  nc->val = buildAclTensor(ACL_FLOAT16, s_shape, nullptr, nc->val_dims, nc->val_strides);
  nc->idx = buildAclTensor(ACL_INT32,   s_shape, nullptr, nc->idx_dims, nc->idx_strides);
  if (nc->src == nullptr || nc->val == nullptr || nc->idx == nullptr) {
    destroyMaxDimCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  const int64_t dim_val = static_cast<int64_t>(x_shape.dimNum) - 1;
  aclError ret = aclnnMaxDimGetWorkspaceSize(nc->src, dim_val, true, nc->val, nc->idx,
                                              &nc->ws_size, &nc->exec);
  if (ret != ACL_SUCCESS || nc->exec == nullptr) {
    destroyMaxDimCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (aclSetAclOpExecutorRepeatable(nc->exec) != ACL_SUCCESS) {
    destroyMaxDimCache(nc);
    return atb::ERROR_INTERNAL_ERROR;
  }
  nc->repeatable = true;
  ws_bytes = alignUp(nc->ws_size);
  cache_ptr = nc;
  return atb::NO_ERROR;
}

atb::Status updateMaxDimAddrs(MaxDimCache* c, void* src_ptr, void* val_ptr, void* idx_ptr) {
  if (c == nullptr || !c->repeatable) return atb::ERROR_INTERNAL_ERROR;
  if (src_ptr == nullptr || val_ptr == nullptr || idx_ptr == nullptr) return atb::ERROR_INTERNAL_ERROR;
  if (c->bound_src != src_ptr) {
    if (aclSetInputTensorAddr(c->exec, 0, c->src, src_ptr) != ACL_SUCCESS) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    c->bound_src = src_ptr;
  }
  if (c->bound_val != val_ptr) {
    if (aclSetOutputTensorAddr(c->exec, 0, c->val, val_ptr) != ACL_SUCCESS) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    c->bound_val = val_ptr;
  }
  if (c->bound_idx != idx_ptr) {
    if (aclSetOutputTensorAddr(c->exec, 1, c->idx, idx_ptr) != ACL_SUCCESS) {
      return atb::ERROR_INTERNAL_ERROR;
    }
    c->bound_idx = idx_ptr;
  }
  return atb::NO_ERROR;
}

// ATB op helpers
atb::Operation* createMulsOp(float s) {
  atb::infer::ElewiseParam p;
  p.elewiseType = atb::infer::ElewiseParam::ELEWISE_MULS;
  p.mulsParam.varAttr = s;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(p, &op));
  return op;
}

atb::Operation* createDivOp() {
  atb::infer::ElewiseParam p;
  p.elewiseType = atb::infer::ElewiseParam::ELEWISE_REALDIV;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(p, &op));
  return op;
}

atb::Operation* createMulOp() {
  atb::infer::ElewiseParam p;
  p.elewiseType = atb::infer::ElewiseParam::ELEWISE_MUL;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(p, &op));
  return op;
}

atb::Operation* createLinearW8A8Op() {
  atb::infer::LinearParam lp;
  lp.transposeA = false;
  lp.transposeB = true;
  lp.hasBias = true;
  lp.outDataType = ACL_FLOAT16;
  lp.enAccum = false;
  lp.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
  lp.quantMode  = atb::infer::LinearParam::PER_CHANNEL;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(lp, &op));
  return op;
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

// ---------------------------------------------------------------------------
AscendDynamicLinearW8A8PluginOperation::AscendDynamicLinearW8A8PluginOperation(
    std::string name_suffix)
    : name_suffix_(std::move(name_suffix)) {
  buildAtbOps();
}

AscendDynamicLinearW8A8PluginOperation::~AscendDynamicLinearW8A8PluginOperation() {
  if (muls_op_ != nullptr) {
    atb::DestroyOperation(muls_op_);
    muls_op_ = nullptr;
  }
  if (div_op_ != nullptr) {
    atb::DestroyOperation(div_op_);
    div_op_ = nullptr;
  }
  if (linear_op_ != nullptr) {
    atb::DestroyOperation(linear_op_);
    linear_op_ = nullptr;
  }
  if (mul_dequant_op_ != nullptr) {
    atb::DestroyOperation(mul_dequant_op_);
    mul_dequant_op_ = nullptr;
  }

  auto* abs_c = static_cast<AclUnaryCache*>(abs_cache_);
  auto* round_c = static_cast<AclUnaryCache*>(round_cache_);
  auto* clamp_c = static_cast<AclUnaryCache*>(clamp_cache_);
  auto* cast_c = static_cast<AclUnaryCache*>(cast_cache_);
  auto* maxdim_c = static_cast<MaxDimCache*>(maxdim_cache_);
  destroyAclUnaryCache(abs_c);
  destroyAclUnaryCache(round_c);
  destroyAclUnaryCache(clamp_c);
  destroyAclUnaryCache(cast_c);
  destroyMaxDimCache(maxdim_c);
  abs_cache_ = round_cache_ = clamp_cache_ = cast_cache_ = maxdim_cache_ = nullptr;
}

void AscendDynamicLinearW8A8PluginOperation::buildAtbOps() {
  muls_op_        = createMulsOp(1.0f / 127.0f);
  div_op_         = createDivOp();
  linear_op_      = createLinearW8A8Op();
  mul_dequant_op_ = createMulOp();
}

std::string AscendDynamicLinearW8A8PluginOperation::GetName() const {
  return "AscendDynamicLinearW8A8PluginOperation" + name_suffix_;
}
uint32_t AscendDynamicLinearW8A8PluginOperation::GetInputNum() const {
  return INPUT_NUM;
}

uint32_t AscendDynamicLinearW8A8PluginOperation::GetOutputNum() const {
  return OUTPUT_NUM;
}

atb::Status AscendDynamicLinearW8A8PluginOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& in, atb::SVector<atb::TensorDesc>& out) const {
  if (in.size() != INPUT_NUM || out.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  out.at(Y_OUTPUT_INDEX).dtype = ACL_FLOAT16;
  out.at(Y_OUTPUT_INDEX).format = ACL_FORMAT_ND;
  out.at(Y_OUTPUT_INDEX).shape = in.at(X_INPUT_INDEX).shape;
  out.at(Y_OUTPUT_INDEX).shape.dims[out.at(Y_OUTPUT_INDEX).shape.dimNum - 1] = in.at(WEIGHT_INPUT_INDEX).shape.dims[0];
  return atb::NO_ERROR;
}

// ---------------------------------------------------------------------------
// Setup: build/validate ACL executor caches + query ATB workspace sizes.
// ---------------------------------------------------------------------------
atb::Status AscendDynamicLinearW8A8PluginOperation::Setup(const atb::VariantPack& vp,
                                                           uint64_t& workspace_size,
                                                           atb::Context* context) {
  if (vp.inTensors.size() != INPUT_NUM || vp.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const auto& x_desc = vp.inTensors.at(X_INPUT_INDEX).desc;
  const auto& w_desc = vp.inTensors.at(WEIGHT_INPUT_INDEX).desc;
  const atb::Dims s_shape = scaleShape(x_desc.shape);
  const atb::Dims y_shape = outputShape(x_desc.shape, w_desc.shape.dims[0]);

  // Buffer sizes
  atb::Tensor x_fp16_fake = makeTensorDesc(ACL_FLOAT16, x_desc.shape);
  atb::Tensor s_fp16_fake = makeTensorDesc(ACL_FLOAT16, s_shape);
  atb::Tensor s_i32_fake  = makeTensorDesc(ACL_INT32,   s_shape);
  atb::Tensor x_i8_fake   = makeTensorDesc(ACL_INT8,    x_desc.shape);
  atb::Tensor y_fp16_fake = makeTensorDesc(ACL_FLOAT16, y_shape);

  ws_layout_.buf_a_bytes   = alignUp(x_fp16_fake.dataSize);
  ws_layout_.max_abs_bytes = alignUp(s_fp16_fake.dataSize);
  ws_layout_.idx_buf_bytes = alignUp(s_i32_fake.dataSize);
  ws_layout_.scale_x_bytes = alignUp(s_fp16_fake.dataSize);
  ws_layout_.buf_b_bytes   = alignUp(x_fp16_fake.dataSize);
  ws_layout_.x_int8_bytes  = alignUp(x_i8_fake.dataSize);
  ws_layout_.y_pre_bytes   = alignUp(y_fp16_fake.dataSize);

  // ACL executor caches (rebuilds only when tensor shape changes)
  auto st = setupAbsCache(abs_cache_,     ACL_FLOAT16, x_desc.shape, ws_layout_.abs_ws_bytes);
  if (st != atb::NO_ERROR) return st;
  st = setupMaxDimCache(maxdim_cache_, x_desc.shape, s_shape, ws_layout_.maxdim_ws_bytes);
  if (st != atb::NO_ERROR) return st;
  st = setupRoundCache(round_cache_,   ACL_FLOAT16, x_desc.shape, ws_layout_.round_ws_bytes);
  if (st != atb::NO_ERROR) return st;
  st = setupClampCache(clamp_cache_,   ACL_FLOAT16, x_desc.shape, ws_layout_.clamp_ws_bytes);
  if (st != atb::NO_ERROR) return st;
  st = setupCastCache(cast_cache_,     ACL_FLOAT16, ACL_INT8, x_desc.shape, ws_layout_.cast_ws_bytes);
  if (st != atb::NO_ERROR) return st;

  // ATB op workspace queries
  {
    atb::Tensor i = makeTensorDesc(ACL_FLOAT16, s_shape);
    atb::Tensor o = makeTensorDesc(ACL_FLOAT16, s_shape);
    atb::VariantPack p;
    p.inTensors = {i};
    p.outTensors = {o};
    st = muls_op_->Setup(p, ws_layout_.muls_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
    ws_layout_.muls_ws_bytes = alignUp(ws_layout_.muls_ws_bytes);
  }
  {
    atb::Tensor a = makeTensorDesc(ACL_FLOAT16, x_desc.shape);
    atb::Tensor b = makeTensorDesc(ACL_FLOAT16, s_shape);
    atb::Tensor o = makeTensorDesc(ACL_FLOAT16, x_desc.shape);
    atb::VariantPack p;
    p.inTensors = {a, b};
    p.outTensors = {o};
    st = div_op_->Setup(p, ws_layout_.div_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
    ws_layout_.div_ws_bytes = alignUp(ws_layout_.div_ws_bytes);
  }
  {
    atb::Tensor xi = makeTensorDesc(ACL_INT8,    x_desc.shape);
    atb::Tensor yo = makeTensorDesc(ACL_FLOAT16, y_shape);
    atb::VariantPack p;
    p.inTensors  = {xi,
                    vp.inTensors.at(WEIGHT_INPUT_INDEX),
                    vp.inTensors.at(BIAS_INPUT_INDEX),
                    vp.inTensors.at(DEQ_SCALE_INPUT_INDEX)};
    p.outTensors = {yo};
    st = linear_op_->Setup(p, ws_layout_.linear_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
    ws_layout_.linear_ws_bytes = alignUp(ws_layout_.linear_ws_bytes);
  }
  {
    atb::Tensor a = makeTensorDesc(ACL_FLOAT16, y_shape);
    atb::Tensor b = makeTensorDesc(ACL_FLOAT16, s_shape);
    atb::Tensor o = vp.outTensors.at(Y_OUTPUT_INDEX);
    atb::VariantPack p;
    p.inTensors = {a, b};
    p.outTensors = {o};
    st = mul_dequant_op_->Setup(p, ws_layout_.mul_dequant_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
    ws_layout_.mul_dequant_ws_bytes = alignUp(ws_layout_.mul_dequant_ws_bytes);
  }

  // Layout offsets
  uint64_t off = 0;
  auto next = [&](uint64_t bytes) {
    uint64_t o = off;
    off += bytes;
    return o;
  };
  ws_layout_.buf_a_off          = next(ws_layout_.buf_a_bytes);
  ws_layout_.max_abs_off        = next(ws_layout_.max_abs_bytes);
  ws_layout_.idx_buf_off        = next(ws_layout_.idx_buf_bytes);
  ws_layout_.scale_x_off        = next(ws_layout_.scale_x_bytes);
  ws_layout_.buf_b_off          = next(ws_layout_.buf_b_bytes);
  ws_layout_.x_int8_off         = next(ws_layout_.x_int8_bytes);
  ws_layout_.y_pre_off          = next(ws_layout_.y_pre_bytes);
  ws_layout_.abs_ws_off         = next(ws_layout_.abs_ws_bytes);
  ws_layout_.maxdim_ws_off      = next(ws_layout_.maxdim_ws_bytes);
  ws_layout_.round_ws_off       = next(ws_layout_.round_ws_bytes);
  ws_layout_.clamp_ws_off       = next(ws_layout_.clamp_ws_bytes);
  ws_layout_.cast_ws_off        = next(ws_layout_.cast_ws_bytes);
  ws_layout_.muls_ws_off        = next(ws_layout_.muls_ws_bytes);
  ws_layout_.div_ws_off         = next(ws_layout_.div_ws_bytes);
  ws_layout_.linear_ws_off      = next(ws_layout_.linear_ws_bytes);
  ws_layout_.mul_dequant_ws_off = next(ws_layout_.mul_dequant_ws_bytes);
  ws_layout_.total              = off;
  workspace_size                = ws_layout_.total;
  return atb::NO_ERROR;
}

// ---------------------------------------------------------------------------
// Execute: all ACL ops reuse cached executors (pointer-update only in hot path).
// ---------------------------------------------------------------------------
atb::Status AscendDynamicLinearW8A8PluginOperation::Execute(const atb::VariantPack& vp,
                                                             uint8_t* workspace,
                                                             uint64_t workspace_size,
                                                             atb::Context* context) {
  if (vp.inTensors.size() != INPUT_NUM || vp.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  if (muls_op_ == nullptr || div_op_ == nullptr || linear_op_ == nullptr || mul_dequant_op_ == nullptr) {
    return atb::ERROR_INTERNAL_ERROR;
  }
  if (workspace == nullptr && workspace_size > 0) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }

  const auto& x_desc = vp.inTensors.at(X_INPUT_INDEX).desc;
  const auto& w_desc = vp.inTensors.at(WEIGHT_INPUT_INDEX).desc;
  const atb::Dims s_shape = scaleShape(x_desc.shape);
  const atb::Dims y_shape = outputShape(x_desc.shape, w_desc.shape.dims[0]);
  auto* stream = getGlobalAtbStream();

  uint8_t* buf_a_ptr         = workspace + ws_layout_.buf_a_off;
  uint8_t* max_abs_ptr        = workspace + ws_layout_.max_abs_off;
  uint8_t* idx_buf_ptr        = workspace + ws_layout_.idx_buf_off;
  uint8_t* scale_x_ptr        = workspace + ws_layout_.scale_x_off;
  uint8_t* buf_b_ptr          = workspace + ws_layout_.buf_b_off;
  uint8_t* x_int8_ptr         = workspace + ws_layout_.x_int8_off;
  uint8_t* y_pre_ptr          = workspace + ws_layout_.y_pre_off;
  uint8_t* abs_ws_ptr         = workspace + ws_layout_.abs_ws_off;
  uint8_t* maxdim_ws_ptr      = workspace + ws_layout_.maxdim_ws_off;
  uint8_t* round_ws_ptr       = workspace + ws_layout_.round_ws_off;
  uint8_t* clamp_ws_ptr       = workspace + ws_layout_.clamp_ws_off;
  uint8_t* cast_ws_ptr        = workspace + ws_layout_.cast_ws_off;
  uint8_t* muls_ws_ptr        = workspace + ws_layout_.muls_ws_off;
  uint8_t* div_ws_ptr         = workspace + ws_layout_.div_ws_off;
  uint8_t* linear_ws_ptr      = workspace + ws_layout_.linear_ws_off;
  uint8_t* mul_dequant_ws_ptr = workspace + ws_layout_.mul_dequant_ws_off;

  // ATB binary/unary helpers (Setup+Execute in Execute phase)
  auto run_atb_unary = [&](atb::Operation* op,
                          aclDataType idt, const atb::Dims& ish, void* iptr,
                          aclDataType odt, const atb::Dims& osh, void* optr,
                          uint8_t* ws, uint64_t ws_bytes) -> atb::Status {
      atb::Tensor ti = makeTensorDesc(idt, ish, static_cast<uint8_t*>(iptr));
      atb::Tensor to = makeTensorDesc(odt, osh, static_cast<uint8_t*>(optr));
      atb::VariantPack p;
      p.inTensors = {ti};
      p.outTensors = {to};
      uint64_t q = 0;
      auto st = op->Setup(p, q, context);
    if (st != atb::NO_ERROR) return st;
    return op->Execute(p, ws, ws_bytes, context);
  };
  auto run_atb_binary = [&](atb::Operation* op,
                           aclDataType adt, const atb::Dims& ash, void* aptr,
                           aclDataType bdt, const atb::Dims& bsh, void* bptr,
                           aclDataType odt, const atb::Dims& osh, void* optr,
                           uint8_t* ws, uint64_t ws_bytes) -> atb::Status {
    atb::Tensor ta = makeTensorDesc(adt, ash, static_cast<uint8_t*>(aptr));
      atb::Tensor tb = makeTensorDesc(bdt, bsh, static_cast<uint8_t*>(bptr));
      atb::Tensor to = makeTensorDesc(odt, osh, static_cast<uint8_t*>(optr));
      atb::VariantPack p;
      p.inTensors = {ta, tb};
      p.outTensors = {to};
      uint64_t q = 0;
      auto st = op->Setup(p, q, context);
    if (st != atb::NO_ERROR) return st;
    return op->Execute(p, ws, ws_bytes, context);
  };

  // ---- Step 1: buf_a = abs(x)  [cached executor] ----
  {
    auto* c = static_cast<AclUnaryCache*>(abs_cache_);
    if (!c || !c->exec) return atb::ERROR_INTERNAL_ERROR;
    auto st = updateUnaryAddrs(c, vp.inTensors.at(X_INPUT_INDEX).deviceData, buf_a_ptr);
    if (st != atb::NO_ERROR) return st;
    MLLM_ACL_CHECK(aclnnAbs(ws_layout_.abs_ws_bytes > 0 ? abs_ws_ptr : nullptr,
                              c->ws_size, c->exec, stream));
  }

  // ---- Step 2: max_abs = maxDim(buf_a, dim=-1, keepdim=T)  [cached executor] ----
  {
    auto* c = static_cast<MaxDimCache*>(maxdim_cache_);
    if (!c || !c->exec) return atb::ERROR_INTERNAL_ERROR;
    auto st = updateMaxDimAddrs(c, buf_a_ptr, max_abs_ptr, idx_buf_ptr);
    if (st != atb::NO_ERROR) return st;
    MLLM_ACL_CHECK(aclnnMaxDim(ws_layout_.maxdim_ws_bytes > 0 ? maxdim_ws_ptr : nullptr,
                                 c->ws_size, c->exec, stream));
  }

  // ---- Step 3: scale_x = max_abs * (1/127)  [ATB MULS] ----
  {
    auto st = run_atb_unary(muls_op_,
                           ACL_FLOAT16, s_shape, max_abs_ptr,
                           ACL_FLOAT16, s_shape, scale_x_ptr,
                           muls_ws_ptr, ws_layout_.muls_ws_bytes);
    if (st != atb::NO_ERROR) return st;
  }

  // ---- Step 4: buf_b = x / scale_x  [ATB REALDIV, broadcast] ----
  {
    auto st = run_atb_binary(div_op_,
                            ACL_FLOAT16, x_desc.shape, vp.inTensors.at(X_INPUT_INDEX).deviceData,
                            ACL_FLOAT16, s_shape,      scale_x_ptr,
                            ACL_FLOAT16, x_desc.shape, buf_b_ptr,
                            div_ws_ptr, ws_layout_.div_ws_bytes);
    if (st != atb::NO_ERROR) return st;
  }

  // ---- Step 5: buf_a = round(buf_b)  [cached executor] ----
  {
    auto* c = static_cast<AclUnaryCache*>(round_cache_);
    if (!c || !c->exec) return atb::ERROR_INTERNAL_ERROR;
    auto st = updateUnaryAddrs(c, buf_b_ptr, buf_a_ptr);
    if (st != atb::NO_ERROR) return st;
    MLLM_ACL_CHECK(aclnnRound(ws_layout_.round_ws_bytes > 0 ? round_ws_ptr : nullptr,
                                c->ws_size, c->exec, stream));
  }

  // ---- Step 6: buf_b = clamp(buf_a, -128, 127)  [cached executor] ----
  {
    auto* c = static_cast<AclUnaryCache*>(clamp_cache_);
    if (!c || !c->exec) return atb::ERROR_INTERNAL_ERROR;
    auto st = updateUnaryAddrs(c, buf_a_ptr, buf_b_ptr);
    if (st != atb::NO_ERROR) return st;
    MLLM_ACL_CHECK(aclnnClamp(ws_layout_.clamp_ws_bytes > 0 ? clamp_ws_ptr : nullptr,
                                c->ws_size, c->exec, stream));
  }

  // ---- Step 7: x_int8 = cast(buf_b, INT8)  [cached executor] ----
  {
    auto* c = static_cast<AclUnaryCache*>(cast_cache_);
    if (!c || !c->exec) return atb::ERROR_INTERNAL_ERROR;
    auto st = updateUnaryAddrs(c, buf_b_ptr, x_int8_ptr);
    if (st != atb::NO_ERROR) return st;
    MLLM_ACL_CHECK(aclnnCast(ws_layout_.cast_ws_bytes > 0 ? cast_ws_ptr : nullptr,
                               c->ws_size, c->exec, stream));
  }

  // ---- Step 8: y_pre = ATB Linear W8A8(x_int8, weight, bias, deq_scale_w) ----
  {
    atb::Tensor xi = makeTensorDesc(ACL_INT8,    x_desc.shape, x_int8_ptr);
    atb::Tensor yo = makeTensorDesc(ACL_FLOAT16, y_shape,      y_pre_ptr);
    atb::VariantPack lvp;
    lvp.inTensors  = {xi,
                      vp.inTensors.at(WEIGHT_INPUT_INDEX),
                      vp.inTensors.at(BIAS_INPUT_INDEX),
                      vp.inTensors.at(DEQ_SCALE_INPUT_INDEX)};
    lvp.outTensors = {yo};
    uint64_t q = 0;
    auto st = linear_op_->Setup(lvp, q, context);
    if (st != atb::NO_ERROR) return st;
    st = linear_op_->Execute(lvp, linear_ws_ptr, ws_layout_.linear_ws_bytes, context);
    if (st != atb::NO_ERROR) return st;
  }

  // ---- Step 9: y = y_pre * scale_x  [ATB MUL, broadcast] ----
  {
    auto st = run_atb_binary(mul_dequant_op_,
                            ACL_FLOAT16, y_shape, y_pre_ptr,
                            ACL_FLOAT16, s_shape, scale_x_ptr,
                            ACL_FLOAT16, y_shape, vp.outTensors.at(Y_OUTPUT_INDEX).deviceData,
                            mul_dequant_ws_ptr, ws_layout_.mul_dequant_ws_bytes);
    if (st != atb::NO_ERROR) return st;
  }

  return atb::NO_ERROR;
}

atb::Operation* createDynamicLinearW8A8PluginGraphOp(std::string name_suffix) {
  return new AscendDynamicLinearW8A8PluginOperation(std::move(name_suffix));
}

}  // namespace mllm::ascend
