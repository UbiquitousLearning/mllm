// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/ops/AscendLinearDynamicW8A8.hpp"

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_max_dim.h>
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cast.h>
#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ascend {
namespace MLLM_ANONYMOUS_NAMESPACE {

bool shouldEnableDynamicW8A8Eager() {
  const char* enabled = std::getenv("MLLM_ASCEND_ENABLE_DYNAMIC_W8A8");
  return enabled != nullptr && enabled[0] == '1';
}

std::vector<int64_t> toDims(const Tensor::shape_t& shape) {
  std::vector<int64_t> dims(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) dims[i] = static_cast<int64_t>(shape[i]);
  return dims;
}

std::vector<int64_t> makeStrides(const std::vector<int64_t>& dims) {
  int nd = static_cast<int>(dims.size());
  std::vector<int64_t> strides(nd, 1);
  for (int i = nd - 2; i >= 0; --i) strides[i] = strides[i + 1] * dims[i + 1];
  return strides;
}

aclTensor* makeAclTensor(void* ptr, aclDataType dtype, const std::vector<int64_t>& dims) {
  auto strides = makeStrides(dims);
  int nd = static_cast<int>(dims.size());
  return aclCreateTensor(dims.data(), nd, dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(), nd, ptr);
}

template <typename GetWorkspaceFn, typename ExecuteFn>
void runUnaryAclnn(GetWorkspaceFn get_workspace, ExecuteFn execute, aclTensor* src, aclTensor* dst) {
  auto stream = getGlobalAtbStream();
  auto& mem_mgr = getAscendMemoryManager();

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  MLLM_ACL_CHECK(get_workspace(src, dst, &workspace_size, &executor));
  void* workspace = nullptr;
  int block_id = -1;
  if (workspace_size > 0) {
    mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
    mem_mgr.getBlockPtr(block_id, workspace);
  }
  MLLM_ACL_CHECK(execute(workspace, workspace_size, executor, stream));
  if (block_id != -1) mem_mgr.freeBlock(block_id);
}

void runAtbElewise(atb::infer::ElewiseParam::ElewiseType type,
                          const Tensor& a,
                          const Tensor& b,
                          Tensor& out) {
  atb::infer::ElewiseParam param;
  param.elewiseType = type;
  atb::Operation* op = nullptr;
  MLLM_ATB_CHECK(atb::CreateOperation(param, &op));

  atb::Tensor atb_a, atb_b, atb_out;
  fillAtbTensor(a, atb_a);
  fillAtbTensor(b, atb_b);
  fillAtbTensor(out, atb_out);

  atb::VariantPack variant_pack;
  variant_pack.inTensors = {atb_a, atb_b};
  variant_pack.outTensors = {atb_out};

  uint64_t workspace_size = 0;
  MLLM_ATB_CHECK(op->Setup(variant_pack, workspace_size, getGlobalAtbContext()));
  void* workspace = nullptr;
  int block_id = -1;
  if (workspace_size > 0) {
    getAscendMemoryManager().allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
    getAscendMemoryManager().getBlockPtr(block_id, workspace);
  }
  MLLM_ATB_CHECK(op->Execute(variant_pack,
                             reinterpret_cast<uint8_t*>(workspace),
                             workspace_size,
                             getGlobalAtbContext()));
  if (block_id != -1) getAscendMemoryManager().freeBlock(block_id);
  atb::DestroyOperation(op);
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

void runLinearDynamicW8A8Eager(const std::string& layer_name,
                               const Tensor& x,
                               const Tensor& weight,
                               const Tensor& bias_int32_npu,
                               const Tensor& deq_scale_w_npu,
                               Tensor& y) {
  // Experimental dynamic W8A8 eager path.
  //
  // Static calibrated W8A8 is the production path and is executed by the Qwen
  // Ascend decoder graph/plugin. This eager path exists only for accuracy/debug
  // probes and is disabled by default. Set MLLM_ASCEND_ENABLE_DYNAMIC_W8A8=1 to
  // run it intentionally.
  if (!shouldEnableDynamicW8A8Eager()) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AscendLinearOp: INT8 eager W8A8 is disabled for layer {}. "
                    "Static W8A8 should run through the Qwen Ascend decoder graph. "
                    "Set MLLM_ASCEND_ENABLE_DYNAMIC_W8A8=1 only for eager accuracy/debug tests.",
                    layer_name);
  }
  MLLM_RT_ASSERT_EQ(x.dtype(), kFloat16);
  MLLM_RT_ASSERT(!deq_scale_w_npu.isNil());

  auto* atb_ctx = getGlobalAtbContext();
  auto stream = getGlobalAtbStream();
  auto& mem_mgr = getAscendMemoryManager();

  const auto& xshape = x.shape();
  const int ndim = static_cast<int>(xshape.size());
  auto xdims = toDims(xshape);

  // scale_x[..., 1] = max(abs(x), dim=-1, keepdim=true) / 127
  Tensor abs_x = Tensor::empty(xshape, kFloat16, kAscend).alloc();
  {
    auto acl_x = makeAclTensor(x.ptr<void>(), ACL_FLOAT16, xdims);
    auto acl_abs = makeAclTensor(abs_x.ptr<void>(), ACL_FLOAT16, xdims);
    runUnaryAclnn(aclnnAbsGetWorkspaceSize, aclnnAbs, acl_x, acl_abs);
    aclDestroyTensor(acl_x);
    aclDestroyTensor(acl_abs);
  }

  Tensor::shape_t scale_shape(xshape.begin(), xshape.end());
  scale_shape.back() = 1;
  Tensor max_abs = Tensor::empty(scale_shape, kFloat16, kAscend).alloc();
  Tensor idx_buf = Tensor::empty(scale_shape, kInt32, kAscend).alloc();
  {
    auto sdims = toDims(scale_shape);
    auto acl_in = makeAclTensor(abs_x.ptr<void>(), ACL_FLOAT16, xdims);
    auto acl_out = makeAclTensor(max_abs.ptr<void>(), ACL_FLOAT16, sdims);
    auto acl_idx = makeAclTensor(idx_buf.ptr<void>(), ACL_INT32, sdims);
    int64_t dim = static_cast<int64_t>(ndim) - 1;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    MLLM_ACL_CHECK(aclnnMaxDimGetWorkspaceSize(acl_in, dim, true, acl_out, acl_idx, &workspace_size, &executor));
    void* workspace = nullptr;
    int block_id = -1;
    if (workspace_size > 0) {
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
      mem_mgr.getBlockPtr(block_id, workspace);
    }
    MLLM_ACL_CHECK(aclnnMaxDim(workspace, workspace_size, executor, stream));
    if (block_id != -1) mem_mgr.freeBlock(block_id);
    aclDestroyTensor(acl_in);
    aclDestroyTensor(acl_out);
    aclDestroyTensor(acl_idx);
  }

  Tensor scale_x_npu = Tensor::empty(scale_shape, kFloat16, kAscend).alloc();
  {
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ELEWISE_MULS;
    param.mulsParam.varAttr = 1.0f / 127.0f;
    atb::Operation* op = nullptr;
    MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
    atb::Tensor atb_max_abs, atb_scale;
    fillAtbTensor(max_abs, atb_max_abs);
    fillAtbTensor(scale_x_npu, atb_scale);
    atb::VariantPack variant_pack;
    variant_pack.inTensors = {atb_max_abs};
    variant_pack.outTensors = {atb_scale};
    uint64_t workspace_size = 0;
    MLLM_ATB_CHECK(op->Setup(variant_pack, workspace_size, atb_ctx));
    void* workspace = nullptr;
    int block_id = -1;
    if (workspace_size > 0) {
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
      mem_mgr.getBlockPtr(block_id, workspace);
    }
    MLLM_ATB_CHECK(op->Execute(variant_pack, reinterpret_cast<uint8_t*>(workspace), workspace_size, atb_ctx));
    if (block_id != -1) mem_mgr.freeBlock(block_id);
    atb::DestroyOperation(op);
  }

  // x_int8 = cast(clamp(round(x / scale_x), -128, 127), int8)
  Tensor x_scaled = Tensor::empty(xshape, kFloat16, kAscend).alloc();
  runAtbElewise(atb::infer::ElewiseParam::ELEWISE_REALDIV, x, scale_x_npu, x_scaled);

  Tensor x_round = Tensor::empty(xshape, kFloat16, kAscend).alloc();
  {
    auto acl_in = makeAclTensor(x_scaled.ptr<void>(), ACL_FLOAT16, xdims);
    auto acl_out = makeAclTensor(x_round.ptr<void>(), ACL_FLOAT16, xdims);
    runUnaryAclnn(aclnnRoundGetWorkspaceSize, aclnnRound, acl_in, acl_out);
    aclDestroyTensor(acl_in);
    aclDestroyTensor(acl_out);
  }

  Tensor x_clamped = Tensor::empty(xshape, kFloat16, kAscend).alloc();
  {
    auto acl_in = makeAclTensor(x_round.ptr<void>(), ACL_FLOAT16, xdims);
    auto acl_out = makeAclTensor(x_clamped.ptr<void>(), ACL_FLOAT16, xdims);
    float min_val = -128.0f;
    float max_val = 127.0f;
    auto* acl_min = aclCreateScalar(&min_val, ACL_FLOAT);
    auto* acl_max = aclCreateScalar(&max_val, ACL_FLOAT);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    MLLM_ACL_CHECK(aclnnClampGetWorkspaceSize(acl_in, acl_min, acl_max, acl_out, &workspace_size, &executor));
    void* workspace = nullptr;
    int block_id = -1;
    if (workspace_size > 0) {
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
      mem_mgr.getBlockPtr(block_id, workspace);
    }
    MLLM_ACL_CHECK(aclnnClamp(workspace, workspace_size, executor, stream));
    if (block_id != -1) mem_mgr.freeBlock(block_id);
    aclDestroyScalar(acl_min);
    aclDestroyScalar(acl_max);
    aclDestroyTensor(acl_in);
    aclDestroyTensor(acl_out);
  }

  Tensor x_int8 = Tensor::empty(xshape, kInt8, kAscend).alloc();
  {
    auto acl_in = makeAclTensor(x_clamped.ptr<void>(), ACL_FLOAT16, xdims);
    auto acl_out = makeAclTensor(x_int8.ptr<void>(), ACL_INT8, xdims);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    MLLM_ACL_CHECK(aclnnCastGetWorkspaceSize(acl_in, ACL_INT8, acl_out, &workspace_size, &executor));
    void* workspace = nullptr;
    int block_id = -1;
    if (workspace_size > 0) {
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
      mem_mgr.getBlockPtr(block_id, workspace);
    }
    MLLM_ACL_CHECK(aclnnCast(workspace, workspace_size, executor, stream));
    if (block_id != -1) mem_mgr.freeBlock(block_id);
    aclDestroyTensor(acl_in);
    aclDestroyTensor(acl_out);
  }

  // y_pre = Linear W8A8 PER_CHANNEL(x_int8, weight, bias_i32, deq_scale_w)
  Tensor::shape_t y_shape(xshape.begin(), xshape.end());
  y_shape.back() = weight.shape()[0];
  Tensor y_pre = Tensor::empty(y_shape, kFloat16, kAscend).alloc();
  {
    atb::infer::LinearParam param;
    param.transposeA = false;
    param.transposeB = true;
    param.hasBias = true;
    param.outDataType = ACL_FLOAT16;
    param.enAccum = false;
    param.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
    param.quantMode = atb::infer::LinearParam::PER_CHANNEL;
    atb::Operation* op = nullptr;
    MLLM_ATB_CHECK(atb::CreateOperation(param, &op));
    atb::Tensor atb_x_int8, atb_weight, atb_bias, atb_deq_scale, atb_y_pre;
    fillAtbTensor(x_int8, atb_x_int8);
    fillAtbTensor(weight, atb_weight);
    fillAtbTensor(bias_int32_npu, atb_bias);
    fillAtbTensor(deq_scale_w_npu, atb_deq_scale);
    fillAtbTensor(y_pre, atb_y_pre);
    atb::VariantPack variant_pack;
    variant_pack.inTensors = {atb_x_int8, atb_weight, atb_bias, atb_deq_scale};
    variant_pack.outTensors = {atb_y_pre};
    uint64_t workspace_size = 0;
    MLLM_ATB_CHECK(op->Setup(variant_pack, workspace_size, atb_ctx));
    void* workspace = nullptr;
    int block_id = -1;
    if (workspace_size > 0) {
      mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), block_id);
      mem_mgr.getBlockPtr(block_id, workspace);
    }
    MLLM_ATB_CHECK(op->Execute(variant_pack, reinterpret_cast<uint8_t*>(workspace), workspace_size, atb_ctx));
    if (block_id != -1) mem_mgr.freeBlock(block_id);
    atb::DestroyOperation(op);
  }

  // y = y_pre * scale_x, broadcast from [..., 1] to [..., N].
  runAtbElewise(atb::infer::ElewiseParam::ELEWISE_MUL, y_pre, scale_x_npu, y);
  syncGlobalAtbStream();
}

}  // namespace mllm::ascend
