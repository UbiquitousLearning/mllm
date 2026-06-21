// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "KernelTestHelper.hpp"

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_max_dim.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

class AscendAclnnKernelTest : public KernelTest {
 public:
  AscendAclnnKernelTest() = default;
  ~AscendAclnnKernelTest() override = default;

  // Test aclnnCast FP16 -> INT8 (alternative to ELEWISE_CAST which fails on 310B)
  bool AclnnCastFloat16ToInt8Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;          // NOLINT
    using namespace mllm::ascend;  // NOLINT

    for (auto& shape : shapes) {
      // 1. Random FP16 input in [-10, 10], compute expected INT8 via truncation
      Tensor x_cpu = Tensor::random(shape, -10, 10, kFloat16, kCPU);
      Tensor ref_cpu = Tensor::empty(shape, kInt8, kCPU).alloc();
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<int8_t>();
        for (size_t i = 0; i < x_cpu.numel(); ++i) {
          float v = MLLM_FP16_TO_FP32(x_ptr[i]);
          // clamp to [-128, 127] then truncate (same as hardware cast behavior)
          v = std::max(-128.0f, std::min(127.0f, v));
          r_ptr[i] = static_cast<int8_t>(v);
        }
      }

      // 2. Move input to NPU, allocate INT8 output
      auto x_npu = x_cpu.to(kAscend);
      Tensor y_npu = Tensor::empty(shape, kInt8, kAscend).alloc();

      // 3. Build aclTensor descriptors
      auto makeAclTensor = [](const Tensor& t, aclDataType acl_dtype) -> aclTensor* {
        const auto& sh = t.shape();
        int ndim = static_cast<int>(sh.size());
        std::vector<int64_t> dims(ndim), strides(ndim);
        for (int i = 0; i < ndim; ++i) dims[i] = static_cast<int64_t>(sh[i]);
        int64_t stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
          strides[i] = stride;
          stride *= dims[i];
        }
        return aclCreateTensor(dims.data(), ndim, acl_dtype, strides.data(), 0,
                               ACL_FORMAT_ND, dims.data(), ndim, t.ptr<void>());
      };

      aclTensor* acl_x = makeAclTensor(x_npu, ACL_FLOAT16);
      aclTensor* acl_y = makeAclTensor(y_npu, ACL_INT8);
      if (acl_x == nullptr || acl_y == nullptr) {
        std::cerr << "[AclnnCastTest] aclCreateTensor failed\n";
        if (acl_x) aclDestroyTensor(acl_x);
        if (acl_y) aclDestroyTensor(acl_y);
        return false;
      }

      // 4. aclnnCast: first stage, get workspace size
      uint64_t ws_size = 0;
      aclOpExecutor* executor = nullptr;
      auto ret = aclnnCastGetWorkspaceSize(acl_x, ACL_INT8, acl_y, &ws_size, &executor);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[AclnnCastTest] aclnnCastGetWorkspaceSize FAILED, ret=" << ret << "\n";
        aclDestroyTensor(acl_x);
        aclDestroyTensor(acl_y);
        return false;
      }

      // 5. Allocate workspace on NPU
      void* workspace = nullptr;
      int ws_block_id = -1;
      if (ws_size > 0) {
        auto& mem_mgr = getAscendMemoryManager();
        mem_mgr.allocateBlock(static_cast<uint32_t>(ws_size), ws_block_id);
        mem_mgr.getBlockPtr(ws_block_id, workspace);
      }

      // 6. aclnnCast: second stage, execute
      aclrtStream stream = getGlobalAtbStream();
      ret = aclnnCast(workspace, ws_size, executor, stream);
      syncGlobalAtbStream();

      if (ws_block_id != -1) getAscendMemoryManager().freeBlock(ws_block_id);
      aclDestroyTensor(acl_x);
      aclDestroyTensor(acl_y);

      if (ret != ACL_SUCCESS) {
        std::cerr << "[AclnnCastTest] aclnnCast FAILED, ret=" << ret << "\n";
        return false;
      }

      // 7. Compare result
      auto y_cpu = y_npu.to(kCPU);
      auto* y_ptr = y_cpu.ptr<int8_t>();
      auto* r_ptr = ref_cpu.ptr<int8_t>();
      auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
      for (size_t i = 0; i < y_cpu.numel(); ++i) {
        if (y_ptr[i] != r_ptr[i]) {
          std::cerr << "[AclnnCastTest] mismatch at i=" << i
                    << " input=" << MLLM_FP16_TO_FP32(x_ptr[i])
                    << " expect=" << static_cast<int>(r_ptr[i])
                    << " got=" << static_cast<int>(y_ptr[i]) << "\n";
          return false;
        }
      }
      std::cout << "[AclnnCastTest] shape test PASSED\n";
    }
    return true;
  }

  // Test aclnnAbs (FP16): element-wise absolute value
  bool AclnnAbsFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;          // NOLINT
    using namespace mllm::ascend;  // NOLINT

    auto makeAclTensor = [](const Tensor& t, aclDataType acl_dtype) -> aclTensor* {
      const auto& sh = t.shape();
      int ndim = static_cast<int>(sh.size());
      std::vector<int64_t> dims(ndim), strides(ndim);
      for (int i = 0; i < ndim; ++i) dims[i] = static_cast<int64_t>(sh[i]);
      int64_t stride = 1;
      for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
      }
      return aclCreateTensor(dims.data(), ndim, acl_dtype, strides.data(), 0,
                             ACL_FORMAT_ND, dims.data(), ndim, t.ptr<void>());
    };

    for (auto& shape : shapes) {
      Tensor x_cpu = Tensor::random(shape, -10, 10, kFloat16, kCPU);
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          float v = MLLM_FP16_TO_FP32(x_ptr[i]);
          r_ptr[i] = static_cast<mllm_fp16_t>(std::abs(v));
        }
      }

      auto x_npu = x_cpu.to(kAscend);
      Tensor y_npu = Tensor::empty(shape, kFloat16, kAscend).alloc();

      aclTensor* acl_x = makeAclTensor(x_npu, ACL_FLOAT16);
      aclTensor* acl_y = makeAclTensor(y_npu, ACL_FLOAT16);
      if (acl_x == nullptr || acl_y == nullptr) {
        std::cerr << "[AclnnAbsTest] aclCreateTensor failed\n";
        if (acl_x) aclDestroyTensor(acl_x);
        if (acl_y) aclDestroyTensor(acl_y);
        return false;
      }

      uint64_t ws_size = 0;
      aclOpExecutor* executor = nullptr;
      auto ret = aclnnAbsGetWorkspaceSize(acl_x, acl_y, &ws_size, &executor);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[AclnnAbsTest] GetWorkspaceSize FAILED, ret=" << ret << "\n";
        aclDestroyTensor(acl_x);
        aclDestroyTensor(acl_y);
        return false;
      }

      void* workspace = nullptr;
      int ws_block_id = -1;
      if (ws_size > 0) {
        auto& mem_mgr = getAscendMemoryManager();
        mem_mgr.allocateBlock(static_cast<uint32_t>(ws_size), ws_block_id);
        mem_mgr.getBlockPtr(ws_block_id, workspace);
      }

      aclrtStream stream = getGlobalAtbStream();
      ret = aclnnAbs(workspace, ws_size, executor, stream);
      syncGlobalAtbStream();

      if (ws_block_id != -1) getAscendMemoryManager().freeBlock(ws_block_id);
      aclDestroyTensor(acl_x);
      aclDestroyTensor(acl_y);

      if (ret != ACL_SUCCESS) {
        std::cerr << "[AclnnAbsTest] Execute FAILED, ret=" << ret << "\n";
        return false;
      }

      auto y_cpu = y_npu.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        std::cerr << "[AclnnAbsTest] mismatch\n";
        return false;
      }
      std::cout << "[AclnnAbsTest] shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
      }
      std::cout << "] PASSED\n";
    }
    return true;
  }

  // Test aclnnMaxDim (FP16, keepdim=true): max along last dim with dummy indices output
  bool AclnnMaxDimFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;          // NOLINT
    using namespace mllm::ascend;  // NOLINT

    auto makeAclTensor = [](const std::vector<int64_t>& dims, aclDataType dtype, void* dev_ptr) -> aclTensor* {
      int ndim = static_cast<int>(dims.size());
      std::vector<int64_t> strides(ndim);
      int64_t stride = 1;
      for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
      }
      return aclCreateTensor(dims.data(), ndim, dtype, strides.data(), 0,
                             ACL_FORMAT_ND, dims.data(), ndim, dev_ptr);
    };

    for (auto& shape : shapes) {
      size_t rows = 1;
      for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
      size_t cols = shape.back();

      Tensor x_cpu = Tensor::random(shape, -10, 10, kFloat16, kCPU);
      Tensor ref_cpu = Tensor::empty({static_cast<Tensor::shape_t::value_type>(rows)}, kFloat16, kCPU).alloc();
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        for (size_t r = 0; r < rows; ++r) {
          float max_val = -1e30f;
          for (size_t c = 0; c < cols; ++c) {
            float v = MLLM_FP16_TO_FP32(x_ptr[r * cols + c]);
            max_val = std::max(max_val, v);
          }
          r_ptr[r] = static_cast<mllm_fp16_t>(max_val);
        }
      }

      auto x_npu = x_cpu.to(kAscend);

      std::vector<int64_t> out_dims;
      for (size_t i = 0; i + 1 < shape.size(); ++i) out_dims.push_back(static_cast<int64_t>(shape[i]));
      out_dims.push_back(1);
      std::vector<int64_t> idx_dims(out_dims.begin(), out_dims.end());

      Tensor y_npu = Tensor::empty(shape.size() == 1
                                       ? std::vector<Tensor::shape_t::value_type>{1}
                                       : std::vector<Tensor::shape_t::value_type>(out_dims.begin(), out_dims.end()),
                                   kFloat16, kAscend)
                         .alloc();
      Tensor idx_npu = Tensor::empty(shape.size() == 1
                                         ? std::vector<Tensor::shape_t::value_type>{1}
                                         : std::vector<Tensor::shape_t::value_type>(idx_dims.begin(), idx_dims.end()),
                                     kInt32, kAscend)
                           .alloc();

      std::vector<int64_t> x_dims_int64(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) x_dims_int64[i] = static_cast<int64_t>(shape[i]);

      std::vector<int64_t> y_dims_int64(shape.size());
      for (size_t i = 0; i + 1 < shape.size(); ++i) y_dims_int64[i] = static_cast<int64_t>(shape[i]);
      y_dims_int64[shape.size() - 1] = 1;

      aclTensor* acl_x = makeAclTensor(x_dims_int64, ACL_FLOAT16, x_npu.ptr<void>());
      aclTensor* acl_y = makeAclTensor(y_dims_int64, ACL_FLOAT16, y_npu.ptr<void>());
      aclTensor* acl_idx = makeAclTensor(y_dims_int64, ACL_INT32, idx_npu.ptr<void>());

      if (!acl_x || !acl_y || !acl_idx) {
        std::cerr << "[AclnnMaxDimTest] aclCreateTensor failed\n";
        if (acl_x) aclDestroyTensor(acl_x);
        if (acl_y) aclDestroyTensor(acl_y);
        if (acl_idx) aclDestroyTensor(acl_idx);
        return false;
      }

      uint64_t ws_size = 0;
      aclOpExecutor* executor = nullptr;
      int64_t dim = static_cast<int64_t>(shape.size()) - 1;
      auto ret = aclnnMaxDimGetWorkspaceSize(acl_x, dim, true, acl_y, acl_idx, &ws_size, &executor);
      if (ret != ACL_SUCCESS) {
        std::cerr << "[AclnnMaxDimTest] GetWorkspaceSize FAILED, ret=" << ret << "\n";
        aclDestroyTensor(acl_x);
        aclDestroyTensor(acl_y);
        aclDestroyTensor(acl_idx);
        return false;
      }

      void* workspace = nullptr;
      int ws_block_id = -1;
      if (ws_size > 0) {
        auto& mem_mgr = getAscendMemoryManager();
        mem_mgr.allocateBlock(static_cast<uint32_t>(ws_size), ws_block_id);
        mem_mgr.getBlockPtr(ws_block_id, workspace);
      }

      aclrtStream stream = getGlobalAtbStream();
      ret = aclnnMaxDim(workspace, ws_size, executor, stream);
      syncGlobalAtbStream();

      if (ws_block_id != -1) getAscendMemoryManager().freeBlock(ws_block_id);
      aclDestroyTensor(acl_x);
      aclDestroyTensor(acl_y);
      aclDestroyTensor(acl_idx);

      if (ret != ACL_SUCCESS) {
        std::cerr << "[AclnnMaxDimTest] Execute FAILED, ret=" << ret << "\n";
        return false;
      }

      auto y_cpu = y_npu.to(kCPU);
      auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
      auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
      float max_err = 0.0f;
      for (size_t i = 0; i < rows; ++i) {
        float got = MLLM_FP16_TO_FP32(y_ptr[i]);
        float ref = MLLM_FP16_TO_FP32(r_ptr[i]);
        max_err = std::max(max_err, std::abs(got - ref));
      }
      std::cout << "[AclnnMaxDimTest] shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
      }
      std::cout << "] max_err=" << max_err;
      if (max_err > 0.1f) {
        std::cerr << " FAIL\n";
        return false;
      }
      std::cout << " PASSED\n";
    }
    return true;
  }
};
