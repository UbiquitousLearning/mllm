// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/ascend/ops/AscendElewiseOps.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "KernelTestHelper.hpp"

#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>
#include <iostream>
#include <vector>

class AscendKernelTest : public KernelTest {
 public:
  AscendKernelTest() = default;
  ~AscendKernelTest() override = default;

  // Test Add operation with different shapes
  bool AddFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor y_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          r_ptr[i] = x_ptr[i] + y_ptr[i];
        }
      }

      // 3. Move inputs to Ascend and run Add (z = x + y)
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = y_cpu.to(kAscend);
      auto z_ascend = x_ascend + y_ascend;

      // 4. Move result back to CPU and compare with reference using allClose
      auto z_cpu = z_ascend.to(kCPU);
      auto result = mllm::test::allClose(z_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }

  // Test Sub operation with different shapes
  bool SubFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor y_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          r_ptr[i] = x_ptr[i] - y_ptr[i];
        }
      }

      // 3. Move inputs to Ascend and run Sub (z = x - y)
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = y_cpu.to(kAscend);
      auto z_ascend = x_ascend - y_ascend;

      // 4. Move result back to CPU and compare with reference using allClose
      auto z_cpu = z_ascend.to(kCPU);
      auto result = mllm::test::allClose(z_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }

  // Test Mul operation with different shapes
  bool MulFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;  // NOLINT
    for (auto& shape : shapes) {
      // 1. Construct random FP16 inputs on CPU
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor y_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);

      // 2. Compute reference result (FP16) on CPU
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* y_ptr = y_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          r_ptr[i] = x_ptr[i] * y_ptr[i];
        }
      }

      // 3. Move inputs to Ascend and run Mul (z = x * y)
      auto x_ascend = x_cpu.to(kAscend);
      auto y_ascend = y_cpu.to(kAscend);
      auto z_ascend = x_ascend * y_ascend;

      // 4. Move result back to CPU and compare with reference using allClose
      auto z_cpu = z_ascend.to(kCPU);
      auto result = mllm::test::allClose(z_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }

  bool MulScalarFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes, float scalar = 0.5f) {
    using namespace mllm;  // NOLINT
    using namespace mllm::ascend;  // NOLINT

    for (auto& shape : shapes) {
      Tensor x_cpu = Tensor::random(shape, -3, 3, kFloat16, kCPU);
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);

      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        auto num_elements = x_cpu.numel();
        for (size_t i = 0; i < num_elements; ++i) {
          float value = static_cast<float>(x_ptr[i]) * scalar;
          r_ptr[i] = static_cast<mllm_fp16_t>(value);
        }
      }

      auto x_ascend = x_cpu.to(kAscend);
      Tensor y_ascend = Tensor::emptyLike(x_ascend).alloc();

      atb::infer::ElewiseParam param;
      param.elewiseType = atb::infer::ElewiseParam::ELEWISE_MULS;
      param.mulsParam.varAttr = scalar;

      atb::Operation* op = nullptr;
      auto st = atb::CreateOperation(param, &op);
      if (st != atb::NO_ERROR || op == nullptr) {
        std::cerr << "[MulScalarFloat16Test] CreateOperation failed, status=" << static_cast<int>(st) << std::endl;
        return false;
      }

      atb::Tensor atb_x;
      atb::Tensor atb_y;
      fillAtbTensor(x_ascend, atb_x);
      fillAtbTensor(y_ascend, atb_y);

      atb::VariantPack vp;
      vp.inTensors = {atb_x};
      vp.outTensors = {atb_y};

      auto* ctx = getGlobalAtbContext();
      uint64_t workspace_size = 0;
      st = op->Setup(vp, workspace_size, ctx);
      if (st != atb::NO_ERROR) {
        std::cerr << "[MulScalarFloat16Test] Setup failed, status=" << static_cast<int>(st) << std::endl;
        atb::DestroyOperation(op);
        return false;
      }

      void* workspace = nullptr;
      int workspace_block_id = -1;
      if (workspace_size > 0) {
        auto& mem_mgr = getAscendMemoryManager();
        mem_mgr.allocateBlock(static_cast<uint32_t>(workspace_size), workspace_block_id);
        mem_mgr.getBlockPtr(workspace_block_id, workspace);
      }

      st = op->Execute(vp, reinterpret_cast<uint8_t*>(workspace), workspace_size, ctx);
      syncGlobalAtbStream();

      if (workspace_block_id != -1) {
        auto& mem_mgr = getAscendMemoryManager();
        mem_mgr.freeBlock(workspace_block_id);
      }
      atb::DestroyOperation(op);

      if (st != atb::NO_ERROR) {
        return false;
      }

      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        return false;
      }
    }
    return true;
  }

};
