// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/mllm.hpp"
#include "KernelTestHelper.hpp"

#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>

#include <cstdint>
#include <iostream>
#include <vector>

class AscendBroadcastKernelTest : public KernelTest {
 public:
  AscendBroadcastKernelTest() = default;
  ~AscendBroadcastKernelTest() override = default;

  // Test ATB ELEWISE_REALDIV (broadcast): [M,K] / [M,1] element-wise division
  bool RealDivBroadcastFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;          // NOLINT
    using namespace mllm::ascend;  // NOLINT

    for (auto& shape : shapes) {
      size_t rows = 1;
      for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
      size_t cols = shape.back();

      Tensor x_cpu = Tensor::random(shape, -10, 10, kFloat16, kCPU);
      std::vector<Tensor::shape_t::value_type> scale_vec_shape;
      for (size_t i = 0; i + 1 < shape.size(); ++i) scale_vec_shape.push_back(shape[i]);
      scale_vec_shape.push_back(1);
      Tensor scale_cpu = Tensor::random(scale_vec_shape, 1.0f, 10.0f, kFloat16, kCPU);
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* s_ptr = scale_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        for (size_t r = 0; r < rows; ++r) {
          float scale = MLLM_FP16_TO_FP32(s_ptr[r]);
          for (size_t c = 0; c < cols; ++c) {
            r_ptr[r * cols + c] = static_cast<mllm_fp16_t>(MLLM_FP16_TO_FP32(x_ptr[r * cols + c]) / scale);
          }
        }
      }

      auto x_npu = x_cpu.to(kAscend);
      auto scale_npu = scale_cpu.to(kAscend);

      Tensor y_npu = Tensor::empty(shape, kFloat16, kAscend).alloc();

      atb::infer::ElewiseParam param;
      param.elewiseType = atb::infer::ElewiseParam::ELEWISE_REALDIV;

      atb::Operation* op = nullptr;
      auto st = atb::CreateOperation(param, &op);
      if (st != atb::NO_ERROR || op == nullptr) {
        std::cerr << "[RealDivBroadcastTest] CreateOperation failed, status=" << static_cast<int>(st) << "\n";
        return false;
      }

      atb::Tensor atb_x, atb_scale, atb_y;
      fillAtbTensor(x_npu, atb_x);
      fillAtbTensor(scale_npu, atb_scale);
      fillAtbTensor(y_npu, atb_y);

      atb::VariantPack vp;
      vp.inTensors = {atb_x, atb_scale};
      vp.outTensors = {atb_y};

      auto* ctx = getGlobalAtbContext();
      uint64_t workspace_size = 0;
      st = op->Setup(vp, workspace_size, ctx);
      if (st != atb::NO_ERROR) {
        std::cerr << "[RealDivBroadcastTest] Setup failed, status=" << static_cast<int>(st) << "\n";
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

      if (workspace_block_id != -1) getAscendMemoryManager().freeBlock(workspace_block_id);
      atb::DestroyOperation(op);

      if (st != atb::NO_ERROR) {
        std::cerr << "[RealDivBroadcastTest] Execute failed, status=" << static_cast<int>(st) << "\n";
        return false;
      }

      auto y_cpu = y_npu.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-1f, 1e-1f);
      if (!result.is_close) {
        std::cerr << "[RealDivBroadcastTest] mismatch\n";
        return false;
      }
      std::cout << "[RealDivBroadcastTest] shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
      }
      std::cout << "] PASSED\n";
    }
    return true;
  }

  // Test ATB ELEWISE_MUL (broadcast): [M,K] * [M,1] element-wise multiplication
  bool MulBroadcastFloat16Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
    using namespace mllm;          // NOLINT
    using namespace mllm::ascend;  // NOLINT

    for (auto& shape : shapes) {
      size_t rows = 1;
      for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
      size_t cols = shape.back();

      Tensor x_cpu = Tensor::random(shape, -10, 10, kFloat16, kCPU);
      std::vector<Tensor::shape_t::value_type> scale_vec_shape;
      for (size_t i = 0; i + 1 < shape.size(); ++i) scale_vec_shape.push_back(shape[i]);
      scale_vec_shape.push_back(1);
      Tensor scale_cpu = Tensor::random(scale_vec_shape, -2.0f, 2.0f, kFloat16, kCPU);
      Tensor ref_cpu = Tensor::zeros(shape, kFloat16, kCPU);
      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* s_ptr = scale_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();
        for (size_t r = 0; r < rows; ++r) {
          float scale = MLLM_FP16_TO_FP32(s_ptr[r]);
          for (size_t c = 0; c < cols; ++c) {
            r_ptr[r * cols + c] = static_cast<mllm_fp16_t>(MLLM_FP16_TO_FP32(x_ptr[r * cols + c]) * scale);
          }
        }
      }

      auto x_npu = x_cpu.to(kAscend);
      auto scale_npu = scale_cpu.to(kAscend);

      Tensor y_npu = Tensor::empty(shape, kFloat16, kAscend).alloc();

      atb::infer::ElewiseParam param;
      param.elewiseType = atb::infer::ElewiseParam::ELEWISE_MUL;

      atb::Operation* op = nullptr;
      auto st = atb::CreateOperation(param, &op);
      if (st != atb::NO_ERROR || op == nullptr) {
        std::cerr << "[MulBroadcastTest] CreateOperation failed, status=" << static_cast<int>(st) << "\n";
        return false;
      }

      atb::Tensor atb_x, atb_scale, atb_y;
      fillAtbTensor(x_npu, atb_x);
      fillAtbTensor(scale_npu, atb_scale);
      fillAtbTensor(y_npu, atb_y);

      atb::VariantPack vp;
      vp.inTensors = {atb_x, atb_scale};
      vp.outTensors = {atb_y};

      auto* ctx = getGlobalAtbContext();
      uint64_t workspace_size = 0;
      st = op->Setup(vp, workspace_size, ctx);
      if (st != atb::NO_ERROR) {
        std::cerr << "[MulBroadcastTest] Setup failed, status=" << static_cast<int>(st) << "\n";
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

      if (workspace_block_id != -1) getAscendMemoryManager().freeBlock(workspace_block_id);
      atb::DestroyOperation(op);

      if (st != atb::NO_ERROR) {
        std::cerr << "[MulBroadcastTest] Execute failed, status=" << static_cast<int>(st) << "\n";
        return false;
      }

      auto y_cpu = y_npu.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-1f, 1e-1f);
      if (!result.is_close) {
        std::cerr << "[MulBroadcastTest] mismatch\n";
        return false;
      }
      std::cout << "[MulBroadcastTest] shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
      }
      std::cout << "] PASSED\n";
    }
    return true;
  }
};
