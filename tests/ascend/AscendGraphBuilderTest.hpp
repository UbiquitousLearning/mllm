// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/backends/ascend/graph/AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/graph/AscendGraphExecutor.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "KernelTestHelper.hpp"
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Test fixture for Ascend Graph Builder tests
 */
class AscendGraphBuilderTest : public KernelTest {
 public:
  AscendGraphBuilderTest() = default;
  ~AscendGraphBuilderTest() override = default;

  /**
   * @brief Test simple Linear graph construction and execution
   *
   * This test creates a 1-operator graph (Linear) and verifies:
   * 1. Graph construction succeeds
   * 2. Graph can be executed without errors
   * 3. Workspace is allocated
   *
   * @return true if test passes, false otherwise
   */
  bool LinearGraphTest() {
    using namespace mllm;

    std::cout << "[GraphBuilderTest] Testing Linear graph construction" << std::endl;

    const int batch_size = 1;
    const int seq_len = 4;
    const int in_features = 8;
    const int out_features = 16;

    try {
      atb::infer::LinearParam linearParam;
      linearParam.transposeA = false;
      linearParam.transposeB = true;
      linearParam.hasBias = false;
      linearParam.outDataType = ACL_DT_UNDEFINED;
      linearParam.enAccum = false;
      linearParam.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
      linearParam.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;

      atb::Operation* linear_op = nullptr;
      auto st = atb::CreateOperation(linearParam, &linear_op);
      if (st != atb::NO_ERROR || linear_op == nullptr) {
        std::cerr << "Failed to create Linear operation, status=" << static_cast<int>(st) << std::endl;
        return false;
      }

      auto linearInferShape = [](const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                 atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
        if (inTensorDescs.size() < 2 || outTensorDescs.empty()) {
          return atb::NO_ERROR;
        }

        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto& output_desc = outTensorDescs.at(0);
        const auto& weight_desc = inTensorDescs.at(1);
        if (output_desc.shape.dimNum > 0 && weight_desc.shape.dimNum > 0) {
          output_desc.shape.dims[output_desc.shape.dimNum - 1] = weight_desc.shape.dims[0];
        }
        return atb::NO_ERROR;
      };

      ascend::AscendGraphBuilder builder;
      builder.beginGraph(
          "LinearTestGraph",
          {"input", "weight"},
          {"output"},
          linearInferShape
      );

      builder.addOperation(
          linear_op,
          {"input", "weight"},
          {"output"}
      );

      atb::Operation* graph_op = builder.build();
      if (graph_op == nullptr) {
        std::cerr << "Failed to build graph" << std::endl;
        atb::DestroyOperation(linear_op);
        return false;
      }

      std::cout << "[GraphBuilderTest] Graph built successfully: " << builder.graphName() << std::endl;

      auto input = Tensor::random({batch_size, seq_len, in_features}, -1.0f, 1.0f, kFloat16, kAscend);
      auto weight = Tensor::random({out_features, in_features}, -0.5f, 0.5f, kFloat16, kAscend);
      auto output = Tensor::empty({batch_size, seq_len, out_features}, kFloat16, kAscend).alloc();

      ascend::AscendGraphExecutor executor(graph_op, ascend::getGlobalAtbContext());
      std::vector<Tensor> inputs = {input, weight};
      std::vector<Tensor> outputs = {output};
      executor.execute(inputs, outputs);

      std::cout << "[GraphBuilderTest] Graph executed successfully" << std::endl;
      std::cout << "[GraphBuilderTest] Workspace size: " << executor.workspaceSize() << " bytes" << std::endl;
      // Do not destroy linear_op here.
      // Its lifetime is tied to the graph/executor teardown path, and
      // early destroy can cause a crash when returning from this test.

      std::cout << "[GraphBuilderTest] Test passed!" << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "[GraphBuilderTest] Exception: " << e.what() << std::endl;
      return false;
    }
  }

  /**
   * @brief Accuracy validation for Linear graph.
   *
   * Compares graph output against:
   * 1) Direct single Linear ATB operator output.
   * 2) CPU reference matmul implementation.
   */
  bool LinearGraphAccuracyTest() {
    using namespace mllm;

    std::cout << "[GraphBuilderTest] Testing Linear graph accuracy" << std::endl;

    const int batch_size = 1;
    const int seq_len = 4;
    const int in_features = 8;
    const int out_features = 16;

    auto buildLinearParam = []() {
      atb::infer::LinearParam p;
      p.transposeA = false;
      p.transposeB = true;
      p.hasBias = false;
      p.outDataType = ACL_DT_UNDEFINED;
      p.enAccum = false;
      p.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
      p.quantMode = atb::infer::LinearParam::QUANT_UNDEFINED;
      return p;
    };

    auto linearInferShape = [](const atb::SVector<atb::TensorDesc>& inTensorDescs,
                               atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
      if (inTensorDescs.size() < 2 || outTensorDescs.empty()) {
        return atb::NO_ERROR;
      }
      outTensorDescs.at(0) = inTensorDescs.at(0);
      auto& output_desc = outTensorDescs.at(0);
      const auto& weight_desc = inTensorDescs.at(1);
      if (output_desc.shape.dimNum > 0 && weight_desc.shape.dimNum > 0) {
        output_desc.shape.dims[output_desc.shape.dimNum - 1] = weight_desc.shape.dims[0];
      }
      return atb::NO_ERROR;
    };

    auto checkClose = [](const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float atol,
                         float rtol,
                         const std::string& tag) -> bool {
      if (actual.size() != expected.size()) {
        std::cerr << "[GraphBuilderTest][" << tag << "] size mismatch: actual=" << actual.size()
                  << ", expected=" << expected.size() << std::endl;
        return false;
      }

      float max_abs_diff = 0.0f;
      size_t mismatch_cnt = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        const float thr = atol + rtol * std::abs(expected[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        if (diff > thr) {
          if (mismatch_cnt < 5) {
            std::cerr << "[GraphBuilderTest][" << tag << "] mismatch idx=" << i
                      << ", actual=" << actual[i]
                      << ", expected=" << expected[i]
                      << ", diff=" << diff
                      << ", thr=" << thr << std::endl;
          }
          mismatch_cnt++;
        }
      }

      std::cout << "[GraphBuilderTest][" << tag << "] max_abs_diff=" << max_abs_diff
                << ", mismatches=" << mismatch_cnt << std::endl;
      return mismatch_cnt == 0;
    };

    try {
      // Deterministic input/weight data for stable regression checks.
      const size_t input_numel = static_cast<size_t>(batch_size) * seq_len * in_features;
      const size_t weight_numel = static_cast<size_t>(out_features) * in_features;

      std::vector<float> input_fp32(input_numel);
      std::vector<float> weight_fp32(weight_numel);
      for (size_t i = 0; i < input_numel; ++i) {
        input_fp32[i] = (static_cast<int>(i % 19) - 9) * 0.07f;
      }
      for (size_t i = 0; i < weight_numel; ++i) {
        weight_fp32[i] = (static_cast<int>(i % 23) - 11) * 0.03f;
      }

      std::vector<mllm_fp16_t> input_fp16(input_numel);
      std::vector<mllm_fp16_t> weight_fp16(weight_numel);
      for (size_t i = 0; i < input_numel; ++i) {
        input_fp16[i] = static_cast<mllm_fp16_t>(input_fp32[i]);
      }
      for (size_t i = 0; i < weight_numel; ++i) {
        weight_fp16[i] = static_cast<mllm_fp16_t>(weight_fp32[i]);
      }

      auto input_cpu = Tensor::fromVector<mllm_fp16_t>(input_fp16, {batch_size, seq_len, in_features}, kFloat16, kCPU);
      auto weight_cpu = Tensor::fromVector<mllm_fp16_t>(weight_fp16, {out_features, in_features}, kFloat16, kCPU);

      auto input = input_cpu.to(kAscend);
      auto weight = weight_cpu.to(kAscend);
      auto graph_output = Tensor::empty({batch_size, seq_len, out_features}, kFloat16, kAscend).alloc();
      auto direct_output = Tensor::empty({batch_size, seq_len, out_features}, kFloat16, kAscend).alloc();

      // 1) Build and execute graph Linear.
      atb::Operation* linear_graph_node_op = nullptr;
      auto st = atb::CreateOperation(buildLinearParam(), &linear_graph_node_op);
      if (st != atb::NO_ERROR || linear_graph_node_op == nullptr) {
        std::cerr << "Failed to create graph node Linear op, status=" << static_cast<int>(st) << std::endl;
        return false;
      }

      ascend::AscendGraphBuilder builder;
      builder.beginGraph("LinearAccuracyGraph", {"input", "weight"}, {"output"}, linearInferShape);
      builder.addOperation(linear_graph_node_op, {"input", "weight"}, {"output"});

      atb::Operation* graph_op = builder.build();
      if (graph_op == nullptr) {
        std::cerr << "Failed to build accuracy graph" << std::endl;
        atb::DestroyOperation(linear_graph_node_op);
        return false;
      }

      {
        ascend::AscendGraphExecutor executor(graph_op, ascend::getGlobalAtbContext());
        std::vector<Tensor> inputs = {input, weight};
        std::vector<Tensor> outputs = {graph_output};
        executor.execute(inputs, outputs);
      }

      // 2) Execute direct single Linear op with the same input.
      atb::Operation* direct_linear_op = nullptr;
      st = atb::CreateOperation(buildLinearParam(), &direct_linear_op);
      if (st != atb::NO_ERROR || direct_linear_op == nullptr) {
        std::cerr << "Failed to create direct Linear op, status=" << static_cast<int>(st) << std::endl;
        return false;
      }

      atb::VariantPack pack;
      pack.inTensors.resize(2);
      pack.outTensors.resize(1);
      ascend::fillAtbTensor(input, pack.inTensors[0]);
      ascend::fillAtbTensor(weight, pack.inTensors[1]);
      ascend::fillAtbTensor(direct_output, pack.outTensors[0]);

      uint64_t workspace_size = 0;
      st = direct_linear_op->Setup(pack, workspace_size, ascend::getGlobalAtbContext());
      if (st != atb::NO_ERROR) {
        atb::DestroyOperation(direct_linear_op);
        std::cerr << "Direct Linear Setup failed, status=" << static_cast<int>(st) << std::endl;
        return false;
      }

      void* workspace = nullptr;
      if (workspace_size > 0) {
        auto acl_ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (acl_ret != ACL_SUCCESS) {
          atb::DestroyOperation(direct_linear_op);
          std::cerr << "Direct Linear workspace alloc failed, acl_ret=" << static_cast<int>(acl_ret) << std::endl;
          return false;
        }
      }

      st = direct_linear_op->Execute(pack,
                                     reinterpret_cast<uint8_t*>(workspace),
                                     workspace_size,
                                     ascend::getGlobalAtbContext());
      if (workspace != nullptr) {
        aclrtFree(workspace);
      }
      atb::DestroyOperation(direct_linear_op);

      if (st != atb::NO_ERROR) {
        std::cerr << "Direct Linear Execute failed, status=" << static_cast<int>(st) << std::endl;
        return false;
      }

      ascend::syncGlobalAtbStream();

      // 3) Compare graph output vs direct op output.
      auto graph_host = ascend::copyAscendTensorToHost(graph_output);
      auto direct_host = ascend::copyAscendTensorToHost(direct_output);
      const bool graph_vs_direct_ok = checkClose(graph_host, direct_host, 3e-2f, 3e-2f, "graph_vs_direct");

      // 4) CPU reference: Y = X * W^T.
      std::vector<float> cpu_ref(static_cast<size_t>(batch_size) * seq_len * out_features, 0.0f);
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
          for (int o = 0; o < out_features; ++o) {
            float acc = 0.0f;
            for (int i = 0; i < in_features; ++i) {
              const size_t x_idx = (static_cast<size_t>(b) * seq_len + s) * in_features + i;
              const size_t w_idx = static_cast<size_t>(o) * in_features + i;
              acc += input_fp32[x_idx] * weight_fp32[w_idx];
            }
            const size_t y_idx = (static_cast<size_t>(b) * seq_len + s) * out_features + o;
            cpu_ref[y_idx] = acc;
          }
        }
      }

      const bool graph_vs_cpu_ok = checkClose(graph_host, cpu_ref, 8e-2f, 8e-2f, "graph_vs_cpu");
      if (!graph_vs_direct_ok || !graph_vs_cpu_ok) {
        std::cerr << "[GraphBuilderTest] Accuracy check failed" << std::endl;
        return false;
      }

      std::cout << "[GraphBuilderTest] Accuracy check passed" << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "[GraphBuilderTest] Accuracy test exception: " << e.what() << std::endl;
      return false;
    }
  }

  /**
   * @brief Test that AscendGraphBuilder can be created and destroyed
   *
   * @return true if test passes
   */
  bool BasicCreationTest() {
    using namespace mllm;
    std::cout << "[GraphBuilderTest] Testing basic builder creation" << std::endl;

    try {
      ascend::AscendGraphBuilder builder;
      std::cout << "[GraphBuilderTest] Builder created successfully" << std::endl;
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[GraphBuilderTest] Failed to create builder: " << e.what() << std::endl;
      return false;
    }
  }
};
