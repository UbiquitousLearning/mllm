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
#include <atb/infer_op_params.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

class AscendLinearSoftmaxGraphTest : public KernelTest {
 public:
  AscendLinearSoftmaxGraphTest() = default;
  ~AscendLinearSoftmaxGraphTest() override = default;

  bool LinearSoftmaxGraphPassTest() {
    using namespace mllm;

    std::cout << "[GraphBuilderTest] Testing Linear+Softmax graph construction" << std::endl;

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

      atb::infer::SoftmaxParam softmaxParam;
      softmaxParam.axes.push_back(2);  // [B, S, C] -> softmax on C

      atb::Operation* softmax_op = nullptr;
      st = atb::CreateOperation(softmaxParam, &softmax_op);
      if (st != atb::NO_ERROR || softmax_op == nullptr) {
        std::cerr << "Failed to create Softmax operation, status=" << static_cast<int>(st) << std::endl;
        atb::DestroyOperation(linear_op);
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
          "LinearSoftmaxTestGraph",
          {"input", "weight"},
          {"output"},
          linearInferShape
      );

      builder.addOperation(
          linear_op,
          {"input", "weight"},
          {"linear_out"}
      );

      builder.addOperation(
          softmax_op,
          {"linear_out"},
          {"output"}
      );

      atb::Operation* graph_op = builder.build();
      if (graph_op == nullptr) {
        std::cerr << "Failed to build Linear+Softmax graph" << std::endl;
        atb::DestroyOperation(softmax_op);
        atb::DestroyOperation(linear_op);
        return false;
      }

      auto input = Tensor::random({batch_size, seq_len, in_features}, -1.0f, 1.0f, kFloat16, kAscend);
      auto weight = Tensor::random({out_features, in_features}, -0.5f, 0.5f, kFloat16, kAscend);
      auto output = Tensor::empty({batch_size, seq_len, out_features}, kFloat16, kAscend).alloc();

      ascend::AscendGraphExecutor executor(graph_op, ascend::getGlobalAtbContext());
      std::vector<Tensor> inputs = {input, weight};
      std::vector<Tensor> outputs = {output};
      executor.execute(inputs, outputs);

      std::cout << "[GraphBuilderTest] Linear+Softmax graph executed successfully" << std::endl;
      std::cout << "[GraphBuilderTest] Workspace size: " << executor.workspaceSize() << " bytes" << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "[GraphBuilderTest] Exception: " << e.what() << std::endl;
      return false;
    }
  }

  bool LinearSoftmaxGraphAccuracyTest() {
    using namespace mllm;

    std::cout << "[GraphBuilderTest] Testing Linear+Softmax graph accuracy" << std::endl;

    const int batch_size = 1;
    const int seq_len = 4;
    const int in_features = 8;
    const int out_features = 16;

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
      const size_t input_numel = static_cast<size_t>(batch_size) * seq_len * in_features;
      const size_t weight_numel = static_cast<size_t>(out_features) * in_features;

      std::vector<float> input_fp32(input_numel);
      std::vector<float> weight_fp32(weight_numel);
      for (size_t i = 0; i < input_numel; ++i) {
        input_fp32[i] = (static_cast<int>(i % 17) - 8) * 0.08f;
      }
      for (size_t i = 0; i < weight_numel; ++i) {
        weight_fp32[i] = (static_cast<int>(i % 29) - 14) * 0.025f;
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
      auto output = Tensor::empty({batch_size, seq_len, out_features}, kFloat16, kAscend).alloc();

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

      atb::infer::SoftmaxParam softmaxParam;
      softmaxParam.axes.push_back(2);

      atb::Operation* softmax_op = nullptr;
      st = atb::CreateOperation(softmaxParam, &softmax_op);
      if (st != atb::NO_ERROR || softmax_op == nullptr) {
        std::cerr << "Failed to create Softmax operation, status=" << static_cast<int>(st) << std::endl;
        atb::DestroyOperation(linear_op);
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
          "LinearSoftmaxAccuracyGraph",
          {"input", "weight"},
          {"output"},
          linearInferShape
      );

      builder.addOperation(linear_op, {"input", "weight"}, {"linear_out"});
      builder.addOperation(softmax_op, {"linear_out"}, {"output"});

      atb::Operation* graph_op = builder.build();
      if (graph_op == nullptr) {
        std::cerr << "Failed to build Linear+Softmax accuracy graph" << std::endl;
        atb::DestroyOperation(softmax_op);
        atb::DestroyOperation(linear_op);
        return false;
      }

      {
        ascend::AscendGraphExecutor executor(graph_op, ascend::getGlobalAtbContext());
        std::vector<Tensor> inputs = {input, weight};
        std::vector<Tensor> outputs = {output};
        executor.execute(inputs, outputs);
      }

      auto graph_host = ascend::copyAscendTensorToHost(output);

      std::vector<float> cpu_ref(static_cast<size_t>(batch_size) * seq_len * out_features, 0.0f);
      std::vector<float> logits(static_cast<size_t>(batch_size) * seq_len * out_features, 0.0f);

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
            logits[y_idx] = acc;
          }

          float max_logit = logits[(static_cast<size_t>(b) * seq_len + s) * out_features];
          for (int o = 1; o < out_features; ++o) {
            const size_t idx = (static_cast<size_t>(b) * seq_len + s) * out_features + o;
            max_logit = std::max(max_logit, logits[idx]);
          }

          float exp_sum = 0.0f;
          for (int o = 0; o < out_features; ++o) {
            const size_t idx = (static_cast<size_t>(b) * seq_len + s) * out_features + o;
            cpu_ref[idx] = std::exp(logits[idx] - max_logit);
            exp_sum += cpu_ref[idx];
          }
          for (int o = 0; o < out_features; ++o) {
            const size_t idx = (static_cast<size_t>(b) * seq_len + s) * out_features + o;
            cpu_ref[idx] /= exp_sum;
          }
        }
      }

      const bool ok = checkClose(graph_host, cpu_ref, 3e-2f, 3e-2f, "linear_softmax_vs_cpu");
      if (!ok) {
        std::cerr << "[GraphBuilderTest] Linear+Softmax accuracy check failed" << std::endl;
        return false;
      }

      std::cout << "[GraphBuilderTest] Linear+Softmax accuracy check passed" << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "[GraphBuilderTest] Accuracy test exception: " << e.what() << std::endl;
      return false;
    }
  }
};
