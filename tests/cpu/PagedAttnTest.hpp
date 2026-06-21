// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"
#include "mllm/nn/Functional.hpp"

class PagedModule : public mllm::nn::Module {
  bool return_attn_;
  mllm::nn::PagedAttn paged_attn_layer;

 public:
  explicit PagedModule(int32_t head_q, int32_t head_kv, bool attn_output) {
    paged_attn_layer = reg<mllm::nn::PagedAttn>("paged", head_q / head_kv, false, false, true);
    return_attn_ = attn_output;
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs, const std::vector<mllm::AnyValue>& args) override {
    auto& Q = inputs[0];
    auto& K = inputs[1];
    auto& V = inputs[2];
    auto& index = inputs[3];
    auto& causal_mask = inputs[4];

    auto o = paged_attn_layer(Q, K, V, index, causal_mask);
    if (return_attn_) {
      return {o[0], o[1]};
    } else {
      return {o[0]};
    }
    return {};
  }
};

class PagedAttnTest : public KernelTest {
 public:
  PagedAttnTest() = default;
  ~PagedAttnTest() override = default;

  bool oneCase(const std::pair<mllm::Tensor::shape_t, mllm::Tensor::shape_t>& shape) {
    auto q_shape = shape.first;
    auto kv_shape = shape.second;
    auto B = q_shape[0];
    auto S_Q = q_shape[1];
    auto H_Q = q_shape[2];
    auto S_KV = kv_shape[1];
    auto H_KV = kv_shape[2];
    auto D = q_shape[3];

    // [B, S, H, D] - BSHD format
    auto Q = mllm::Tensor::random(q_shape, -1, 1, mllm::kFloat32, mllm::kCPU);
    auto K = mllm::Tensor::random(kv_shape, -1, 1, mllm::kFloat32, mllm::kCPU);
    auto V = mllm::Tensor::random(kv_shape, -1, 1, mllm::kFloat32, mllm::kCPU);

    // Calculate head repeat times (Query_Head should be divisible by Key_Head)
    int32_t head_repeat_times = H_Q / H_KV;
    auto net = PagedModule(H_Q, H_KV, true);

    // Build Index
    auto index = mllm::Tensor::arange(0, S_KV, 1, mllm::kInt32, mllm::kCPU);
    auto mask = mllm::Tensor::zeros({S_Q, S_KV}, mllm::kFloat32, mllm::kCPU);
    auto mask_data = mask.ptr<mllm::mllm_fp32_t>();

    if (S_Q != 1) {
      for (int i = 0; i < S_Q; ++i) {
        for (int j = 0; j < S_KV; ++j) {
          if (j > i) {
            mask_data[i * S_KV + j] = mllm::DataTypeInfo<mllm::mllm_fp32_t>::min();
          } else {
            mask_data[i * S_KV + j] = 0.0f;
          }
        }
      }
    }

    // Build mask
    auto output = net(Q, K, V, index, mask);

    auto scale = 1.0f / sqrtf(static_cast<float>(D));

    // Calculate ref attention
    mllm::Tensor ref_attn = mllm::Tensor::nil();
    mllm::Tensor ref_o = mllm::Tensor::nil();
    {
      // [B, S, H, D] -> [B, H, S, D]
      Q = Q.transpose(1, 2);
      K = K.transpose(1, 2);
      V = V.transpose(1, 2);

      auto attn_weight = mllm::nn::functional::matmul(Q, K.repeat(head_repeat_times, 1), false, true);
      attn_weight = attn_weight * scale;
      attn_weight = attn_weight + mask;
      attn_weight = mllm::nn::functional::softmax(attn_weight, -1);
      ref_attn = attn_weight;
      ref_o = mllm::nn::functional::matmul(attn_weight, V.repeat(head_repeat_times, 1)).transpose(1, 2);
    }

    auto output_ok = mllm::test::allClose(ref_o, output[0]);
    auto attn_output_ok = mllm::test::allClose(ref_attn, output[1]);

    if (!output_ok || !attn_output_ok) {
      mllm::print(output_ok);
      mllm::print(attn_output_ok);
      MLLM_ERROR("output_ok: {}, attn_output_ok: {}", output_ok, attn_output_ok);
    }

    return output_ok && attn_output_ok;
  }

  bool manyCases(const std::vector<std::pair<mllm::Tensor::shape_t, mllm::Tensor::shape_t>>& shapes) {
    for (const auto& s : shapes) {
      if (!oneCase(s)) { return false; }
    }
    return true;
  }
};
