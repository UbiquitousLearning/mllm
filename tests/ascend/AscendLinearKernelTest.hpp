// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/nn/Functional.hpp"
#include "KernelTestHelper.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/backends/ascend/memory/AscendMemoryManager.hpp"
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_trans_quant_param.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_clamp.h>
#include <atb/atb_infer.h>
#include <atb/infer_op_params.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>


class AscendLinearKernelTest : public KernelTest {
 public:
  AscendLinearKernelTest() = default;
  ~AscendLinearKernelTest() override = default;

  bool LinearFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, int, int>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      auto input_shape = std::get<0>(test_case);
      int in_channels = std::get<1>(test_case);
      int out_channels = std::get<2>(test_case);

      std::cout << "[LinearTest] Testing shape=[";
      for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
      }
      std::cout << "], in=" << in_channels << ", out=" << out_channels << std::endl;

      // 1. Construct random FP16 inputs on CPU
      // x: [M, K] where K = in_channels
      Tensor x_cpu = Tensor::random(input_shape, -1, 1, kFloat16, kCPU);

      // Weight shape for ATB: [K, N] where K=in_channels, N=out_channels
      Tensor weight_cpu = Tensor::random({in_channels, out_channels}, -0.5, 0.5, kFloat16, kCPU);

      // 2. Compute reference result on CPU
      // y = x @ weight, where x is [M, K], weight is [K, N], output is [M, N]
      auto output_shape = input_shape;
      output_shape[output_shape.size() - 1] = out_channels;
      Tensor ref_cpu = Tensor::zeros(output_shape, kFloat16, kCPU);

      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* w_ptr = weight_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        size_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
          batch_size *= input_shape[i];
        }

        for (size_t b = 0; b < batch_size; ++b) {
          for (int o = 0; o < out_channels; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_channels; ++i) {
              float x_val = MLLM_FP16_TO_FP32(x_ptr[b * in_channels + i]);
              float w_val = MLLM_FP16_TO_FP32(w_ptr[i * out_channels + o]);  // weight is [K, N]
              sum += x_val * w_val;
            }
            r_ptr[b * out_channels + o] = MLLM_FP32_TO_FP16(sum);
          }
        }
      }

      // 3. Move inputs to Ascend and run Linear via matmul
      auto x_ascend = x_cpu.to(kAscend);
      auto weight_ascend = weight_cpu.to(kAscend);

      // Use matmul: y = x @ weight
      auto y_ascend = nn::functional::matmul(x_ascend, weight_ascend, false, false);

      // 4. Move result back to CPU and compare with reference
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        std::cout << "[LinearTest] FAILED!" << std::endl;
        return false;
      }
      std::cout << "[LinearTest] PASSED" << std::endl;
    }
    return true;
  }

  bool LinearWithBiasFloat16Test(const std::vector<std::tuple<mllm::Tensor::shape_t, int, int>>& test_cases) {
    using namespace mllm;  // NOLINT
    for (auto& test_case : test_cases) {
      auto input_shape = std::get<0>(test_case);
      int in_channels = std::get<1>(test_case);
      int out_channels = std::get<2>(test_case);

      std::cout << "[LinearWithBiasTest] Testing shape=[";
      for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
      }
      std::cout << "], in=" << in_channels << ", out=" << out_channels << std::endl;

      // 1. Create random input, weight and bias on CPU
      Tensor x_cpu = Tensor::random(input_shape, -1, 1, kFloat16, kCPU);
      // Weight shape: [out_channels, in_channels]
      Tensor weight_cpu = Tensor::random({out_channels, in_channels}, -0.5, 0.5, kFloat16, kCPU);
      // Bias shape: [1, out_channels] for ATB Linear (2D tensor required)
      Tensor bias_cpu = Tensor::random({1, out_channels}, -0.1, 0.1, kFloat16, kCPU);

      // 2. Compute reference result on CPU
      auto output_shape = input_shape;
      output_shape[output_shape.size() - 1] = out_channels;
      Tensor ref_cpu = Tensor::zeros(output_shape, kFloat16, kCPU);

      {
        auto* x_ptr = x_cpu.ptr<mllm_fp16_t>();
        auto* w_ptr = weight_cpu.ptr<mllm_fp16_t>();
        auto* b_ptr = bias_cpu.ptr<mllm_fp16_t>();
        auto* r_ptr = ref_cpu.ptr<mllm_fp16_t>();

        size_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
          batch_size *= input_shape[i];
        }

        // y = x @ W^T + b, where W is [out_channels, in_channels]
        for (size_t b = 0; b < batch_size; ++b) {
          for (int o = 0; o < out_channels; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_channels; ++i) {
              float x_val = MLLM_FP16_TO_FP32(x_ptr[b * in_channels + i]);
              float w_val = MLLM_FP16_TO_FP32(w_ptr[o * in_channels + i]);
              sum += x_val * w_val;
            }
            float bias_val = MLLM_FP16_TO_FP32(b_ptr[o]);
            sum += bias_val;
            r_ptr[b * out_channels + o] = MLLM_FP32_TO_FP16(sum);
          }
        }
      }

      // 3. Move tensors to Ascend and run linear
      auto x_ascend = x_cpu.to(kAscend);
      auto weight_ascend = weight_cpu.to(kAscend);
      auto bias_ascend = bias_cpu.to(kAscend);

      // Use nn::functional::linear directly
      auto y_ascend = nn::functional::linear(x_ascend, weight_ascend, bias_ascend);

      // 4. Compare result with reference
      auto y_cpu = y_ascend.to(kCPU);
      auto result = mllm::test::allClose(y_cpu, ref_cpu, 1e-2f, 1e-2f);
      if (!result.is_close) {
        std::cout << "[LinearWithBiasTest] FAILED!" << std::endl;
        return false;
      }
      std::cout << "[LinearWithBiasTest] PASSED" << std::endl;
    }
    return true;
  }

  // -----------------------------------------------------------------------
  // LinearW8A8EndToEndPipelineTest
  //
  // Verifies the corrected W8A8 activation-quantization pipeline on 310B.
  // The pipeline stays entirely in FP16 — no FP32 intermediate is needed
  // because scale_x = max(|x|)/127, so x*inv_scale is bounded by ±127
  // (well within FP16 range of ±65504).
  //
  //   x_fp16 → [ATB ELEWISE_MULS, *inv_scale]  → x_scaled_fp16   (≤±127)
  //           → [aclnnRound, FP16]              → x_round_fp16    (integer)
  //           → [aclnnClamp(-128,127), FP16]    → x_clamped_fp16
  //           → [aclnnCast, FP16→INT8]          → x_int8          (safe: integer cast)
  //
  //   x_int8, weight_int8, bias_i32, deq_scale → [ATB Linear W8A8] → y_fp16
  //
  // Why NOT FP32: ATB ELEWISE_MULS does not support FP32 input on 310B
  //   (verified: ELEWISE_CAST FP16→FP32 works, but ELEWISE_MULS FP32 fails).
  // Why round before cast: aclnnCast truncates toward zero; after aclnnRound
  //   the values are already integers, so truncation equals rounding.
  //
  // The test returns false at the first failing step with a diagnostic message.
  // Activation quantization match vs CPU reference is printed informationally.
  // Final output tolerance: atol=0.5, rtol=5% (INT8 quant rounding budget).
  // -----------------------------------------------------------------------
  bool LinearW8A8EndToEndPipelineTest(
      const std::vector<std::tuple<mllm::Tensor::shape_t, int, int>>& test_cases) {
    using namespace mllm;          // NOLINT
    using namespace mllm::ascend;  // NOLINT

    for (const auto& tc : test_cases) {
      const auto input_shape = std::get<0>(tc);
      const int in_channels = std::get<1>(tc);
      const int out_channels = std::get<2>(tc);
      size_t num_tokens = 1;
      for (size_t i = 0; i + 1 < input_shape.size(); ++i) num_tokens *= input_shape[i];

      std::cout << "[W8A8PipelineTest] M=" << num_tokens
                << " K=" << in_channels
                << " N=" << out_channels << std::endl;

      // -------------------------------------------------------------------
      // 1. Random FP32 reference data
      // -------------------------------------------------------------------
      srand(42);
      std::vector<float> x_fp32(num_tokens * in_channels), w_fp32(out_channels * in_channels);
      for (auto& v : x_fp32) v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
      for (auto& v : w_fp32) v = static_cast<float>(rand()) / RAND_MAX - 0.5f;

      // -------------------------------------------------------------------
      // 2. Quantize weight per-channel: scale_w[n] = max(|w[n,:]|) / 127
      // -------------------------------------------------------------------
      std::vector<float> scale_w(out_channels);
      for (int n = 0; n < out_channels; ++n) {
        float absmax = 1e-8f;
        for (int k = 0; k < in_channels; ++k) absmax = std::max(absmax, std::abs(w_fp32[n * in_channels + k]));
        scale_w[n] = absmax / 127.0f;
      }
      Tensor w_int8_cpu = Tensor::empty({out_channels, in_channels}, kInt8, kCPU).alloc();
      {
        auto* ptr = w_int8_cpu.ptr<int8_t>();
        for (int n = 0; n < out_channels; ++n) {
          for (int k = 0; k < in_channels; ++k) {
            float v = w_fp32[n * in_channels + k] / scale_w[n];
            ptr[n * in_channels + k] = static_cast<int8_t>(
                std::max(-128.0f, std::min(127.0f, std::round(v))));
          }
        }
      }

      // -------------------------------------------------------------------
      // 3. Activation scale (per-tensor): scale_x = max(|x|) / 127
      //    inv_scale = 1/scale_x  ≤ 127 after multiplication (no FP16 overflow)
      // -------------------------------------------------------------------
      float amax = 1e-8f;
      for (auto v : x_fp32) amax = std::max(amax, std::abs(v));
      const float scale_x   = amax / 127.0f;
      const float inv_scale = 1.0f / scale_x;

      // -------------------------------------------------------------------
      // 4. CPU reference activation quantization: round then clamp
      // -------------------------------------------------------------------
      std::vector<int8_t> x_q_ref(num_tokens * in_channels);
      for (size_t i = 0; i < num_tokens * in_channels; ++i) {
        float v = std::round(x_fp32[i] * inv_scale);
        v = std::max(-128.0f, std::min(127.0f, v));
        x_q_ref[i] = static_cast<int8_t>(v);
      }

      // -------------------------------------------------------------------
      // 5. Build deq_scale[n] = scale_x * scale_w[n], convert to uint64
      // -------------------------------------------------------------------
      std::vector<float> deq_fp32(out_channels);
      for (int n = 0; n < out_channels; ++n) deq_fp32[n] = scale_x * scale_w[n];

      uint64_t* deq_u64_host = nullptr;
      uint64_t  deq_u64_cnt  = 0;
      auto acl_st = aclnnTransQuantParam(
          deq_fp32.data(), static_cast<uint64_t>(out_channels),
          nullptr, 0, &deq_u64_host, &deq_u64_cnt);
      if (acl_st != ACL_SUCCESS || deq_u64_host == nullptr) {
        std::cout << "[W8A8PipelineTest] aclnnTransQuantParam FAILED, status=" << acl_st << std::endl;
        return false;
      }
      Tensor deq_cpu = Tensor::empty({out_channels}, kUInt64, kCPU).alloc();
      std::memcpy(deq_cpu.ptr<uint64_t>(), deq_u64_host, deq_u64_cnt * sizeof(uint64_t));
      free(deq_u64_host);

      // Zero INT32 bias [1, N] — mandatory on 310B quant path
      Tensor bias_cpu = Tensor::empty({1, out_channels}, kInt32, kCPU).alloc();
      std::memset(bias_cpu.ptr<int32_t>(), 0, static_cast<size_t>(out_channels) * sizeof(int32_t));

      // -------------------------------------------------------------------
      // 6. CPU reference output: y_ref[m,n] = sum_k(x_q[m,k]*w[n,k]) * deq[n]
      // -------------------------------------------------------------------
      std::vector<float> y_ref(num_tokens * out_channels, 0.0f);
      {
        const auto* xq = x_q_ref.data();
        const auto* wq = w_int8_cpu.ptr<int8_t>();
        for (size_t m = 0; m < num_tokens; ++m) {
          for (int n = 0; n < out_channels; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < in_channels; ++k) {
              acc += static_cast<int32_t>(xq[m * in_channels + k])
                  * static_cast<int32_t>(wq[n * in_channels + k]);
            }
            y_ref[m * out_channels + n] = static_cast<float>(acc) * deq_fp32[n];
          }
        }
      }

      // -------------------------------------------------------------------
      // 7. Upload x as FP16 to NPU
      // -------------------------------------------------------------------
      Tensor x_fp16_cpu = Tensor::empty(input_shape, kFloat16, kCPU).alloc();
      for (size_t i = 0; i < num_tokens * in_channels; ++i) {
        x_fp16_cpu.ptr<mllm_fp16_t>()[i] = MLLM_FP32_TO_FP16(x_fp32[i]);
      }
      auto x_fp16_npu = x_fp16_cpu.to(kAscend);

      // -------------------------------------------------------------------
      // Step A: ATB ELEWISE_MULS  x_fp16 * inv_scale  (result bounded by ±127)
      // -------------------------------------------------------------------
      Tensor x_scaled_npu = Tensor::empty(input_shape, kFloat16, kAscend).alloc();
      if (!runElewiseMuls(x_fp16_npu, x_scaled_npu, inv_scale)) {
        std::cout << "[W8A8PipelineTest] FAILED at Step A: ATB ELEWISE_MULS FP16" << std::endl;
        return false;
      }
      std::cout << "[W8A8PipelineTest] Step A (ELEWISE_MULS FP16 *inv_scale): OK" << std::endl;

      // -------------------------------------------------------------------
      // Step B: aclnnRound  (round-to-nearest-even, FP16 → FP16)
      //         After this, values are integers stored in FP16.
      // -------------------------------------------------------------------
      Tensor x_round_npu = Tensor::empty(input_shape, kFloat16, kAscend).alloc();
      if (!runAclnnRound(x_scaled_npu, x_round_npu, ACL_FLOAT16)) {
        std::cout << "[W8A8PipelineTest] FAILED at Step B: aclnnRound FP16" << std::endl;
        return false;
      }
      std::cout << "[W8A8PipelineTest] Step B (aclnnRound FP16): OK" << std::endl;

      // -------------------------------------------------------------------
      // Step C: aclnnClamp  [-128, 127]  (FP16 → FP16, explicit saturation)
      // -------------------------------------------------------------------
      Tensor x_clamped_npu = Tensor::empty(input_shape, kFloat16, kAscend).alloc();
      if (!runAclnnClamp(x_round_npu, x_clamped_npu, ACL_FLOAT16, -128.0f, 127.0f)) {
        std::cout << "[W8A8PipelineTest] FAILED at Step C: aclnnClamp FP16" << std::endl;
        return false;
      }
      std::cout << "[W8A8PipelineTest] Step C (aclnnClamp FP16 [-128,127]): OK" << std::endl;

      // -------------------------------------------------------------------
      // Step D: aclnnCast  FP16 → INT8
      //         Values are already integers after Round+Clamp, so truncation
      //         is lossless.
      // -------------------------------------------------------------------
      Tensor x_int8_npu = Tensor::empty(input_shape, kInt8, kAscend).alloc();
      if (!runAclnnCast(x_clamped_npu, x_int8_npu, ACL_FLOAT16, ACL_INT8)) {
        std::cout << "[W8A8PipelineTest] FAILED at Step D: aclnnCast FP16→INT8" << std::endl;
        return false;
      }
      std::cout << "[W8A8PipelineTest] Step D (aclnnCast FP16→INT8): OK" << std::endl;

      // Informational: how well does NPU activation quantization match the CPU reference?
      {
        auto x_int8_host = x_int8_npu.to(kCPU);
        const auto* xnpu = x_int8_host.ptr<int8_t>();
        size_t match = 0;
        for (size_t i = 0; i < num_tokens * in_channels; ++i) {
          if (xnpu[i] == x_q_ref[i]) ++match;
        }
        std::cout << "[W8A8PipelineTest] Activation quant match vs CPU ref: "
                  << match << "/" << (num_tokens * in_channels)
                  << " (" << 100.0 * match / (num_tokens * in_channels) << "%)" << std::endl;
      }

      // -------------------------------------------------------------------
      // Step E: ATB Linear  W8A8 (PER_CHANNEL dequant)  →  y_fp16
      // -------------------------------------------------------------------
      auto w_int8_npu = w_int8_cpu.to(kAscend);
      auto bias_npu   = bias_cpu.to(kAscend);
      auto deq_npu    = deq_cpu.to(kAscend);

      auto out_shape = input_shape;
      out_shape[out_shape.size() - 1] = out_channels;
      Tensor y_npu = Tensor::empty(out_shape, kFloat16, kAscend).alloc();

      if (!runAtbW8A8Linear(x_int8_npu, w_int8_npu, bias_npu, deq_npu, y_npu)) {
        std::cout << "[W8A8PipelineTest] FAILED at Step E: ATB Linear W8A8" << std::endl;
        return false;
      }
      std::cout << "[W8A8PipelineTest] Step E (ATB Linear W8A8): OK" << std::endl;

      // -------------------------------------------------------------------
      // 8. Accuracy check vs CPU reference
      // -------------------------------------------------------------------
      auto y_host = y_npu.to(kCPU);
      const auto* yp = y_host.ptr<mllm_fp16_t>();
      float max_err = 0.0f;
      bool passed = true;
      for (size_t i = 0; i < num_tokens * out_channels; ++i) {
        float got = MLLM_FP16_TO_FP32(yp[i]);
        float err = std::abs(got - y_ref[i]);
        float tol = 0.5f + 0.05f * std::abs(y_ref[i]);  // atol=0.5, rtol=5%
        if (err > tol) {
          if (passed) {
            std::cout << "[W8A8PipelineTest] First mismatch at i=" << i
                      << " got=" << got << " ref=" << y_ref[i]
                      << " err=" << err << " tol=" << tol << std::endl;
          }
          passed = false;
        }
        max_err = std::max(max_err, err);
      }
      std::cout << "[W8A8PipelineTest] M=" << num_tokens
                << " K=" << in_channels
                << " N=" << out_channels
                << (passed ? " PASSED" : " FAILED")
                << " max_err=" << max_err << std::endl;
      if (!passed) return false;
    }
    return true;
  }

 private:
  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------

  // Build a contiguous aclTensor from an MLLM Tensor with an explicit ACL dtype.
  // Caller must call aclDestroyTensor() on the returned pointer.
  static aclTensor* makeAclTensor(const mllm::Tensor& t, aclDataType dt) {
    const auto& sh = t.shape();
    int nd = static_cast<int>(sh.size());
    std::vector<int64_t> dims(nd), strides(nd);
    for (int i = 0; i < nd; ++i) dims[i] = static_cast<int64_t>(sh[i]);
    int64_t s = 1;
    for (int i = nd - 1; i >= 0; --i) { strides[i] = s; s *= dims[i]; }
    return aclCreateTensor(dims.data(), nd, dt, strides.data(), 0,
                           ACL_FORMAT_ND, dims.data(), nd, t.ptr<void>());
  }

  // Run ATB ELEWISE_MULS: y = x * scalar.
  // Works for any ATB-supported float dtype (FP16 on 310B; FP32 NOT supported on 310B).
  static bool runElewiseMuls(const mllm::Tensor& x, mllm::Tensor& y, float scalar) {
    using namespace mllm::ascend;  // NOLINT
    atb::infer::ElewiseParam ep;
    ep.elewiseType       = atb::infer::ElewiseParam::ELEWISE_MULS;
    ep.mulsParam.varAttr = scalar;
    atb::Operation* op = nullptr;
    if (atb::CreateOperation(ep, &op) != atb::NO_ERROR || op == nullptr) return false;
    atb::Tensor atb_x, atb_y;
    fillAtbTensor(x, atb_x);
    fillAtbTensor(y, atb_y);
    atb::VariantPack vp;
    vp.inTensors  = {atb_x};
    vp.outTensors = {atb_y};
    uint64_t ws_size = 0;
    if (op->Setup(vp, ws_size, getGlobalAtbContext()) != atb::NO_ERROR) {
      atb::DestroyOperation(op); return false;
    }
    void* ws = nullptr; int ws_bid = -1;
    if (ws_size > 0) {
      getAscendMemoryManager().allocateBlock(static_cast<uint32_t>(ws_size), ws_bid);
      getAscendMemoryManager().getBlockPtr(ws_bid, ws);
    }
    bool ok = (op->Execute(vp, reinterpret_cast<uint8_t*>(ws), ws_size, getGlobalAtbContext())
               == atb::NO_ERROR);
    syncGlobalAtbStream();
    if (ws_bid != -1) getAscendMemoryManager().freeBlock(ws_bid);
    atb::DestroyOperation(op);
    return ok;
  }

  // Run aclnnRound synchronously on an NPU tensor.
  // src and dst must be kAscend, same shape, and dtype must match dt.
  // Supports: ACL_FLOAT16, ACL_FLOAT, ACL_INT32, ACL_INT64, etc.
  static bool runAclnnRound(const mllm::Tensor& src, mllm::Tensor& dst, aclDataType dt) {
    using namespace mllm::ascend;  // NOLINT
    aclTensor* as = makeAclTensor(src, dt);
    aclTensor* ad = makeAclTensor(dst, dt);
    uint64_t ws_size = 0;
    aclOpExecutor* exe = nullptr;
    aclError ret = aclnnRoundGetWorkspaceSize(as, ad, &ws_size, &exe);
    if (ret != ACL_SUCCESS) {
      aclDestroyTensor(as); aclDestroyTensor(ad);
      return false;
    }
    void* ws = nullptr; int ws_bid = -1;
    if (ws_size > 0) {
      getAscendMemoryManager().allocateBlock(static_cast<uint32_t>(ws_size), ws_bid);
      getAscendMemoryManager().getBlockPtr(ws_bid, ws);
    }
    ret = aclnnRound(ws, ws_size, exe, getGlobalAtbStream());
    syncGlobalAtbStream();
    if (ws_bid != -1) getAscendMemoryManager().freeBlock(ws_bid);
    aclDestroyTensor(as);
    aclDestroyTensor(ad);
    return ret == ACL_SUCCESS;
  }

  // Run aclnnClamp synchronously on an NPU tensor.
  // src and dst must be kAscend, same shape, dtype must match dt.
  static bool runAclnnClamp(const mllm::Tensor& src, mllm::Tensor& dst,
                            aclDataType dt, float min_val, float max_val) {
    using namespace mllm::ascend;  // NOLINT
    aclTensor* as = makeAclTensor(src, dt);
    aclTensor* ad = makeAclTensor(dst, dt);
    aclScalar* sc_min = aclCreateScalar(&min_val, ACL_FLOAT);
    aclScalar* sc_max = aclCreateScalar(&max_val, ACL_FLOAT);
    uint64_t ws_size = 0;
    aclOpExecutor* exe = nullptr;
    aclError ret = aclnnClampGetWorkspaceSize(as, sc_min, sc_max, ad, &ws_size, &exe);
    if (ret != ACL_SUCCESS) {
      aclDestroyScalar(sc_min); aclDestroyScalar(sc_max);
      aclDestroyTensor(as);     aclDestroyTensor(ad);
      return false;
    }
    void* ws = nullptr; int ws_bid = -1;
    if (ws_size > 0) {
      getAscendMemoryManager().allocateBlock(static_cast<uint32_t>(ws_size), ws_bid);
      getAscendMemoryManager().getBlockPtr(ws_bid, ws);
    }
    ret = aclnnClamp(ws, ws_size, exe, getGlobalAtbStream());
    syncGlobalAtbStream();
    if (ws_bid != -1) getAscendMemoryManager().freeBlock(ws_bid);
    aclDestroyScalar(sc_min);
    aclDestroyScalar(sc_max);
    aclDestroyTensor(as);
    aclDestroyTensor(ad);
    return ret == ACL_SUCCESS;
  }

  // Run aclnnCast synchronously on NPU.
  // src and dst must already be allocated on kAscend.
  static bool runAclnnCast(const mllm::Tensor& src, mllm::Tensor& dst,
                           aclDataType src_dt, aclDataType dst_dt) {
    using namespace mllm::ascend;  // NOLINT
    aclTensor* as = makeAclTensor(src, src_dt);
    aclTensor* ad = makeAclTensor(dst, dst_dt);
    uint64_t ws_size = 0;
    aclOpExecutor* exe = nullptr;
    aclError ret = aclnnCastGetWorkspaceSize(as, dst_dt, ad, &ws_size, &exe);
    if (ret != ACL_SUCCESS) {
      aclDestroyTensor(as); aclDestroyTensor(ad);
      return false;
    }
    void* ws = nullptr; int ws_bid = -1;
    if (ws_size > 0) {
      getAscendMemoryManager().allocateBlock(static_cast<uint32_t>(ws_size), ws_bid);
      getAscendMemoryManager().getBlockPtr(ws_bid, ws);
    }
    ret = aclnnCast(ws, ws_size, exe, getGlobalAtbStream());
    syncGlobalAtbStream();
    if (ws_bid != -1) getAscendMemoryManager().freeBlock(ws_bid);
    aclDestroyTensor(as);
    aclDestroyTensor(ad);
    return ret == ACL_SUCCESS;
  }

  // Run ATB W8A8 Linear (PER_CHANNEL dequant) on NPU.
  //   x_int8  : [M, K]   ACL_INT8
  //   w_int8  : [N, K]   ACL_INT8  (transposeB=true, effective [K, N])
  //   bias    : [1, N]   ACL_INT32
  //   deq     : [N]      ACL_UINT64
  //   y       : [M, N]   ACL_FLOAT16 (output, pre-allocated)
  static bool runAtbW8A8Linear(const mllm::Tensor& x_int8, const mllm::Tensor& w_int8,
                               const mllm::Tensor& bias,   const mllm::Tensor& deq,
                               mllm::Tensor& y) {
    using namespace mllm::ascend;  // NOLINT
    atb::infer::LinearParam lp;
    lp.transposeA  = false;
    lp.transposeB  = true;
    lp.hasBias     = true;
    lp.outDataType = ACL_FLOAT16;
    lp.enAccum     = false;
    lp.matmulType  = atb::infer::LinearParam::MATMUL_UNDEFINED;
    lp.quantMode   = atb::infer::LinearParam::PER_CHANNEL;

    atb::Operation* op = nullptr;
    if (atb::CreateOperation(lp, &op) != atb::NO_ERROR || op == nullptr) return false;

    atb::Tensor atb_x, atb_w, atb_b, atb_d, atb_y;
    fillAtbTensor(x_int8, atb_x);
    fillAtbTensor(w_int8, atb_w);
    fillAtbTensor(bias,   atb_b);
    fillAtbTensor(deq,    atb_d);
    fillAtbTensor(y,      atb_y);

    atb::VariantPack vp;
    vp.inTensors  = {atb_x, atb_w, atb_b, atb_d};
    vp.outTensors = {atb_y};

    uint64_t ws_size = 0;
    if (op->Setup(vp, ws_size, getGlobalAtbContext()) != atb::NO_ERROR) {
      atb::DestroyOperation(op); return false;
    }
    void* ws = nullptr; int ws_bid = -1;
    if (ws_size > 0) {
      getAscendMemoryManager().allocateBlock(static_cast<uint32_t>(ws_size), ws_bid);
      getAscendMemoryManager().getBlockPtr(ws_bid, ws);
    }
    bool ok = (op->Execute(vp, reinterpret_cast<uint8_t*>(ws), ws_size, getGlobalAtbContext())
               == atb::NO_ERROR);
    syncGlobalAtbStream();
    if (ws_bid != -1) getAscendMemoryManager().freeBlock(ws_bid);
    atb::DestroyOperation(op);
    return ok;
  }
};
