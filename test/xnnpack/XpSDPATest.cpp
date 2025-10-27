#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

class SDPAModule : public Module {
    Layer sdpa_;

public:
    SDPAModule() {
        sdpa_ = ScaledDotProductAttention("sdpa");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto Q = inputs[0];
        auto K = inputs[1];
        auto V = inputs[2];
        auto out = sdpa_(Q, K, V);
        return {out};
    }
};

TEST_F(XpTest, SDPAModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<SDPAModule>(3, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    // B, H, S, D
    // NOTE:
    // The SDPA need B, H, S, D data layout format.
    int B = 1;
    int S = 3;
    int H = 1;
    int D = 8;
    Tensor Q(B, S, H, D, Backend::global_backends[MLLM_XNNPACK].get(), true);
    Q.setTtype(TensorType::INPUT_TENSOR);
    Tensor K(B, S, H, D, Backend::global_backends[MLLM_XNNPACK].get(), true);
    K.setTtype(TensorType::INPUT_TENSOR);
    Tensor V(B, S, H, D, Backend::global_backends[MLLM_XNNPACK].get(), true);
    V.setTtype(TensorType::INPUT_TENSOR);

    // set data
    // B=1, H=1, S=3, D=8
    // The shape() will return [1, 1, 3, 8]
    for (int i = 0; i < S; ++i) {
        auto p_q = Q.hostPtr<float>() + i * D;
        auto p_k = K.hostPtr<float>() + i * D;
        auto p_v = V.hostPtr<float>() + i * D;
        for (int j = 0; j < D; ++j) {
            *(p_q + j) = (float)j;
            *(p_k + j) = (float)j;
            *(p_v + j) = (float)j;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({Q, K, V})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("SPDA 1, time={} microseconds", duration.count());

    for (int i = 0; i < S; ++i) {
        auto p_o = out.hostPtr<float>() + i * D;
        for (int j = 0; j < D; ++j) {
            EXPECT_EQ(std::abs(*(p_o + j) - (float)j) < 1e-6, true);
        }
    }

    out.printShape();
}