#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

class ReLUModule : public Module {
    Layer softmax_;

public:
    ReLUModule() {
        softmax_ = Softmax(DIMENSION, false, "softmax");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = softmax_(x);
        return {out};
    }
};

TEST_F(XpTest, SoftmaxModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<ReLUModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    // B, S, H, D
    Tensor x(1, 1, 1, 8, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("ReLU 1, time={} microseconds", duration.count());

    std::array<float, 8> gt{
        0.0005766,
        0.0015674,
        0.0042606,
        0.0115816,
        0.0314820,
        0.0855769,
        0.2326222,
        0.6323327,
    };
}