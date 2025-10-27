#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

class ReLUModule : public Module {
    Layer relu_;

public:
    ReLUModule() {
        relu_ = ReLU("activation_relu");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = relu_(x);
        return {out};
    }
};

TEST_F(XpTest, ReLUModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<ReLUModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    Tensor x(1, 1, 1024, 1024, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    for (int i = 0; i < 1024 * 1024; ++i) {
        *(x.hostPtr<float>() + i) = -1.f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("ReLU 1, time={} microseconds", duration.count());

    for (int i = 0; i < 1024 * 1024; ++i) {
        EXPECT_EQ(*(out.hostPtr<float>() + i), 0);
    }

    out.printShape();
}