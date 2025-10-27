#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

class LinearModule : public Module {
    Layer linear_1_;
    Layer linear_2_;
    Layer linear_3_;

public:
    LinearModule() {
        linear_1_ = Linear(1024, 2048, true, "linear_1");
        linear_2_ = Linear(2048, 2048, true, "linear_2");
        linear_3_ = Linear(2048, 2048, true, "linear_3");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        x = linear_1_(x);
        x = linear_2_(x);
        auto out = linear_3_(x);
        return {out};
    }
};

TEST_F(XpTest, LinearModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<LinearModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    Tensor x(1, 1, 256, 1024, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    for (int i = 0; i < 256 * 1024; ++i) {
        *(x.hostPtr<float>() + i) = 1024.f;
    }

    auto out = model({x})[0];

    for (int i = 0; i < 256 * 2048; ++i) {
        EXPECT_EQ(*(out.hostPtr<float>() + i) < 1e-18, true);
    }

    out = model({x})[0];

    for (int i = 0; i < 256 * 2048; ++i) {
        EXPECT_EQ(*(out.hostPtr<float>() + i) < 1e-18, true);
    }

    out = model({x})[0];

    for (int i = 0; i < 256 * 2048; ++i) {
        EXPECT_EQ(*(out.hostPtr<float>() + i) < 1e-18, true);
    }

    out = model({x})[0];

    for (int i = 0; i < 256 * 2048; ++i) {
        EXPECT_EQ(*(out.hostPtr<float>() + i) < 1e-18, true);
    }

    out.printShape();
}