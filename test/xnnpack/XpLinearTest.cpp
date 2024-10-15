#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class LinearModule : public Module {
    Layer linear;

public:
    LinearModule() {
        linear = Linear(1024, 2048, true, "linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = linear(x);
        return {out};
    }
};

TEST(XpLinearTest, LinearModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<LinearModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK] != nullptr, true);

    Tensor x(1, 1, 256, 1024, Backend::global_backends[MLLM_XNNPACK], true);
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

    out.printShape();
}