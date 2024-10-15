#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class RoPEModule : public Module {
    Layer rope_;

public:
    RoPEModule() {
        rope_ = RoPE(RoPEType::HFHUBROPE, 1e5, 1024, "linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = rope_(x);
        return {out};
    }
};

TEST(XpRoPETest, RoPEModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<RoPEModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK] != nullptr, true);

    // rope accpect b, s, h, d.
    Tensor x(1, 256, 1, 1024, Backend::global_backends[MLLM_XNNPACK], true);
    x.setTtype(TensorType::INPUT_TENSOR);

    auto out1 = model({x})[0];

    // run againe to see if xnnpack backend is reset correctly.
    auto out2 = model({x})[0];

    out1.printShape();
    out2.printShape();
}