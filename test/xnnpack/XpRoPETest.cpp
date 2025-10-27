#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

class RoPEModule : public Module {
    Layer rope_;
    Layer linear_;

public:
    RoPEModule() {
        rope_ = RoPE(RoPEType::HFHUBROPE, 1e5, 1024, "rope");
        linear_ = Linear(1024, 1024, true, "linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = linear_(x) + rope_(x);
        return {out};
    }
};

TEST_F(XpTest, RoPEModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::WARN;

    auto model = ::mllm::xnnpack::wrap2xnn<RoPEModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    // rope accpect b, s, h, d.
    Tensor x(1, 1, 256, 1024, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    {
        auto start = std::chrono::high_resolution_clock::now();
        auto out1 = model({x})[0];
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        mllm::xnnpack::Log::warn("RoPEModule 1, time={} microseconds", duration.count());
    }
    // run againe to see if xnnpack backend is reset correctly.
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto out2 = model({x})[0];
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        mllm::xnnpack::Log::warn("RoPEModule 2, time={} microseconds", duration.count());
    }
}