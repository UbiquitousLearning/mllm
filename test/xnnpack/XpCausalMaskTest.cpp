#include "Module.hpp"
#include "Types.hpp"
#include "Layer.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include <gtest/gtest.h>
#include "XpTest.hpp"
#include "xnnpack/XnnpackBackend.hpp"

using namespace mllm;

class CausalMaskModule : public Module {
    Layer causal_mask_;

public:
    CausalMaskModule() {
        causal_mask_ = Causalmask("mask");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = causal_mask_(x);
        return {out};
    }
};

TEST_F(XpTest, CausalMaskModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::INFO;

    auto model = mllm::xnnpack::wrap2xnn<CausalMaskModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    Tensor x(1, 1, 8, 8, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("Causal Mask 1, time={} microseconds", duration.count());

    out.printData<float>();
    out.printShape();
}
