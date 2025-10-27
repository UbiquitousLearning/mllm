#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>
#include "XpTest.hpp"
#include "Context.hpp"

using namespace mllm;

class KVCacheModule : public Module {
    Layer kvcache_;
    Layer linear_in_;
    Layer linear_out_;

public:
    KVCacheModule() {
        kvcache_ = XP_KVCache(1, 10, "kvcache");
        // linear_in_ = Linear(8, 8, true, "linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        // auto out = linear_in_(x);
        auto out = kvcache_(x);
        return {out};
    }
};

TEST_F(XpTest, KVCacheModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<KVCacheModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    Tensor x(1, 1, 1, 8, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    out.printData<float>();

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i * 2;
    }
    out = model({x})[0];
    out.printData<float>();

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i * 3;
    }
    out = model({x})[0];
    out.printData<float>();

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i * 4;
    }
    out = model({x})[0];
    out.printData<float>();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("KVCache 1, time={} microseconds", duration.count());
    out.printShape();
}