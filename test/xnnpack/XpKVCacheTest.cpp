#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class KVCacheModule : public Module {
    Layer kvcache_;
    Layer linear_in_;

public:
    KVCacheModule() {
        kvcache_ = XP_KVCache(3, 10, "kvcache");
        linear_in_ = Linear(8, 8, true, "linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];
        auto out = linear_in_(x);
        out = kvcache_(out);
        return {out};
    }
};

TEST(XpKVCAcheTest, KVCacheModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::ERROR;

    auto model = ::mllm::xnnpack::wrap2xnn<KVCacheModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK] != nullptr, true);

    Tensor x(1, 1, 1, 8, Backend::global_backends[MLLM_XNNPACK], true);
    x.setTtype(TensorType::INPUT_TENSOR);

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    out.printData<float>();
    out = model({x})[0];
    out.printData<float>();
    out = model({x})[0];
    out.printData<float>();
    out = model({x})[0];
    out.printData<float>();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("KVCache 1, time={} microseconds", duration.count());
    out.printShape();
}