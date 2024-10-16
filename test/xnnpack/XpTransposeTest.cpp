#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class TransposeModule : public Module {
    Layer linear_;

public:
    TransposeModule() {
        linear_ = Linear(2048, 4096, true, "linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];

        // B, S, H, D -> B, S, D, H
        auto out = x.transpose(SEQUENCE, DIMENSION);
        return {linear_(out)};
    }
};

TEST(XpReLUTest, ReLUModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::WARN;

    auto model = ::mllm::xnnpack::wrap2xnn<TransposeModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK] != nullptr, true);

    // B, S ,H, D
    Tensor x(1, 1, 2048, 1024, Backend::global_backends[MLLM_XNNPACK], true);
    x.setTtype(TensorType::INPUT_TENSOR);

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("Transpose + Linear 1, time={} microseconds", duration.count());

    out.printShape();
}