#include "Module.hpp"
#include "Types.hpp"
#include "Layer.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

class LinearModule : public Module {
    Layer linear_1;
    Layer linear_2;

public:
    explicit LinearModule(const std::string &base_name) {
        linear_1 = Linear(8, 16, true, base_name + "linear_1");
        linear_2 = Linear(16, 32, true, base_name + "linear_2");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        return {linear_2(linear_1(inputs[0]))};
    }
};

class EmbeddingModule : public Module {
    Layer embedding;
    xnnpack::XpWrapperModule linear_module;

public:
    EmbeddingModule() {
        embedding = Embedding(100, 8, "embedding");
        linear_module = xnnpack::wrap2xnn<LinearModule>(1, 1, "linear_");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto o = embedding(inputs[0]);
        o = linear_module({o})[0];
        return {o};
    }
};

TEST_F(XpTest, CPUAndXnnMixed) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::INFO;

    auto model = EmbeddingModule();
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);
    model.to(BackendType::MLLM_XNNPACK);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    Tensor x(1, 1, 10, 1, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    for (int i = 0; i < 10; ++i) {
        *(x.hostPtr<float>() + i) = (float)i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("EmbeddingModuleTest 1, time={} microseconds", duration.count());

    out.printData<float>();
    out.printShape();
}
