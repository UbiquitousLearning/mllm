#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class AddModule : public Module {
    Layer DirectInput;
    Layer DispatchAll;

public:
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x1 = inputs[0];
        auto x2 = inputs[1];

        auto out = x1 + x2;
        return {out};
    }
};

TEST(XpExternalTensorTest, AddModule) {
    auto model = AddModule();
    model.to(MLLM_XNNPACK);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK] != nullptr, true);

    Tensor x1(1, 1, 4, 4, Backend::global_backends[MLLM_XNNPACK], true);
    Tensor x2(1, 1, 4, 4, Backend::global_backends[MLLM_XNNPACK], true);

    x1.printData<float>();
    x2.printData<float>();

    return;

    auto out = model({x1, x2})[0];

    out.printData<float>();
}