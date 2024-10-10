#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include <gtest/gtest.h>

using namespace mllm;

class AddModule : public Module {
    Layer DirectInput1;
    Layer DirectInput2;
    Layer DirectOutput;
    Layer DispatchAll;

public:
    AddModule() {
        DirectInput1 = Direct(Direct::ExternalInput, "directinput1");
        DirectInput2 = Direct(Direct::ExternalInput, "directinput2");
        DirectOutput = Direct(Direct::ExternalOutput, "directoutput");
        DispatchAll = Dispatch("dispatch");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x1 = DirectInput1(inputs[0]);
        auto x2 = DirectInput2(inputs[1]);

        auto out = x1 + x2;
        out = DirectOutput(out);
        DispatchAll(out);
        return {out};
    }
};

TEST(XpExternalTensorTest, AddModule) {
    auto model = AddModule();
    model.to(MLLM_XNNPACK);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK] != nullptr, true);

    Tensor x1(1, 1, 4, 4, Backend::global_backends[MLLM_XNNPACK], true);
    Tensor x2(1, 1, 4, 4, Backend::global_backends[MLLM_XNNPACK], true);
    x1.setTtype(TensorType::INPUT_TENSOR);
    x2.setTtype(TensorType::INPUT_TENSOR);

    float cnt = 0.f;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            x1.setDataAt<float>(0, 0, i, j, cnt++);
        }
    }

    x1.printData<float>();
    x2.printData<float>();

    auto out = model({x1, x2})[0];

    out.xnn().printData<float>();
}