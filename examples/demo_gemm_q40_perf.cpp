#include "Types.hpp"
#include "cmdline.h"
#include <vector>
#include <array>
#include "Tensor.hpp"
#include "Backend.hpp"
#include "Module.hpp"
#include "Layer.hpp"
#include <chrono>
#include "backends/xnnpack/Utils/Logger.hpp"

using namespace mllm;

class MatmulModule final : public Module {
public:
    explicit MatmulModule() {
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        return {Tensor::mm(inputs[0], inputs[1])};
    }
};

int main(int argc, char **argv) {
    auto model = MatmulModule();
    model.to(MLLM_CPU);

    Tensor inputs0(1, 1, 196, 768, Backend::global_backends[MLLM_CPU], false);
    Tensor inputs1(1, 1, 768, 768, Backend::global_backends[MLLM_CPU], false);
    inputs0.setDtype(MLLM_TYPE_F32);
    inputs1.setDtype(MLLM_TYPE_Q4_0);
    inputs0.setTtype(TensorType::INPUT_TENSOR);
    inputs1.setTtype(TensorType::INPUT_TENSOR);
    inputs0.alloc();
    inputs1.alloc();

    // warmup
    auto o = model({inputs0, inputs1});

    auto start = std::chrono::high_resolution_clock::now();
    o = model({inputs0, inputs1});
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << duration.count() << std::endl;
}
