#include "Types.hpp"
#include "cmdline.h"
#include "xnnpack.h"
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
    explicit MatmulModule(int s) {
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        return {Tensor::mm(inputs[0], inputs[1])};
    }
};

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<int>("seq-len", 's', "sequence length", true, 64);
    cmdParser.parse_check(argc, argv);

    size_t s = cmdParser.get<int>("seq-len");

    auto model = MatmulModule((int32_t)s);
    model.to(MLLM_CPU);

    Tensor inputs0(1, 1, (int32_t)s, (int32_t)s, Backend::global_backends[MLLM_CPU], true);
    Tensor inputs1(1, 1, (int32_t)s, (int32_t)s, Backend::global_backends[MLLM_CPU], true);
    inputs0.setTtype(TensorType::INPUT_TENSOR);
    inputs1.setTtype(TensorType::INPUT_TENSOR);

    // warmup
    auto o = model({inputs0, inputs1});

    auto start = std::chrono::high_resolution_clock::now();
    o = model({inputs0, inputs1});
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("mllm run (FP32 * FP32 -> FP32) shape={}x{}, {}x{}, time={} microseconds", s, s, s, s, duration.count());
}
