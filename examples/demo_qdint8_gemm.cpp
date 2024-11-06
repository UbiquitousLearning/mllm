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
    Layer linear;

public:
    explicit MatmulModule(int s) {
        linear = Linear(s, s, false, ".linear");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        return {linear(inputs[0])};
    }
};

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<int>("seq-len", 's', "sequence length", true, 64);
    cmdParser.parse_check(argc, argv);

    size_t s = cmdParser.get<int>("seq-len");

    xnn_initialize(nullptr);

    Backend::global_backends.emplace(MLLM_XNNPACK, GetBackendCreator(MLLM_XNNPACK)->create({}));

    Tensor inputs1(1, 1, (int32_t)s, (int32_t)s, Backend::global_backends[MLLM_XNNPACK], true);
    Tensor inputs2(1, 1, (int32_t)s, (int32_t)s, Backend::global_backends[MLLM_XNNPACK], true);
    Tensor outputs1(1, 1, (int32_t)s, (int32_t)s, Backend::global_backends[MLLM_XNNPACK], true);

    xnn_subgraph_t subgraph = nullptr;
    xnn_create_subgraph(3, /*flags=*/0, &subgraph);

    uint32_t input1_f32_id = XNN_INVALID_VALUE_ID;
    std::array<size_t, 4> input1_dims{1, 1, s, s};
    xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        input1_dims.size(),
        input1_dims.data(),
        nullptr,
        /*external_id=*/0,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &input1_f32_id);

    uint32_t input1_id = XNN_INVALID_VALUE_ID;
    xnn_define_dynamically_quantized_tensor_value(
        subgraph,
        xnn_datatype_qdint8,
        input1_dims.size(),
        1,
        input1_dims.data(),
        XNN_INVALID_VALUE_ID,
        0,
        &input1_id);

    uint32_t input2_id = XNN_INVALID_VALUE_ID;
    std::array<size_t, 4> input2_dims{1, 1, s, s};
    std::vector<float> channelwise_scale(s, 1.f);
    xnn_define_channelwise_quantized_tensor_value(
        subgraph,
        xnn_datatype_qcint8,
        channelwise_scale.data(),
        input2_dims.size(),
        /*channel_dim=*/1,
        input2_dims.data(),
        inputs2.rawHostPtr(),
        XNN_INVALID_VALUE_ID,
        /*flags=*/0,
        &input2_id);

    uint32_t output1_id = XNN_INVALID_VALUE_ID;
    std::array<size_t, 4> output1_dims{1, 1, s, s};
    xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        output1_dims.size(),
        output1_dims.data(),
        nullptr,
        1,
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
        &output1_id);

    xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr, input1_f32_id,
                     input1_id, /*flags=*/0);
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id,
                                     output1_id, /*flags=*/0);
    xnn_runtime_t rt;
    std::vector<xnn_external_value> exts;
    exts.push_back({0, inputs1.rawHostPtr()});
    exts.push_back({1, outputs1.rawHostPtr()});
    auto threadpool = pthreadpool_create(4);
    xnn_create_runtime_v4(subgraph, nullptr, nullptr, threadpool, 0, &rt);
    xnn_reshape_runtime(rt);
    xnn_setup_runtime_v2(rt, exts.size(), exts.data());

    mllm::xnnpack::Log::warn("Start benchmark");
    auto start = std::chrono::high_resolution_clock::now();
    xnn_invoke_runtime(rt);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("xnn run (QD8 * QC8W -> FP32) shape={}x{}, time={} microseconds", s, s, duration.count());

    inputs1.free();
    inputs2.free();
    outputs1.free();

    auto model = MatmulModule((int32_t)s);
    model.setNoLoadWeightsDtype(MLLM_TYPE_Q4_0_4_4);
    model.to(MLLM_CPU);

    Tensor inputs1_q4k;
    inputs1_q4k.setDtype(MLLM_TYPE_F32);
    inputs1_q4k.reshape(1, 1, int32_t(s), int32_t(s));
    inputs1_q4k.setBackend(Backend::global_backends[MLLM_CPU]);
    inputs1_q4k.alloc();
    inputs1_q4k.setTtype(TensorType::INPUT_TENSOR);
    // inputs1_q4k.setName("inputs-0");

    start = std::chrono::high_resolution_clock::now();
    auto o = model({inputs1_q4k});
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("mllm run (Q4K * Q4K -> FP32) shape={}x{}, time={} microseconds", s, s, duration.count());
}
