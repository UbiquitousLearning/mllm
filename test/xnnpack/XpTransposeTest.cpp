#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "Context.hpp"
#include "backends/xnnpack/XpWrapper.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "xnnpack.h"
#include <gtest/gtest.h>
#include <limits>
#include "XpTest.hpp"

using namespace mllm;

class TransposeModule : public Module {
public:
    explicit TransposeModule() = default;

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = inputs[0];

        // B, S, H, D -> B, H, S, D
        auto out = x.transpose(SEQUENCE, HEAD);
        return {out};
    }
};

TEST_F(XpTest, TransposeModule) {
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::WARN;

    auto model = ::mllm::xnnpack::wrap2xnn<TransposeModule>(1, 1);
    model.setNoLoadWeightsDtype(DataType::MLLM_TYPE_F32);

    EXPECT_EQ(Backend::global_backends[MLLM_XNNPACK].get() != nullptr, true);
    if (XnnpackBackend::enable_legacy_wrapper == false) {
        Log::warn("This test method is dropped. But tested ok in legacy wrapper mode");
        return;
    }

    // B, S ,H, D
    Tensor x(1, 6, 8, 1, Backend::global_backends[MLLM_XNNPACK].get(), true);
    x.setTtype(TensorType::INPUT_TENSOR);

    float cnt = 0.f;
    for (int i = 0; i < x.sequence(); ++i) {
        for (int j = 0; j < x.head(); ++j) {
            x.setDataAt(0, j, i, 0, cnt++);
        }
    }

    x.printData<float>();

    auto start = std::chrono::high_resolution_clock::now();
    auto out = model({x})[0];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mllm::xnnpack::Log::warn("Transpose + Linear 1, time={} microseconds", duration.count());

    out.printData<float>();
    out.printShape();
}

TEST(TransposeTest, RawXnnImpl) {
    Module::initBackend(MLLM_XNNPACK);
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::INFO;

    // B, S, H, D
    Tensor x(1, 1, 2048, 1024, Backend::global_backends[MLLM_XNNPACK].get(), true);

    if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
        ::mllm::xnnpack::Log::error("failed to initialize XNNPACK");
        return;
    }

    // create graph
    xnn_subgraph_t subgraph;
    xnn_create_subgraph(2, 0, &subgraph);

    // create X.
    std::array<size_t, 4> dims{1, 2048, 1, 1024};
    auto status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        dims.size(),
        dims.data(),
        nullptr,
        0,
        XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &x.uuid());
    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_tensor_value {} failed", "x");
        exit(-1);
    }

    // create temporary tensor.
    uint32_t v1;
    std::array<size_t, 4> dims_temp{1, 1024, 1, 2048};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        dims_temp.size(),
        dims_temp.data(),
        nullptr,
        XNN_INVALID_VALUE_ID,
        0,
        &v1);
    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_tensor_value {} failed", "temp");
        exit(-1);
    }

    // create Transpose.
    std::array<size_t, 4> perm{3, 0, 1, 2};
    status = xnn_define_static_transpose(
        subgraph,
        4,
        perm.data(),
        x.uuid(),
        v1,
        0);

    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_static_transpose failed");
        exit(-1);
    }

    // create outexternal output.
    Tensor out(1, 1, 2048, 4096, Backend::global_backends[MLLM_XNNPACK].get(), true);
    std::array<size_t, 4> dims_out{1, 2048, 1, 4096};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        dims_out.size(),
        dims_out.data(),
        nullptr,
        1,
        XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &out.uuid());
    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_tensor_value {} failed", "out");
        exit(-1);
    }

    // create linear
    Tensor weight(1, 1, 1024, 4096, Backend::global_backends[MLLM_XNNPACK].get(), true);
    Tensor bias(1, 1, 1, 4096, Backend::global_backends[MLLM_XNNPACK].get(), true);
    std::array<size_t, 2> dim_weight{1024, 4096};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        dim_weight.size(),
        dim_weight.data(),
        weight.rawHostPtr(),
        XNN_INVALID_VALUE_ID,
        0,
        &weight.uuid());
    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_tensor_value {} failed", "weight");
        exit(-1);
    }
    std::array<size_t, 2> dim_bias{4096};
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        dim_bias.size(),
        dim_bias.data(),
        bias.rawHostPtr(),
        XNN_INVALID_VALUE_ID,
        0,
        &bias.uuid());
    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_tensor_value {} failed", "bias");
        exit(-1);
    }
    status = xnn_define_fully_connected(
        subgraph,
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max(),
        v1,
        weight.uuid(),
        bias.uuid(),
        out.uuid(),
        0);
    if (status != xnn_status_success) {
        ::mllm::xnnpack::Log::error("xnn_define_fully_connected failed");
        exit(-1);
    }

    // launch a runtime
    xnn_runtime_t rt;
    std::vector<xnn_external_value> external_values;
    external_values.push_back({0, x.rawHostPtr()});
    external_values.push_back({1, out.rawHostPtr()});

    auto threadpool = pthreadpool_create(4);
    xnn_create_runtime_v4(subgraph, nullptr, nullptr, threadpool, 0, &rt);
    xnn_reshape_runtime(rt);
    xnn_setup_runtime_v2(rt, external_values.size(), external_values.data());
    xnn_invoke_runtime(rt);
}
