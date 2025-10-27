#include "Layer.hpp"
#include "Module.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include "Context.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include "XpTest.hpp"

using namespace mllm;

TEST_F(XpTest, XNNPACK) {
    Module::initBackend(MLLM_XNNPACK);
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::INFO;

    // inputs
    // B, S, H, D
    Tensor x(1, 1, 8, 8, Backend::global_backends[MLLM_XNNPACK].get(), true);

    if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
        ::mllm::xnnpack::Log::error("failed to initialize XNNPACK");
        return;
    }

    // create graph
    xnn_subgraph_t subgraph;
    xnn_create_subgraph(2, 0, &subgraph);

    // outputs
    Tensor out(1, 1, 8, 8, Backend::global_backends[MLLM_XNNPACK].get(), true);

    // define tensor
    {
        std::array<size_t, 4> dims{1, 8, 1, 8};
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
    }
    {
        std::array<size_t, 4> dims{1, 8, 1, 8};
        auto status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            dims.size(),
            dims.data(),
            nullptr,
            1,
            XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
            &out.uuid());
        if (status != xnn_status_success) {
            ::mllm::xnnpack::Log::error("xnn_define_tensor_value {} failed", "out");
            exit(-1);
        }
    }

    uint32_t slice_inputs;
    uint32_t slice_outputs;

    {
        std::array<size_t, 4> dims{1, 1, 1, 8};
        auto status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            dims.size(),
            dims.data(),
            nullptr,
            XNN_INVALID_VALUE_ID,
            0,
            &slice_inputs);
    }
    {
        std::array<size_t, 4> dims{1, 1, 1, 8};
        auto status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            dims.size(),
            dims.data(),
            nullptr,
            XNN_INVALID_VALUE_ID,
            0,
            &slice_outputs);
    }

    // slice a item and copy to output
    {
        std::array<size_t, 4> offset{0, 0, 0, 0};
        std::array<size_t, 4> new_shape{1, 1, 1, 8};
        auto status = xnn_define_static_slice(
            subgraph,
            4,
            offset.data(),
            new_shape.data(),
            x.uuid(),
            slice_inputs,
            0);
        status = xnn_define_static_slice(
            subgraph,
            4,
            offset.data(),
            new_shape.data(),
            out.uuid(),
            slice_outputs,
            0);
    }

    // copy
    // {
    //     auto status = xnn_define_copy(
    //         subgraph,
    //         slice_inputs,
    //         slice_outputs,
    //         0);
    // }

    for (int i = 0; i < 8; ++i) {
        *(x.hostPtr<float>() + i) = (float)i;
    }

    xnn_runtime_t rt;
    std::vector<xnn_external_value> external_values;
    external_values.push_back({0, x.rawHostPtr()});
    external_values.push_back({1, out.rawHostPtr()});

    auto threadpool = pthreadpool_create(4);
    xnn_create_runtime_v4(subgraph, nullptr, nullptr, threadpool, 0, &rt);
    xnn_reshape_runtime(rt);
    xnn_setup_runtime_v2(rt, external_values.size(), external_values.data());
    xnn_invoke_runtime(rt);

    out.printData<float>();
}