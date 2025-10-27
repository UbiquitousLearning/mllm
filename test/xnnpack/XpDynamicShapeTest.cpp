/**
 * @file XpDynamicShapeTest.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-11-01
 *
 * @copyright Copyright (c) 2024
 *
 * ref: https://github.com/google/XNNPACK/blob/8c3d6e5888968e999fdf4e0c449957e57219691c/test/scaled-dot-product-attention.cc#L666
 */
#include <gtest/gtest.h>
#include <limits>
#include "XpTest.hpp"
#include "xnnpack.h"
#include "Tensor.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "Context.hpp"

using namespace mllm;

TEST_F(XpTest, XpDyanmicShape) {
    ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

    xnn_subgraph_t subgraph = nullptr;
    xnn_create_subgraph(2, 0, &subgraph);

    // define bias and weight
    std::array<size_t, 2> weight_shape{1024, 2048};
    std::array<size_t, 1> bias_shape{1024};
    uint32_t weight_id;
    uint32_t bias_id;
    auto status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        2,
        weight_shape.data(),
        nullptr,
        XNN_INVALID_VALUE_ID,
        0,
        &weight_id);
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        1,
        bias_shape.data(),
        nullptr,
        XNN_INVALID_VALUE_ID,
        0,
        &bias_id);

    // define inputs 1 [B=1, S=16, H=1, D=2048].
    mllm::xnnpack::Log::log_level = mllm::xnnpack::Log::INFO;
    std::array<size_t, 4> input_1_shape{1, 16, 1, 2048};
    std::array<size_t, 4> output_1_shape{1, 16, 1, 1024};
    Tensor inputs_1(1, 1, 16, 2048, Backend::global_backends[MLLM_XNNPACK].get(), true);
    Tensor outputs_1(1, 1, 16, 1024, Backend::global_backends[MLLM_XNNPACK].get(), true);

    // define inputs 2 [B=1, S=32, H=1, D=2048].
    std::array<size_t, 4> input_2_shape{1, 32, 1, 2048};
    std::array<size_t, 4> output_2_shape{1, 32, 1, 1024};
    Tensor inputs_2(1, 1, 32, 2048, Backend::global_backends[MLLM_XNNPACK].get(), true);
    Tensor outputs_2(1, 1, 32, 1024, Backend::global_backends[MLLM_XNNPACK].get(), true);

    std::vector<xnn_external_value> exts;

    // Call Subgraph API
    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        4,
        input_1_shape.data(),
        nullptr,
        0,
        XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &inputs_1.uuid());

    status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        4,
        output_1_shape.data(),
        nullptr,
        1,
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
        &outputs_1.uuid());

    status = xnn_define_fully_connected(
        subgraph,
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max(),
        inputs_1.uuid(),
        weight_id,
        bias_id,
        outputs_1.uuid(),
        0);

    xnn_runtime_t rt;

    // run once
    exts.push_back({0, inputs_1.rawHostPtr()});
    exts.push_back({1, outputs_1.rawHostPtr()});
    auto threadpool = pthreadpool_create(4);
    xnn_create_runtime_v4(subgraph, nullptr, nullptr, threadpool, 0, &rt);
    xnn_reshape_runtime(rt);
    xnn_setup_runtime_v2(rt, exts.size(), exts.data());
    xnn_invoke_runtime(rt);

    // run again
    exts.clear();
    exts.push_back({0, inputs_2.rawHostPtr()});
    exts.push_back({1, outputs_2.rawHostPtr()});
    xnn_reshape_external_value(rt, 0, input_2_shape.size(), input_2_shape.data());
    xnn_reshape_external_value(rt, 1, output_2_shape.size(), output_2_shape.data());
    xnn_setup_runtime_v2(rt, exts.size(), exts.data());
    xnn_invoke_runtime(rt);
}