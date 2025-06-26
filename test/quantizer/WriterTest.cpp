//
// Created by Xiang Li on 23-11-2.
//
#include "gtest/gtest.h"
#include <unordered_map>
#include <vector> //
#include "ParamLoader.hpp"
#include "ParamWriter.hpp"
#include "QuantWriter.hpp"
#include "QuantTest.hpp"
#include "Types.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ4.hpp" // For manual quantization

namespace mllm {

// ReadTest 不需要修改，因为 ParamLoader 的读取接口保持兼容
TEST_F(QuantTest, ReadTest) {
    auto loader = ParamLoader("../bin/quant_test.mllm");
    auto tensor_name = loader.getParamNames();
    ASSERT_EQ(tensor_name.size(), 2);
    ASSERT_EQ(loader.getDataType(tensor_name[0]), DataType::MLLM_TYPE_F32);
    auto [data, size] = loader.load("weight_f0");
    ASSERT_EQ(data[0], 0.0);
    ASSERT_EQ(data[1], 0.0);
    delete[] data; // 释放 loader.load 返回的内存
}

// ** [修改] ** WriteTest 使用新的流式写入 API
TEST_F(QuantTest, WriteTest) {
    auto *loader = new ParamLoader("../bin/quant_test.mllm");
    auto *writer = new ParamWriter("../bin/quant_result.mllm");
    auto tensor_name = loader->getParamNames();
    writer->paddingIndex(tensor_name);
    ASSERT_EQ(tensor_name.size(), 2);

    // 使用 vector<char> 来管理内存，避免手动 new/delete
    std::unordered_map<string, std::vector<unsigned char>> ori_data;

    for (const auto &tensor : tensor_name) {
        auto [data, size] = loader->load(tensor);

        // 存储原始数据以供后续比较
        ori_data[tensor].resize(size);
        memcpy(ori_data[tensor].data(), data, size);

        // ** 使用新的三段式流式写入 **
        writer->beginWriteParam(tensor, DataType::MLLM_TYPE_F32);
        writer->writeChunk(data, size);
        writer->endWriteParam();

        delete[] data; // 释放 loader->load 返回的内存
    }

    writer->writeIndex();
    delete writer;
    delete loader;

    // 验证写入的文件
    auto loader2 = ParamLoader("../bin/quant_result.mllm");
    auto tensor_name2 = loader2.getParamNames();
    ASSERT_EQ(tensor_name2.size(), 2);
    ASSERT_EQ(loader2.getDataType("weight_f1"), DataType::MLLM_TYPE_F32);

    auto [data2, size2] = loader2.load("weight_f1");
    float *fdata = (float *)data2;
    unsigned char *original_raw_data = ori_data["weight_f1"].data();

    // 逐字节比较
    for (size_t i = 0; i < size2; i++) {
        ASSERT_EQ(data2[i], original_raw_data[i]);
    }
    delete[] data2;
}

// ** [修改] ** QuantTest 不再依赖 QuantWriter 内部状态，而是测试端到端的文件输出
TEST_F(QuantTest, QuantTest) {
    const std::string input_path = "../bin/quant_test.mllm";
    const std::string output_path = "../bin/quant_result.mllm";
    const std::string target_tensor_name = "weight_f1";

    // 1. 执行量化，生成输出文件
    auto *quant_writer = new QuantWriter(output_path, input_path);
    ASSERT_EQ(quant_writer->readParams(), 2);
    quant_writer->quantize(DataType::MLLM_TYPE_Q4_0, ""); // 使用新的 quantize API
    delete quant_writer;

    // 2. 加载原始的 FP32 数据，用于生成“期望”的量化结果
    auto *original_loader = new ParamLoader(input_path);
    auto [original_data_ptr, original_size] = original_loader->load(target_tensor_name);

    // 3. 在测试中手动进行量化，得到期望的结果
    uint64_t num_floats = original_size / sizeof(float);
    auto block_t = alloc_quant_block(num_floats, DataType::MLLM_TYPE_Q4_0);
    void *expected_quant_data = block_t.first;
    quantize_row_q4_0(reinterpret_cast<float *>(original_data_ptr), expected_quant_data, num_floats);

    // 4. 从 QuantWriter 生成的文件中加载实际的量化结果
    auto result_loader = ParamLoader(output_path);
    auto tensor_names_from_result = result_loader.getParamNames();
    ASSERT_EQ(tensor_names_from_result.size(), 2);
    ASSERT_EQ(result_loader.getDataType(target_tensor_name), DataType::MLLM_TYPE_Q4_0);
    auto [actual_quant_data_ptr, actual_quant_size] = result_loader.load(target_tensor_name);

    // 5. 比较期望的量化结果和实际的量化结果
    ASSERT_EQ(block_t.second, actual_quant_size); // 尺寸应该一致
    ASSERT_TRUE(compare_eq(
        reinterpret_cast<block_q4_0 *>(expected_quant_data),
        reinterpret_cast<block_q4_0 *>(actual_quant_data_ptr)));

    // 6. 清理内存
    delete[] (char *)expected_quant_data;
    delete[] original_data_ptr;
    delete[] actual_quant_data_ptr;
    delete original_loader;
}
} // namespace mllm