//
// Created by Xiang Li on 23-11-2.
//
#include "gtest/gtest.h"
#include <unordered_map>
#include "ParamLoader.hpp"
#include "ParamWriter.hpp"
#include "QuantWriter.hpp"
#include "QuantTest.hpp"
#include "Types.hpp"
namespace mllm {
TEST_F(QuantTest, ReadTest) {
    auto loader = ParamLoader("../bin/quant_test.mllm");
    auto tensor_name = loader.getParamNames();
    ASSERT_EQ(tensor_name.size(), 2);
    ASSERT_EQ(loader.getDataType(tensor_name[0]), DataType::MLLM_TYPE_F32);
    auto [data, size] = loader.load("weight_f0");
    ASSERT_EQ(data[0], 0.0);
    ASSERT_EQ(data[1], 0.0);
}
TEST_F(QuantTest, WriteTest) {
    auto *loader = new ParamLoader("../bin/quant_test.mllm");
    auto *writer = new ParamWriter("../bin/quant_result.mllm");
    auto tensor_name = loader->getParamNames();
    writer->paddingIndex(tensor_name);
    ASSERT_EQ(tensor_name.size(), 2);
    std::unordered_map<string, unsigned char *> ori_data;
    for (auto tensor : tensor_name) {
        auto [data, size] = loader->load(tensor);
        ori_data[tensor] = data;
        writer->writeParam(tensor, DataType::MLLM_TYPE_F32, data, size);
    }
    writer->writeIndex();
    delete writer;
    delete loader;
    auto loader2 = ParamLoader("../bin/quant_result.mllm");
    auto tensor_name2 = loader2.getParamNames();
    ASSERT_EQ(tensor_name2.size(), 2);
    ASSERT_EQ(loader2.getDataType(tensor_name2[0]), DataType::MLLM_TYPE_F32);
    auto [data, size] = loader2.load("weight_f1");
    float *fdata = (float *)data;
    for (int i = 0; i < size / sizeof(float); i++) {
        ASSERT_EQ(fdata[i], ori_data["weight_f1"][i]);
    }
}
TEST_F(QuantTest, QuantTest) {
    auto *quant = new QuantWriter("../bin/quant_result.mllm", "../bin/quant_test.mllm");
    ASSERT_EQ(quant->readParams(), 2);
    quant->quantParams(DataType::MLLM_TYPE_Q4_0);
    ASSERT_EQ(quant->data_.size(), 2);
    // delete quant;
    auto loader = ParamLoader("../bin/quant_result.mllm");
    auto tensor_name = loader.getParamNames();
    ASSERT_EQ(tensor_name.size(), 2);
    ASSERT_EQ(loader.getDataType(tensor_name[0]), DataType::MLLM_TYPE_Q4_0);
    auto [data, size] = loader.load("weight_f1");
    auto *ori_data = quant->data_["weight_f1"];
    ASSERT_TRUE(compare_eq(reinterpret_cast<block_q4_0 *>(ori_data), reinterpret_cast<block_q4_0 *>(data)));
}
} // namespace mllm
