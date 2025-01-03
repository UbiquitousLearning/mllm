//
// Created by Xiang Li on 23-10-31.
//
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "backends/cpu/quantize/QuantizeQ4.hpp"
#include "backends/cpu/quantize/QuantizeQ8.hpp"
#include <string>
#include "QuantWriter.hpp"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: ./quantize <input_path> <output_path> <quant_type>\n";
        return -1;
    }
    auto input_path = std::string(argv[1]);
    auto output_path = std::string(argv[2]);
    auto quant_type = std::string(argv[3]);
    // std::string input_path = "../models/minicpm-moe-8x2b-fp32.mllm";
    // std::string output_path = "../models/minicpm-moe-8x2b-q4_0_4_4.mllm";
    // std::string quant_type = "Q4_0_4_4";
    mllm::QuantWriter quant_writer(output_path, input_path);
    int param_count = quant_writer.readParams();
    if (param_count <= 0) {
        std::cout << "No params to quantize\n";
        return -1;
    }
    std::cout << "Quantize " << param_count << " params to " << quant_type << "\n";
    if (quant_type == "Q4_0") {
        quant_writer.quantParams(MLLM_TYPE_Q4_0);
    } else if (quant_type == "Q8_0") {
        quant_writer.quantParams(MLLM_TYPE_Q8_0);
    } else if (quant_type == "Q4_K") {
        quant_writer.quantParams(MLLM_TYPE_Q4_K);
    } else if (quant_type == "Q6_K") {
        quant_writer.quantParams(MLLM_TYPE_Q6_K);
    } else if (quant_type == "Q8_K") {
        quant_writer.quantParams(MLLM_TYPE_Q8_K);
    } else if (quant_type == "Q4_0_4_4") {
        quant_writer.quantParams_q4_(MLLM_TYPE_Q4_0_4_4);
    } else {
        std::cout << "Quant type " << quant_type << " is not supported\n";
        return -1;
    }
    return 0;
}