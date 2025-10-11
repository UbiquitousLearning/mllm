//
// Created by Xiang Li on 23-10-31.
//
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include <string>
#include <iostream>
#include "QuantWriter.hpp"
#include "Types.hpp"

const std::vector<std::string> vl_q4x4_2_q4_k_layers;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: ./quantize <input_path> <output_path> <quant_type> [other_flag]\n";
        std::cout << "  quant_type: Q4_0, Q8_0, Q4_K, Q6_K, Q8_K, Q4_0_4_4, KAI_Q4_0, etc.\n";
        std::cout << "  other_flag (optional): 'vl' or 'eager'\n";
        return -1;
    }
    auto input_path = std::string(argv[1]);
    auto output_path = std::string(argv[2]);
    auto quant_type_str = std::string(argv[3]);
    std::string other_flag = "";
    if (argc == 5) {
        other_flag = std::string(argv[4]);
        if (other_flag != "vl" && other_flag != "eager" && other_flag != "qw3") {
            std::cout << "Invalid other_flag. Use 'vl' or 'eager' or 'qw3'.\n";
            return -1;
        }
    }

    DataType quant_type_enum = MLLM_TYPE_COUNT;
    if (quant_type_str == "Q4_0")
        quant_type_enum = MLLM_TYPE_Q4_0;
    else if (quant_type_str == "Q8_0")
        quant_type_enum = MLLM_TYPE_Q8_0;
    else if (quant_type_str == "Q2_K")
        quant_type_enum = MLLM_TYPE_Q2_K;
    else if (quant_type_str == "Q3_K")
        quant_type_enum = MLLM_TYPE_Q3_K;
    else if (quant_type_str == "Q4_K")
        quant_type_enum = MLLM_TYPE_Q4_K;
    else if (quant_type_str == "Q6_K")
        quant_type_enum = MLLM_TYPE_Q6_K;
    else if (quant_type_str == "Q8_K")
        quant_type_enum = MLLM_TYPE_Q8_K;
    else if (quant_type_str == "KAI_Q4_0")
        quant_type_enum = MLLM_TYPE_KLEIDIAI_Q4_0;
    else if (quant_type_str == "Q4_0_4_4")
        quant_type_enum = MLLM_TYPE_Q4_0_4_4;
    else {
        std::cout << "Quant type " << quant_type_str << " is not supported\n";
        return -1;
    }

    mllm::QuantWriter quant_writer(output_path, input_path);
    int param_count = quant_writer.readParams();
    if (param_count <= 0) {
        std::cout << "No params to quantize\n";
        return -1;
    }
    std::cout << "Quantizing " << param_count << " params to " << quant_type_str << " with flag '" << other_flag << "'\n";
    quant_writer.quantize(quant_type_enum, other_flag);
    std::cout << "Quantization finished successfully.\n";

    return 0;
}