//
// Created by Xiang Li on 23-10-31.
//
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include <string>
#include "QuantWriter.hpp"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: ./quantize <input_path> <output_path> <quant_type>\n";
        return -1;
    }
    auto input_path = std::string(argv[1]);
    auto output_path = std::string(argv[2]);
    auto quant_type = std::string(argv[3]);
    auto other_flag = std::string(argv[4]);
    // std::string input_path = "../models/qwen-2.5-1.5b-instruct-fp32.mllm";
    // std::string output_path = "../models/qwen-2.5-1.5b-instruct-kai_q4_0.mllm";
    // std::string input_path = "../models/qwen-2-vl-2b-instruct-fp32.mllm";
    // std::string output_path = "../models/qwen-2-vl-2b-instruct-kai_q4_0.mllm";
    // std::string quant_type = "KAI_Q4_0";
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
    } else if (quant_type == "Q2_K") {
        quant_writer.quantParams(MLLM_TYPE_Q2_K);
    } else if (quant_type == "Q3_K") {
        quant_writer.quantParams(MLLM_TYPE_Q3_K);
    } else if (quant_type == "Q4_K") {
        quant_writer.quantParams(MLLM_TYPE_Q4_K);
    } else if (quant_type == "Q6_K") {
        quant_writer.quantParams(MLLM_TYPE_Q6_K);
    } else if (quant_type == "Q8_K") {
        quant_writer.quantParams(MLLM_TYPE_Q8_K);
    } else if (quant_type == "KAI_Q4_0") {
        // quant_writer.quantParams(MLLM_TYPE_KLEIDIAI_Q4_0);
        if (other_flag == "eager") {
            vl_q4x4_2_q4_k_layers.push_back("v_proj");
        }
        quant_writer.quantParams_kai_vl(MLLM_TYPE_KLEIDIAI_Q4_0);
    } else if (quant_type == "Q4_0_4_4") {
        if (other_flag == "vl") {
            quant_writer.quantParams_q4_vl(MLLM_TYPE_Q4_0_4_4);
        } else {
            quant_writer.quantParams_q4_(MLLM_TYPE_Q4_0_4_4);
        }
    } else {
        std::cout << "Quant type " << quant_type << " is not supported\n";
        return -1;
    }
    return 0;
}