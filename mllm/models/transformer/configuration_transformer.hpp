//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef CONFIGURATION_TRANSFORMER_HPP
#define CONFIGURATION_TRANSFORMER_HPP

#include "Layer.hpp"
#include "Types.hpp"

using namespace mllm;
using namespace std;

class TransformerNameConfig {
public:
    string _q_proj_name;
    string _k_proj_name;
    string _v_proj_name;
    string _o_proj_name;
    string _up_proj_name;
    string _down_proj_name;
    string _attn_base_name;
    string _ffn_base_name;
    string _attn_norm_name;
    string _ffn_norm_name;

    string _qkv_proj_name;

    string _q_norm_name;
    string _k_norm_name;

    string _bias_k_name = "bias_k";
    string _bias_v_name = "bias_v";
};

class TransformerConfig {
public:
    TransformerConfig() {
    }
    string attn_implementation = "flash_attention_2"; // Options: "flash_attention_2", "eager"
    DataType dtype = MLLM_TYPE_F32;
};
#endif // CONFIGURATION_TRANSFORMER_HPP
