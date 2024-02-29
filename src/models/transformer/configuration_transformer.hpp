//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef CONFIGURATION_TRANSFORMER_HPP
#define CONFIGURATION_TRANSFORMER_HPP
using namespace mllm;

class TransformerNameConfig {
public:
    std::string _q_proj_name;
    std::string _k_proj_name;
    std::string _v_proj_name;
    std::string _o_proj_name;
    std::string _gate_proj_name;
    std::string _up_proj_name;
    std::string _down_proj_name;
    std::string _attn_base_name;
    std::string _ffn_base_name;
    std::string _attn_norm_name;
    std::string _ffn_norm_name;
};
#endif //CONFIGURATION_TRANSFORMER_HPP
