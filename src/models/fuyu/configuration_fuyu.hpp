//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_FUYU_HPP
#define CONFIG_FUYU_HPP


using namespace mllm;

class FuyuConfig {
public:
    static int vocab_size;
    static int hidden_dim;
    static int head_size;
    int attn_hidden_dim = hidden_dim/head_size;
    static int mlp_hidden;
    static int block_num;
    static int patch_size;
    static int chl_size;

    static RoPEType RoPE_type;

    static int cache_limit;

    static std::string host_name;
    static std::string _qkv_proj_name;
    static std::string _q_norm_name;
    static std::string _k_norm_name;
    static std::string _o_proj_name;
    static std::string _gate_proj_name;
    static std::string _up_proj_name;
    static std::string _down_proj_name;
    static std::string _attn_base_name;
    static std::string _ffn_base_name;
    static std::string _attn_norm_name;
    static std::string _ffn_norm_name;
    std::string base_name = host_name + "layers."+std::to_string(Module::listIdx)+ ".";
    std::string attn_base_name = base_name+ _attn_base_name;
    std::string ffn_base_name = base_name+ _ffn_base_name;
    std::string qkv_proj_name = attn_base_name+_qkv_proj_name;
    std::string q_norm_name = attn_base_name+_q_norm_name;
    std::string k_norm_name = attn_base_name+_k_norm_name;
    std::string o_proj_name = attn_base_name+_o_proj_name;
    std::string up_proj_name = ffn_base_name+_up_proj_name;
    std::string down_proj_name = ffn_base_name+_down_proj_name;
    std::string attn_norm_name =  base_name+_attn_norm_name;
    std::string ffn_norm_name = base_name+_ffn_norm_name;
    static std::string token_embd_name;
    static std::string vision_embed_tokens_name;
    static std::string post_norm_name;
    static std::string lm_head_name ;

    static void init(int token_limit, string billions = "8B") {
        vocab_size = 262144;
        if (billions == "8B" || billions == "8b") {
            hidden_dim = 4096;
            head_size = 64;
            mlp_hidden = 4096 * 4;
            block_num = 36;
            patch_size = 30;
            chl_size = 3;
        } else {
            throw std::runtime_error("Unsupported model size");
        }
        vision_embed_tokens_name = "vision_embed_tokens";
        token_embd_name = "language_model.model.embed_tokens";
        host_name = "language_model.model.";
        _attn_base_name = "self_attn.";
        _ffn_base_name = "mlp.";
        _qkv_proj_name = "query_key_value";
        _q_norm_name = "q_layernorm";
        _k_norm_name = "k_layernorm";
        _o_proj_name = "dense";
        _up_proj_name = "dense_h_to_4h";
        _down_proj_name = "dense_4h_to_h";
        _attn_norm_name = "input_layernorm";
        _ffn_norm_name = "post_attention_layernorm";
        post_norm_name = "language_model.model.final_layernorm";
        lm_head_name = "language_model.lm_head";

        cache_limit = token_limit;
    }
};
int FuyuConfig::vocab_size;
int FuyuConfig::hidden_dim;
int FuyuConfig::head_size;
int FuyuConfig::mlp_hidden;
int FuyuConfig::block_num;
int FuyuConfig::patch_size;
int FuyuConfig::chl_size;
RoPEType FuyuConfig::RoPE_type;
int FuyuConfig::cache_limit = 700;
std::string FuyuConfig::host_name;
std::string FuyuConfig::_attn_base_name;
std::string FuyuConfig::_ffn_base_name;
std::string FuyuConfig::_qkv_proj_name;
std::string FuyuConfig::_q_norm_name;
std::string FuyuConfig::_k_norm_name;
std::string FuyuConfig::_o_proj_name;
std::string FuyuConfig::_up_proj_name;
std::string FuyuConfig::_down_proj_name;
std::string FuyuConfig::_attn_norm_name;
std::string FuyuConfig::_ffn_norm_name;
std::string FuyuConfig::token_embd_name;
std::string FuyuConfig::vision_embed_tokens_name;
std::string FuyuConfig::post_norm_name;
std::string FuyuConfig::lm_head_name;

#endif //CONFIG_FUYU_HPP
