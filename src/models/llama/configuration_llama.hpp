//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef CONFIG_LLAMA_HPP
#define CONFIG_LLAMA_HPP


using namespace mllm;

class LLaMAConfig {
public:
    static int vocab_size;
    static int hidden_dim;
    static int head_size;
    int attn_hidden_dim = hidden_dim/head_size;
    static int mlp_hidden;
    static int block_num;

    static RoPEType RoPE_type;

    static int cache_limit;

    static std::string host_name;
    static std::string _q_proj_name;
    static std::string _k_proj_name;
    static std::string _v_proj_name;
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
    std::string q_proj_name = attn_base_name+_q_proj_name;
    std::string k_proj_name = attn_base_name+_k_proj_name;
    std::string v_proj_name = attn_base_name+_v_proj_name;
    std::string o_proj_name = attn_base_name+_o_proj_name;
    std::string gate_proj_name = ffn_base_name+_gate_proj_name;
    std::string up_proj_name = ffn_base_name+_up_proj_name;
    std::string down_proj_name = ffn_base_name+_down_proj_name;
    std::string attn_norm_name =  base_name+_attn_norm_name;
    std::string ffn_norm_name = base_name+_ffn_norm_name;
    static std::string token_embd_name;
    static std::string post_norm_name;
    static std::string lm_head_name ;

    static void init(int token_limit, int billions=7, RoPEType type = LLAMAROPE) {
        if(billions == 7) {
            vocab_size = 32000;
            hidden_dim = 4096;
            head_size = 32;
            mlp_hidden = 11008;
            block_num = 32;
        }else {
            throw std::runtime_error("Unsupported model size");
        }
        switch (type) {
        case LLAMAROPE: {
            RoPE_type = RoPEType::LLAMAROPE;
            host_name = "";
            _attn_base_name =  "attention.";
            _ffn_base_name = "feed_forward.";
            _q_proj_name = "wq";
            _k_proj_name = "wk";
            _v_proj_name = "wv";
            _o_proj_name = "wo";
            _gate_proj_name = "w1";
            _up_proj_name = "w3";
            _down_proj_name = "w2";
            _attn_norm_name =  "attention_norm";
            _ffn_norm_name = "ffn_norm";
            token_embd_name = "tok_embeddings";
            post_norm_name = "norm";
            lm_head_name = "output";
            break;
        }
        case HFHUBROPE: {
            RoPE_type = RoPEType::HFHUBROPE;
            host_name = "model.";
            _attn_base_name = "self_attn.";
            _ffn_base_name = "mlp.";
            _q_proj_name = "q_proj";
            _k_proj_name = "k_proj";
            _v_proj_name = "v_proj";
            _o_proj_name = "o_proj";
            _gate_proj_name = "gate_proj";
            _up_proj_name = "up_proj";
            _down_proj_name = "down_proj";
            _attn_norm_name =  "input_layernorm";
            _ffn_norm_name = "post_attention_layernorm";
            token_embd_name = "embed_tokens";
            post_norm_name = "norm";
            lm_head_name = "lm_head";
            break;
        }
        default: {
            throw std::runtime_error("Unsupported llama type");
        }
        }
        cache_limit = token_limit;
    }
};
int LLaMAConfig::vocab_size;
int LLaMAConfig::hidden_dim;
int LLaMAConfig::head_size;
int LLaMAConfig::mlp_hidden;
int LLaMAConfig::block_num;
RoPEType LLaMAConfig::RoPE_type;
int LLaMAConfig::cache_limit = 200;
std::string LLaMAConfig::host_name;
std::string LLaMAConfig::_attn_base_name;
std::string LLaMAConfig::_ffn_base_name;
std::string LLaMAConfig::_q_proj_name;
std::string LLaMAConfig::_k_proj_name;
std::string LLaMAConfig::_v_proj_name;
std::string LLaMAConfig::_o_proj_name;
std::string LLaMAConfig::_gate_proj_name;
std::string LLaMAConfig::_up_proj_name;
std::string LLaMAConfig::_down_proj_name;
std::string LLaMAConfig::_attn_norm_name;
std::string LLaMAConfig::_ffn_norm_name;
std::string LLaMAConfig::token_embd_name;
std::string LLaMAConfig::post_norm_name;
std::string LLaMAConfig::lm_head_name;

#endif //CONFIG_LLAMA_HPP
