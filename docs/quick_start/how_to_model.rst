########################################
How to Support a New LLM: Step-by-Step
########################################

This guide walks you through adding a brand-new Large Language Model (LLM) to the **mllm** inference framework.  
We use **Qwen3** as a running example, but the same workflow applies to any transformer-style model.


*********************************************
Step 1: Acquire the Model Assets
*********************************************

1. Download the original model from Hugging-Face (or any other reputable source).

   Typical files you need:

   * ``config.json``
   * ``tokenizer.json`` / ``tokenizer.model``
   * PyTorch / Safetensors checkpoints (``.bin``, ``.safetensors``)

2. Place everything under a single directory, e.g. ``~/models/Qwen3-0.6B``.

.. note::
   Models obtained from hosting platforms such as Hugging Face or ModelScope (via ``git clone`` or their official CLI) are already organized in a single directory that contains ``config.json``, ``tokenizer.json``, ``tokenizer.model``, checkpoint shards, etc.


You can download Qwen3-0.6B from ModelScope with the following command:

.. code-block:: bash

   git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git

.. note::
   **About Model Version:** 
   Most mllm models on Hugging-Face are in v1 version file format, which has no tensor shape and supports less data types. When loading a model, you can specify the model version. We maintain the compatibility of both v1 and v2 formats in mllm. It is recommended to use v2 format for new models whenever possible.


*********************************************
Step 2: Convert to mllm Format
*********************************************

mllm ships a dedicated converter called ``mllm-convertor``.  
It translates Hugging-Face / PyTorch checkpoints into the internal ``*.mllm`` format.

Install pymllm (Python bindings)
================================

.. code-block:: bash

   bash ./scripts/install_pymllm.sh

.. note::
   Once the `mllm` organisation is approved on PyPI you will be able to run:

   .. code-block:: bash

      pip install pymllm

Run the converter
=================

.. code-block:: bash

   mllm-convertor \
      --input_path  ./Qwen3-0.6B/model.safetensors \
      --output_path ./Qwen3-0.6B/w4a32.mllm \
      --cfg_path    ./Qwen3-0.6B/quant_config.json \
      --pipeline    w4a32_kai_pipeline

For sharded checkpoints the converter automatically follows the ``*.index.json`` file—no manual merging required.

Custom quantization recipe (optional)
========================================

Supply a JSON map if you need quantization for specific layers.

Example snippet:

.. code-block:: json

   {
     // KAI Config
     "^model\\.layers\\.\\d+\\.self_attn\\.q_proj.(bias|weight)": {
       "hints": {
         "quant_method": "kai",
         "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
         "kai_matmul_layout": "mxk_nxk",
         "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
         "shape": [2048, 1024],
         "replace": true
       }
     }
     // GGUF Config
     "^model\\.layers\\.\\d+\\.self_attn\\.q_proj.(bias|weight)": {
       "hints": {
         "quant_method": "gguf",
         "gguf_type": "Q4_0",
         "shape": [2048, 1024],
         "replace": true
       }
     }
     (...)
   }

Save the file (e.g. ``quant_config.json``) and pass it with ``--cfg_path quant_config.json``.

See full ``quant_config.json`` in Appendix.

***************************************************************************
Step 3: (Optional) On-Device Quantization with mllm-quantizer
***************************************************************************

``mllm-convertor`` already embeds the quantization engine, but **mllm-quantizer** is still useful when:

* you target an Android handset and want to quantize **on the phone**, or
* the quantization kernel is only compiled for ARM.

.. code-block:: bash

   mllm-quantizer \
      -i  ./Qwen3-0.6B/model.mllm \
      -c ./Qwen3-0.6B/quant_config.json \
      -iv v2 \
      -o ./Qwen3-0.6B/w4a32.mllm \
      -ov v2

.. note::
   Basically, if you have no ARM DEVICE(Mac with apple silicon or Arm PC) to quantize your model through pymllm in kai settings. You should use mllm-quantizer to quantize your model on your arm devices (maybe android phone).

Supported Quantization Types in mllm-quantizer
==============================================

The `mllm-quantizer` tool supports several quantization types, allowing you to optimize model size and inference speed for different hardware targets. Below is a summary of the main quantization types available:

**GGUF Quantization Types (CPU, cross-platform):**

- `Q4_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q6_K`, `Q8_K`: GGUF per-group quantization.

**KAI Quantization Types (ARM, Apple Silicon):**

- `KAI_fp16_fp16_fp16p_mxk_kxn`: FP16 result, FP16 activation, FP16 weight, packed in kai format.
- `KAI_f32_qai8dxp_qsi4c32p_mxk_nxk`: FP16 result, Int8 activation(asymmetric, per-token), Int4 weight(symmetric, per-group, 32 pack, transposed layout).
- `KAI_f16_qsi8d32p_qai4c32p_mxk_nxk`: FP16 result, Int8 activation(symmetric, per-group, 32 pack), Int4 weight(asymmetric, per-group, 32 pack, transposed layout).

**How to Select Quantization Type:**
- For general CPU deployment, use GGUF types (`Q4_0`, `Q8_0`, etc.).
- For ARM devices (Android, Apple Silicon), use KAI types for best performance and compatibility.
- Specify the quantization type in your config or pipeline when running `mllm-quantizer`.

For more details on each quantization type and their configuration, refer to the quantization implementation in the source code or the `kleidiai documentation <https://github.com/ARM-software/kleidiai/blob/main/kai/ukernels/matmul/README.md>`_.

*********************************************
Step 4: Implement Core C++ Files
*********************************************

mllm mirrors the Hugging-Face *config / tokenizer / model* split.  
Create three files under ``mllm/models/qwen3/``:

1. ``configuration_qwen3.hpp``
2. ``tokenization_qwen3.hpp``
3. ``modeling_qwen3.hpp``

See full ``configuration_qwen3.hpp``, ``tokenization_qwen3.hpp``, ``modeling_qwen3.hpp`` in Appendix.

*********************************************
Step 5: Create an Example Application
*********************************************

Directory layout

.. code-block:: text

   examples/qwen3/
   ├── main.cpp
   └── config_0.6B_w4a32_kai.json

main.cpp
=============

.. code-block:: cpp

   #include "mllm/mllm.hpp"
   #include "mllm/models/qwen3/modeling_qwen3.hpp"
   #include "mllm/models/qwen3/tokenization_qwen3.hpp"
   #include "mllm/models/qwen3/configuration_qwen3.hpp"

   int main(int argc, char* argv[]) {
       mllm::init();

       std::string config_path   = "...";
       std::string tokenizer_path= "...";
       std::string prompt        = "Once upon a time";

       auto cfg       = mllm::models::qwen3::Qwen3Config(config_path);
       auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path);
       auto model     = mllm::models::qwen3::Qwen3ForCausalLM(cfg);

       auto inputs = tokenizer.convertMessage({.prompt = prompt});
       for (auto& step : model.chat(inputs)) {
           std::wcout << tokenizer.detokenize(step.cur_token_id) << std::flush;
       }
       return 0;
   }

config.json (example)
=========================

.. code-block:: json

   {
     "architectures": ["Qwen3ForCausalLM"],
     "bos_token_id": 151643,
     "eos_token_id": 151645,
     "attention_bias": false,
     "hidden_size": 1024,
     "head_dim": 128,
     "intermediate_size": 3072,
     "num_attention_heads": 16,
     "num_key_value_heads": 8,
     "num_hidden_layers": 28,
     "max_position_embeddings": 40960,
     "rms_norm_eps": 1e-06,
     "vocab_size": 151936,
     "max_cache_length": 2048,
     "rope_theta": 1000000.0,
     "tie_word_embeddings": true,
     "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"
   }

Build & run

.. code-block:: bash

   python task.py tasks/build_<...>.py

******************************
Step 6: Open a Pull Request
******************************

1. Fork the official repository.
2. Create a feature branch: ``git checkout -b add-qwen3``.
3. Commit atomic changes with clear messages:

   .. code-block:: text

      [Qwen3] Add configuration loader
      [Qwen3] Implement Sentence-Piece tokenizer
      [Qwen3] Add CI unit test and 2-sample generation

4. Push and open a PR.  

A maintainer will review numerical correctness, coding style, and CI status before merging.

***************
Conclusion
***************

You have now walked through the complete life-cycle of integrating a new LLM into mllm:

1. Acquire Models → 2. Convert to MLLM models → 3. (Optionally) Quantize → 4. Implement C++ stubs → 5. Example & test → 6. PR.

Following the checklist above guarantees that your model will load efficiently, run everywhere mllm runs, and is maintainable by the community. Happy hacking!

***************
Appendix
***************

quant_config.json
===================

.. code-block:: json

   {
    "^model\\.layers\\.\\d+\\.self_attn\\.q_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                2048,
                1024
            ],
            "replace": true
        }
    },
    "^model\\.layers\\.\\d+\\.self_attn\\.k_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                1024,
                1024
            ],
            "replace": true
        }
    },
    "^model\\.layers\\.\\d+\\.self_attn\\.v_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                1024,
                1024
            ],
            "replace": true
        }
    },
    "^model\\.layers\\.\\d+\\.self_attn\\.o_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                1024,
                2048
            ],
            "replace": true
        }
    },
    "^model\\.layers\\.\\d+\\.mlp\\.gate_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                3072,
                1024
            ],
            "replace": true
        }
    },
    "^model\\.layers\\.\\d+\\.mlp\\.up_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                3072,
                1024
            ],
            "replace": true
        }
    },
    "^model\\.layers\\.\\d+\\.mlp\\.down_proj.(bias|weight)": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                1024,
                3072
            ],
            "replace": true
        }
    },
    "lm_head.weight": {
        "hints": {
            "quant_method": "kai",
            "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
            "kai_matmul_layout": "mxk_nxk",
            "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
            "shape": [
                151936,
                1024
            ],
            "replace": false,
            "rename": "lm_head_out.weight"
        }
    }
   }

configuration_qwen3.hpp
==========================

.. code-block:: cpp
   
   // Copyright (c) MLLM Team.
   // Licensed under the MIT License.
   #pragma once

   #include "mllm/core/aops/LinearOp.hpp"
   #include "mllm/engine/ConfigFile.hpp"

   namespace mllm::models::qwen3 {

   struct Qwen3Config : protected ConfigFile {
     Qwen3Config() = default;

     explicit Qwen3Config(const std::string& file_path) : ConfigFile(file_path) {
       // Init all
       attention_bias = data()["attention_bias"];
       hidden_size = data()["hidden_size"];
       intermediate_size = data()["intermediate_size"];
       num_attention_heads = data()["num_attention_heads"];
       num_key_value_heads = data()["num_key_value_heads"];
       num_hidden_layers = data()["num_hidden_layers"];
       max_position_embeddings = data()["max_position_embeddings"];
       rms_norm_eps = data()["rms_norm_eps"];
       vocab_size = data()["vocab_size"];
       head_dim = data()["head_dim"];

       bos_token_id = data()["bos_token_id"];
       eos_token_id = data()["eos_token_id"];
       rope_theta = data()["rope_theta"];

       tie_word_embeddings = data()["tie_word_embeddings"];
       max_cache_length = data()["max_cache_length"];

       linear_impl_type = aops::str2LinearImplTypes(data()["linear_impl_type"]);
     }

     bool attention_bias = false;
     int32_t hidden_size = 1024;
     int32_t head_dim = 128;
     int32_t intermediate_size = 3072;
     int32_t num_attention_heads = 16;
     int32_t num_key_value_heads = 8;
     int32_t num_hidden_layers = 28;
     int32_t max_position_embeddings = 40960;
     float rms_norm_eps = 1e-06;
     int32_t vocab_size = 151936;

     int64_t bos_token_id = 151643;
     int64_t eos_token_id = 151645;
     float rope_theta = 1000000.0;

     bool tie_word_embeddings = true;
     int32_t max_cache_length = 2048;
     int32_t end_of_text_token_id = 151645;

     aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;
   };

   }  // namespace mllm::models::qwen3


modeling_qwen3.hpp
=====================

.. code-block:: cpp

   // Copyright (c) MLLM Team.
   // Licensed under the MIT License.

   #include "mllm/mllm.hpp"
   #include "mllm/nn/Module.hpp"
   #include "mllm/nn/Nn.hpp"
   #include "mllm/nn/Functional.hpp"
   #include "mllm/nn/lmcache/StaticCache.hpp"
   #include "mllm/models/qwen3/configuration_qwen3.hpp"
   #include "mllm/utils/Enumerate.hpp"
   #include "mllm/models/ARGeneration.hpp"

   namespace mllm::models::qwen3 {

   inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
     auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
     auto inv_freq_ptr = inv_freq.ptr<float>();
     for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
     return inv_freq;
   }

   inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                      float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
     auto batch_size = position_ids.shape()[0];
     auto seq_len = position_ids.shape()[1];
     auto inv_freq_len = inv_freq.shape()[0];
     auto dim = inv_freq_len * 2;

     // Create freqs tensor: position_ids @ inv_freq
     auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
     auto freqs_ptr = freqs.ptr<float>();
     auto position_ids_ptr = position_ids.ptr<int64_t>();
     auto inv_freq_ptr = inv_freq.ptr<float>();

     // Compute freqs = position_ids[:, :, None] @ inv_freq[None, :]
     for (int b = 0; b < batch_size; ++b) {
       for (int s = 0; s < seq_len; ++s) {
         auto pos = position_ids_ptr[b * seq_len + s];
         for (int d = 0; d < inv_freq_len; ++d) {
           freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
         }
       }
     }

     // Create sin and cos tensors with shape [batch_size, seq_len, dim]
     auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
     auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
     auto sin_ptr = sin_emb.ptr<float>();
     auto cos_ptr = cos_emb.ptr<float>();

     // Compute sin and cos embeddings: emb = [freqs, freqs]
     for (int b = 0; b < batch_size; ++b) {
       for (int s = 0; s < seq_len; ++s) {
         for (int d = 0; d < inv_freq_len; ++d) {
           auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
           auto sin_val = std::sin(freq) * attention_scaling;
           auto cos_val = std::cos(freq) * attention_scaling;

           // Store the same values in both halves: [freqs, freqs]
           sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
           sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
           cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
           cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
         }
       }
     }

     return {sin_emb, cos_emb};
   }

   class Qwen3MLP final : public nn::Module {
     nn::Linear gate_proj_;
     nn::Linear up_proj_;
     nn::Linear down_proj_;
     nn::SiLU silu_;

    public:
     Qwen3MLP() = default;
     Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
       gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
       silu_ = reg<nn::SiLU>("act");
       up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
       down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
     }

     std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
       auto x = gate_proj_(inputs[0]);
       x = silu_(x);
       auto y = up_proj_(inputs[0]);
       x = x * y;
       x = down_proj_(x);
       return {x};
     }
   };

   class Qwen3Attention final : public nn::Module {
     nn::Linear q_proj_;
     nn::Linear k_proj_;
     nn::Linear v_proj_;
     nn::Linear o_proj_;
     nn::RMSNorm rms_norm_q_;
     nn::RMSNorm rms_norm_k_;
     nn::RoPE q_rope_;
     nn::RoPE k_rope_;
     nn::CausalMask mask_;
     nn::Softmax softmax_;

     int hidden_size_;
     int head_dim_;
     int num_attention_heads_;
     int num_key_value_heads_;
     int num_key_value_groups_;

    public:
     Qwen3Attention() = default;

     Qwen3Attention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
       hidden_size_ = cfg.hidden_size;
       num_attention_heads_ = cfg.num_attention_heads;
       num_key_value_heads_ = cfg.num_key_value_heads;
       head_dim_ = cfg.head_dim;
       num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

       q_proj_ =
           reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
       k_proj_ =
           reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
       v_proj_ =
           reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
       o_proj_ =
           reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

       rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
       rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

       q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
       k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);

       mask_ = reg<nn::CausalMask>("mask");
       softmax_ = reg<nn::Softmax>("softmax", -1);
     }

     std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
       auto x = inputs[0];
       auto llm_embedding_sin = inputs[1];
       auto llm_embedding_cos = inputs[2];
       auto past_kv_cache = args[0].get<nn::StaticCache*>();

       // [B, S, H * D]
       auto query_states = q_proj_(x);
       auto key_states = k_proj_(x);
       auto value_states = v_proj_(x);

       int B = inputs[0].shape()[0];
       int S = inputs[0].shape()[1];

       // [B, S, H, D]
       query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
       key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
       value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

       // [B, S, H, D]
       query_states = rms_norm_q_(query_states);
       key_states = rms_norm_k_(key_states);

       // [B, H, S, D]
       query_states = query_states.transpose(1, 2);
       key_states = key_states.transpose(1, 2);
       value_states = value_states.transpose(1, 2);

       // [B, H, S, D]
       query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
       key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

       // [B, H, S, D]
       std::tie(key_states, value_states) = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);

       Tensor attn;
       if (key_states.dtype() == kFloat32) {
         // attention weight
         // [B, H, S, S]
         attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
         attn = mask_(attn);
         attn = softmax_(attn);
       } else if (key_states.dtype() == kFloat16) {
         attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
         attn = mask_(attn);
         attn = softmax_(attn);
         attn = attn.to(kFloat16);
       }

       // attn output
       // [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
       auto output = nn::functional::matmul(attn, value_states);
       // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
       output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
       output = o_proj_(output);

       return {output};
     }

     int layer_idx_;
   };

   class Qwen3Decoder final : public nn::Module {
    public:
     Qwen3Attention self_attn_;
     Qwen3MLP mlp_;
     nn::RMSNorm input_layer_norm_;
     nn::RMSNorm post_attention_layer_norm_;

     Qwen3Decoder() = default;

     Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
       self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
       mlp_ = reg<Qwen3MLP>("mlp", cfg);
       input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
       post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
     }

     std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
       auto llm_embedding_sin = inputs[1];
       auto llm_embedding_cos = inputs[2];
       auto& kv_cache = args[0];

       auto x = input_layer_norm_(inputs[0]);
       x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
       auto tmp = x + inputs[0];
       x = post_attention_layer_norm_(tmp);
       x = mlp_(x)[0];
       x = x + tmp;
       return {x};
     }
   };

   class Qwen3Text final : public nn::Module {
     nn::ModuleList<Qwen3Decoder> decode_blocks_;
     nn::RMSNorm norm_;
     nn::Embedding embedding_;

    public:
     Qwen3Text() = default;

     Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
       decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
       for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
       norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
       embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
     }

     std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
       auto& blocks = decode_blocks_.list();

       // X is already embedded
       auto x = embedding_(inputs[0]);

       auto llm_embedding_sin = inputs[1];
       auto llm_embedding_cos = inputs[2];
       auto& kv_cache = args[0];

       for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }

       x = norm_(x);

       return {x};
     }
   };

   class Qwen3ForCausalLM : public ARGeneration, public nn::Module {
    public:
     explicit Qwen3ForCausalLM(const Qwen3Config& cfg) : cfg(cfg) {
       kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                   cfg.num_attention_heads,  // q_heads
                                   cfg.num_key_value_heads,  // kv_heads
                                   cfg.head_dim,             // kv_dim
                                   kFloat32,                 // k_dtype
                                   kFloat32,                 // v_dtype
                                   kCPU,                     // device_type
                                   false                     // use_fa2
       );
       eos_token_id_ = cfg.end_of_text_token_id;
       max_length_ = cfg.max_cache_length;
       tie_word_embeddings_ = cfg.tie_word_embeddings;

       llm = reg<Qwen3Text>("model", cfg);

       if (cfg.tie_word_embeddings) {
         // NOTE:
         // model.lm_head.weight is quantization weights of model.embed_tokens.weight
         lm_head_ = reg<nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
       }

       // Init inv freq
       auto inv = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
       registerBuffer("inv_freq", inv);
     }

     ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
       auto sequence = input.at("sequence");

       // Generate position_ids for the current sequence
       auto batch_size = sequence.shape()[0];
       auto seq_len = sequence.shape()[1];

       Tensor position_ids = Tensor::nil();
       if (input.count("position_ids")) {
         // Use existing position_ids for decode phase
         position_ids = input.at("position_ids");

         // For decode phase, increment the last position
         if (seq_len == 1) {
           auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
           position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
           *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
         }
       } else {
         // Generate position_ids for prefill phase
         position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
         auto position_ids_ptr = position_ids.ptr<int64_t>();
         for (int b = 0; b < batch_size; ++b) {
           for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
         }
       }

       // Generate RoPE embeddings using the inv_freq buffer
       auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);

       sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

       // clip x to one seq length
       {
         auto S = sequence.shape()[1];
         sequence = sequence[{kAll, {S - 1}, kAll}];
       }
       if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

       return {
           {"sequence", sequence},
           {"position_ids", position_ids},
       };
     }

    private:
     const Qwen3Config& cfg;
     Qwen3Text llm;
     nn::Linear lm_head_;
     bool tie_word_embeddings_;
     nn::StaticCache kv_cache_;
   };

   }  // namespace mllm::models::qwen3

tokenization_qwen3.hpp
========================

.. code-block:: cpp

   // Copyright (c) MLLM Team.
   // Licensed under the MIT License.
   #pragma once

   #include <vector>
   #include <unordered_map>

   #include "mllm/preprocessor/tokenizers/BPE.hpp"
   #include "mllm/models/ARGeneration.hpp"
   #include "mllm/preprocessor/tokenizers/Unicode.hpp"
   #include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

   namespace mllm::models::qwen3 {

   // we need to handle this:
   //
   // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
   // ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
   inline bool qwen3TokenizerMatchPattern(const std::wstring& str, size_t& pos, std::wstring& matched) {
     if (pos >= str.size()) return false;

     // 1. Match contractions: "'s|'t|'re|'ve|'m|'ll|'d"
     static const std::wstring contractions[] = {L"'s", L"'t", L"'re", L"'ve", L"'m", L"'ll", L"'d"};
     for (const auto& contraction : contractions) {
       if (pos + contraction.size() <= str.size() && str.compare(pos, contraction.size(), contraction) == 0) {
         matched = contraction;
         pos += contraction.size();
         return true;
       }
     }

     // 2. Match [^\r\n\p{L}\p{N}]?\p{L}+ (non-letter/digit followed by letters)
     {
       size_t original_pos = pos;
       bool has_prefix = false;
       matched.clear();

       // Check optional non-letter/digit prefix (excluding \r\n)
       if (!preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos]) && str[pos] != L'\r' && str[pos] != L'\n') {
         matched += str[pos];
         ++pos;
         has_prefix = true;
       }

       // Require at least one letter
       if (pos < str.size() && preprocessor::isLetter(str[pos])) {
         do {
           matched += str[pos];
           ++pos;
         } while (pos < str.size() && preprocessor::isLetter(str[pos]));
         return true;
       } else {
         // Rollback if no letters after prefix
         if (has_prefix) {
           pos = original_pos;
           matched.clear();
         }
       }
     }

     // 3. Match \p{N} (digits)
     if (preprocessor::isDigit(str[pos])) {
       matched = str.substr(pos, 1);
       ++pos;
       return true;
     }

     // 4. Match ?[^\s\p{L}\p{N}]+[\r\n]* (punctuation/symbols with optional space prefix)
     {
       size_t original_pos = pos;
       matched.clear();
       size_t start = pos;

       // Optional space
       if (str[pos] == L' ') { ++pos; }

       // Require at least one non-letter/digit/whitespace
       if (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos]) && !preprocessor::isDigit(str[pos])) {
         do {
           ++pos;
         } while (pos < str.size() && !std::iswspace(str[pos]) && !preprocessor::isLetter(str[pos])
                  && !preprocessor::isDigit(str[pos]));

         // Capture from start (after optional space) to current pos
         matched = str.substr(start, pos - start);

         // Capture trailing newlines
         while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
           matched += str[pos];
           ++pos;
         }
         return true;
       } else {
         // Rollback if no symbols found
         pos = original_pos;
       }
     }

     // 5. Match \s*[\r\n]+ (newlines with leading whitespace)
     {
       size_t start = pos;
       while (pos < str.size() && std::iswspace(str[pos])) ++pos;
       if (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) {
         while (pos < str.size() && (str[pos] == L'\r' || str[pos] == L'\n')) ++pos;
         matched = str.substr(start, pos - start);
         return true;
       } else {
         pos = start;
       }
     }

     // 6. Match \s+(?!\S) (whitespace not followed by non-space)
     if (std::iswspace(str[pos])) {
       size_t start = pos;
       while (pos < str.size() && std::iswspace(str[pos])) ++pos;
       // Check if at end or followed by whitespace
       if (pos >= str.size() || std::iswspace(str[pos])) {
         matched = str.substr(start, pos - start);
         return true;
       } else {
         pos = start;
       }
     }

     // 7. Match remaining whitespace
     if (std::iswspace(str[pos])) {
       size_t start = pos;
       while (pos < str.size() && std::iswspace(str[pos])) ++pos;
       matched = str.substr(start, pos - start);
       return true;
     }

     return false;
   }

   inline bool qwen3Regex(const std::string& str, std::vector<std::wstring>& splitted) {
     auto w_string = preprocessor::utf8string2WideString(str);
     size_t pos = 0;
     while (pos < w_string.size()) {
       std::wstring matched;
       if (qwen3TokenizerMatchPattern(w_string, pos, matched)) {
         splitted.push_back(matched);
       } else {
         ++pos;
       }
     }
     return true;
   }

   struct Qwen3Message {
     std::string prompt;
     static inline std::string message_template =
         "<|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
   };

   class Qwen3Tokenizer final : public mllm::preprocessor::AutoTokenizer {
    public:
     explicit Qwen3Tokenizer(const std::string& file_path) {
       preprocessor::initLocal();
       preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
       for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
       bpe_.initFromSentencePieceJson(file_path);
       special_tokens_trie_.add(L"<|endoftext|>");
       special_tokens_trie_.add(L"<|im_start|>");
       special_tokens_trie_.add(L"<|im_end|>");
       special_tokens_trie_.add(L"<|object_ref_start|>");
       special_tokens_trie_.add(L"<|object_ref_end|>");
       special_tokens_trie_.add(L"<|box_start|>");
       special_tokens_trie_.add(L"<|box_end|>");
       special_tokens_trie_.add(L"<|quad_start|>");
       special_tokens_trie_.add(L"<|quad_end|>");
       special_tokens_trie_.add(L"<|vision_start|>");
       special_tokens_trie_.add(L"<|vision_end|>");
       special_tokens_trie_.add(L"<|vision_pad|>");
       special_tokens_trie_.add(L"<|image_pad|>");
       special_tokens_trie_.add(L"<|video_pad|>");
       special_tokens_trie_.add(L"<think>");
       special_tokens_trie_.add(L"</think>");
     }

     std::vector<std::wstring> _tokenize(const std::string& str) override {
       std::vector<std::wstring> ret;
       std::vector<std::wstring> splitted;
       ::mllm::models::qwen3::qwen3Regex(str, splitted);
       for (const auto& s : splitted) {
         auto utf_8_str = preprocessor::wideString2Utf8String(s);
         std::wstring mapped_str;
         for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }

         auto bpe_ts = bpe_._bpe(mapped_str);

         for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
       }

       return ret;
     }

     std::vector<std::wstring> tokenize(const std::string& str) override {
       auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
       std::vector<std::wstring> all_tokens;
       for (const auto& token : tokens) {
         if (special_tokens_trie_.isSpecialToken(token)) {
           all_tokens.emplace_back(token);
           continue;
         }
         auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
         all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
       }
       return all_tokens;
     }

     std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

     std::wstring detokenize(int64_t pos_idx) override {
       auto str = _detokenize(pos_idx);
       std::string utf_8_str;
       for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
       return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
     }

     Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
       std::vector<int64_t> ids;
       ids.reserve(strs.size());
       for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
       Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                        .setMemType(kExtraInput)
                        .setName("qwen2-tokenizer-i0")
                        .alloc();

       auto ptr = ret.ptr<int64_t>();
       for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

       return ret;
     }

     ARGenerationOutputPast convertMessage(const Qwen3Message& message) {
       // process prompt
       auto applied_string = Qwen3Message::message_template;
       size_t pos = applied_string.find("{{{prompt}}}");
       applied_string.replace(pos, 12, message.prompt);

       // process sequence
       auto sequence_str = tokenize(applied_string);
       std::vector<int64_t> ids;
       ids.reserve(sequence_str.size());
       for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

       // Get sequence Tensor
       Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                             .setMemType(kNormal)
                             .setName("qwen2-tokenizer-i0")
                             .alloc();

       auto ptr = sequence.ptr<int64_t>();
       for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

       return {
           {"sequence", sequence},
       };
     }

    private:
     // For text
     preprocessor::BPE bpe_;
     std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
     std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
   };

   }  // namespace mllm::models::qwen3
