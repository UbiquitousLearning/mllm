#include "mllm/backends/qnn/aot_rt/TokenGenerator.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include <cstring>
#include <utility>

namespace mllm::qnn::aot {

template<typename T>
TokenGenerator<T>::TokenGenerator(mllm::preprocessor::AutoTokenizer* tokenizer, KVCacheManager<T>* kv_manager,
                                  std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids, QnnAOTConfig config)
    : tokenizer_(tokenizer), kv_manager_(kv_manager), eos_ids_(std::move(eos_ids)), config_(config) {
  std::string graph_name = "model.0.s1";
  module_ = std::make_unique<QnnAOTModule>(graph_name);
  module_->to(kQNN);
}

template<typename T>
void TokenGenerator<T>::init_io() {
  input_tensors_.reserve(4 + 2 * config_.num_layers);

  // 1. Input IDs
  auto input_ids = Tensor::empty({1, 1}, kInt32, kQNN).alloc();
  input_ids.setName("input_ids");
  input_tensors_.push_back(input_ids);

  // 2. Position IDs
  auto pos_ids = Tensor::empty({1}, kInt32, kQNN).alloc();
  pos_ids.setName("position_ids");
  input_tensors_.push_back(pos_ids);

  // 3. Attention Mask
  auto attn_mask = Tensor::empty({1, 1, 1, config_.context_len}, kUInt16, kQNN).alloc();
  attn_mask.setName("attention_mask");
  input_tensors_.push_back(attn_mask);

  // 4. KV Caches
  const auto& k_caches = kv_manager_->getKCache();
  const auto& v_caches = kv_manager_->getVCache();
  for (int l = 0; l < config_.num_layers; ++l) {
    // K
    auto k_tensor = Tensor::empty({1, (int)config_.num_heads, config_.head_dim, config_.context_len}, config_.kv_dtype, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("past_key_" + std::to_string(l));
    input_tensors_.push_back(k_tensor);

    // V
    auto v_tensor =
        Tensor::empty({1, (int)config_.num_heads, config_.context_len - 1, config_.head_dim}, config_.kv_dtype, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("past_value_" + std::to_string(l));
    input_tensors_.push_back(v_tensor);
  }

  // Output Tensors
  output_tensors_.reserve(1 + 2 * config_.num_layers);

  // 1. Logits
  auto logits = Tensor::empty({1, 1, 1, config_.vocab_size}, kUInt16, kQNN).alloc();
  logits.setName("logits");
  output_tensors_.push_back(logits);

  // 2. KV Caches, should be consistant with modeling file, or it will cause error
  for (int l = 0; l < config_.num_layers; ++l) {
    // K Output
    auto k_tensor = Tensor::empty({1, (int)config_.num_heads, config_.head_dim, 1}, config_.kv_dtype, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].output_buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("present_key_" + std::to_string(l));
    output_tensors_.push_back(k_tensor);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    // V Output
    auto v_tensor = Tensor::empty({1, (int)config_.num_heads, 1, config_.head_dim}, config_.kv_dtype, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].output_buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("present_value_" + std::to_string(l));
    output_tensors_.push_back(v_tensor);
  }
}

template<typename T>
const std::vector<float>& TokenGenerator<T>::get_all_logits() {
  return token_all_logits_;
}

template<typename T>
void TokenGenerator<T>::prepare_io(uint64_t cur_token, int64_t start_pos) {
  // 1. Input IDs
  int32_t* input_ids_ptr = input_tensors_[0].ptr<int32_t>();
  input_ids_ptr[0] = (int32_t)cur_token;

  // 2. Position IDs
  int32_t* pos_ids_ptr = input_tensors_[1].ptr<int32_t>();
  pos_ids_ptr[0] = (int32_t)start_pos;
}

template<typename T>
int64_t TokenGenerator<T>::generate(std::vector<uint64_t>& tokens, int64_t start_pos, int32_t seq_len,
                                    const std::function<void(const std::string&)>& token_callback, bool dump_logits) {
  int64_t current_pos = start_pos;
  uint64_t next_token = tokens.back();
  int64_t generated_count = 0;

  // Ensure KV cache is arranged for decode (1 token)
  kv_manager_->rearrangeCache(1);

  module_->setOutputTensors(output_tensors_);

  for (int i = 0; i < seq_len; ++i) {
    if (current_pos >= config_.context_len) { break; }

    prepare_io(next_token, current_pos);

    // Run forward
    auto module_input = input_tensors_;
    output_tensors_ = (*module_)(module_input);

    // Update KV Cache
    int32_t n_update = 1;
    kv_manager_->updateCache(1, current_pos, n_update, {});

    // Get logits
    auto logits = output_tensors_[0].to(kCPU).squeeze(0);

    // Sample
    auto cur_token = module_->sampleGreedy(logits);

    next_token = cur_token;
    tokens.push_back(next_token);
    current_pos++;
    generated_count++;

    if (token_callback) {
      std::wstring wstr = tokenizer_->detokenize(next_token);
      std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
      token_callback(str);
    }

    if (eos_ids_ && eos_ids_->count(next_token)) { break; }
  }

  return generated_count;
}

// Explicit instantiations
template class TokenGenerator<uint16_t>;
template class TokenGenerator<uint8_t>;

}  // namespace mllm::qnn::aot
