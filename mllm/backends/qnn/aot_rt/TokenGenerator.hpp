#pragma once

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/core/Tensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include <functional>

namespace mllm::qnn::aot {

template<typename T>
class TokenGenerator {
 public:
  TokenGenerator(mllm::preprocessor::AutoTokenizer* tokenizer, KVCacheManager<T>* kv_manager,
                 std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids, QnnAOTConfig config);

  virtual ~TokenGenerator() {
    // Clear module's output tensors before member tensors are destroyed
    // to avoid double-free or use-after-free issues
    if (module_) { module_->setOutputTensors({}); }
    output_tensors_.clear();
    input_tensors_.clear();
  }

  void init_io();

  virtual const std::vector<float>& get_all_logits();

  virtual int64_t generate(std::vector<int64_t>& tokens, int64_t start_pos, int32_t seq_len,
                           const std::function<void(const std::string&)>& token_callback, bool dump_logits);

 protected:
  mllm::preprocessor::AutoTokenizer* tokenizer_;
  std::unique_ptr<QnnAOTModule> module_;
  KVCacheManager<T>* kv_manager_;
  std::unique_ptr<std::unordered_set<uint64_t>> eos_ids_;
  QnnAOTConfig config_;

  std::vector<mllm::Tensor> input_tensors_;
  std::vector<mllm::Tensor> output_tensors_;
  std::vector<float> token_all_logits_;

  void prepare_io(uint64_t cur_token, int64_t start_pos);
};

}  // namespace mllm::qnn::aot
