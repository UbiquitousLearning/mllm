#include <fmt/core.h>
#include <cstdint>
#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>

#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/models/qwen_npu/tokenization_qwen.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

using mllm::Argparse;

MLLM_MAIN({
  mllm::initQnnBackend();

  const std::string config_path = "./config_1.8B_w8a16_qnn.json";
  const std::string model_path = "./qwen1.5-1.8b-chat-rot-qnn.mllm";

  auto qwen_tokenizer = mllm::models::qwen_npu::QwenTokenizer("./tokenizer.json", "./qwen_merges.txt");

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;

  auto cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path);
  auto model = mllm::models::qwen_npu::QwenForCausalLM("", cfg);

  auto param = mllm::load(model_path, file_version);
  model.load(param);

  mllm::models::ARGenerationOutputPast inputs{{"sequence", mllm::Tensor::empty({1, 128}, mllm::kInt64, mllm::kCPU).alloc()}};

  auto irs = model.trace(inputs, {});

  // QNN Graph Rewrite Pass
  mllm::ir::PassManager rewritePM(irs["model"]);
  rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
  rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
  rewritePM.run();

  // have a look at the IR after QNN Graph Rewrite Pass
  mllm::redirect("qwen_npu.mir", [&]() { mllm::print(irs["model"]); });

  // QNN Graph Build Pass
  mllm::ir::PassManager graphBuildPM(irs["model"]);
  graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
  graphBuildPM.run();

  // Debug: Check registered buffer count after graph build
  {
    auto qnn_backend = mllm::Context::instance().getBackend(mllm::kQNN);
    if (qnn_backend) {
      auto allocator = std::static_pointer_cast<mllm::qnn::QNNAllocator>(qnn_backend->allocator());
      if (allocator) {
        auto stats = allocator->getRegisteredBufferStats();
        MLLM_INFO("After graph build: {} buffers registered, {} MB", stats.count, stats.total_bytes / (1024 * 1024));
      }
    }
  }

  // cache has been updated due to trace, clear cache
  model.model.clearKVCache();

  auto raw_input_tokens = qwen_tokenizer.convertMessage({.prompt = "What can you do?"})["sequence"];
  print(raw_input_tokens);
  MLLM_INFO("raw_input_tokens shape: {} {}", raw_input_tokens.shape()[0], raw_input_tokens.shape()[1]);
  
  const int chunk_size = 128;
  int real_seq = static_cast<int>(raw_input_tokens.shape()[1]);
  const int eos_token_id = 151645;
  if (real_seq <= 0 || real_seq >= chunk_size) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kShapeError, "Invalid input length {} for chunk size {}", real_seq, chunk_size);
  }

  // manually set input data as fill op is not supported in QNN
  // IMPORTANT: inputs["sequence"] was created before trace and may have been processed by QNN backend
  // Recreate inputs from raw_input_tokens to ensure we're using fresh CPU memory
  mllm::models::ARGenerationOutputPast prefill_inputs{{"sequence", mllm::Tensor::empty({1, chunk_size}, mllm::kInt64, mllm::kCPU).alloc()}};
  auto ptr = prefill_inputs["sequence"].ptr<int64_t>();
  auto input_data = raw_input_tokens.ptr<int64_t>();
  
  // Copy tokenized input data
  for (int i = 0; i < real_seq; ++i) { ptr[i] = input_data[i]; }
  for (int i = real_seq; i < chunk_size; ++i) { ptr[i] = -1; }
  
  bool data_matches = true;
  for (int i = 0; i < real_seq; ++i) {
    if (ptr[i] != input_data[i]) {
      MLLM_ERROR("Data mismatch at index {}: expected {}, got {}", i, input_data[i], ptr[i]);
      data_matches = false;
    }
  }
  if (!data_matches) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kShapeError, "Failed to copy input data correctly");
  }

  // Prefill
  MLLM_INFO("=== Prefill Phase ===");
  MLLM_INFO("Input sequence length: {}", real_seq);
  
  // Debug: Verify prefill_inputs data right before forward
  {
    auto verify_ptr = prefill_inputs["sequence"].ptr<int64_t>();
    MLLM_INFO("prefill_inputs[\"sequence\"] right before forward (first 10): {} {} {} {} {} {} {} {} {} {}", 
              verify_ptr[0], verify_ptr[1], verify_ptr[2], verify_ptr[3], verify_ptr[4], 
              verify_ptr[5], verify_ptr[6], verify_ptr[7], verify_ptr[8], verify_ptr[9]);
    MLLM_INFO("prefill_inputs[\"sequence\"] device: {}, bytes: {}", 
              (int)prefill_inputs["sequence"].device(), prefill_inputs["sequence"].bytes());
  }
  
  auto prefill_output = model.forward(prefill_inputs, {{"seq_len", mllm::AnyValue(real_seq)}});
  auto& prefill_logits = prefill_output["sequence"];
  auto sampled = model.sampleGreedy(prefill_logits);
  prefill_logits.delete_();
  prefill_output.erase("sequence");
  MLLM_INFO("Prefill generated token id: {}", sampled);
  std::wcout << qwen_tokenizer.detokenize(sampled);

  // Decode loop - 新方案：每次完整 prefill
  int current_seq_len = real_seq;
  auto& sequence_tensor = prefill_inputs["sequence"];
  auto sequence_ptr = sequence_tensor.ptr<int64_t>();

  // write first token into padding
  sequence_ptr[current_seq_len] = sampled;
  current_seq_len++;

  // Clean up prefill output
  prefill_output.clear();

  // Debug: Check registered buffer count after prefill
  {
    auto qnn_backend = mllm::Context::instance().getBackend(mllm::kQNN);
    if (qnn_backend) {
      auto allocator = std::static_pointer_cast<mllm::qnn::QNNAllocator>(qnn_backend->allocator());
      if (allocator) {
        auto stats = allocator->getRegisteredBufferStats();
        MLLM_INFO("After prefill: {} buffers registered, {} MB", stats.count, stats.total_bytes / (1024 * 1024));
      }
    }
  }

  MLLM_INFO("=== Decode Phase ===");
  MLLM_INFO("Starting decode loop, initial seq_len: {}", current_seq_len);

  int decode_step = 0;
  while (current_seq_len < chunk_size) {
    decode_step++;
    MLLM_INFO("--- Decode Step {} ---", decode_step);
    MLLM_INFO("Current sequence length: {}", current_seq_len);

    // Reset KV cache to 0 for full prefill (重新计算所有 KV cache)
    model.setKVCacheSeqCnt(0);

    // IMPORTANT: Use full [1, 128] tensor, not a slice
    // QNN backend expects fixed-size input tensor [1, 128] as defined during graph build
    // We only use the first current_seq_len tokens, the rest are padding (-1)
    // Ensure padding area is properly set to -1
    for (int i = current_seq_len; i < chunk_size; ++i) {
      sequence_ptr[i] = -1;
    }

    // Use full sequence tensor - QNN backend will handle the size correctly
    // The seq_len parameter tells the model how many tokens are actually valid
    mllm::models::ARGenerationOutputPast decode_input{
        {"sequence", sequence_tensor},  // Use full [1, 128] tensor, not a slice
    };

    MLLM_INFO("Decode input sequence length: {} (using full [1, {}] tensor)", current_seq_len, chunk_size);
    
    // Forward with full sequence - this is a full prefill, not incremental decode
    // seq_len parameter tells the model to only process first current_seq_len tokens
    auto decode_output = model.forward(decode_input, {{"seq_len", mllm::AnyValue(current_seq_len)}});

    // Print KV cache length after decode
    auto kv_cache_len = model.getKVCacheSeqCnt(0);  // Get KV cache length from layer 0
    MLLM_INFO("KV cache length after decode step {}: {}", decode_step, kv_cache_len);

    // Sample next token
    auto& decode_logits = decode_output["sequence"];
    auto next_token = model.sampleGreedy(decode_logits);
    MLLM_INFO("Generated token id: {}", next_token);
    std::wcout << qwen_tokenizer.detokenize(next_token);

    // Check termination
    if (next_token == eos_token_id) {
      MLLM_INFO("EOS token detected, stopping decode");
      break;
    }

    // Write new token into sequence buffer
    sequence_ptr[current_seq_len] = next_token;
    current_seq_len++;
    MLLM_INFO("Updated sequence length: {}", current_seq_len);

    // Clean up - no need to keep position_ids since we're doing full prefill each time
    decode_logits.delete_();
    decode_output.clear();

    // Debug: Check registered buffer count after each decode step
    {
      auto qnn_backend = mllm::Context::instance().getBackend(mllm::kQNN);
      if (qnn_backend) {
        auto allocator = std::static_pointer_cast<mllm::qnn::QNNAllocator>(qnn_backend->allocator());
        if (allocator) {
          auto stats = allocator->getRegisteredBufferStats();
          MLLM_INFO("After decode step {}: {} buffers registered, {} MB", 
                    decode_step, stats.count, stats.total_bytes / (1024 * 1024));
        }
      }
    }
  }

  MLLM_INFO("=== Decode Complete ===");
  MLLM_INFO("Total decode steps: {}", decode_step);
  MLLM_INFO("Final sequence length: {}", current_seq_len);
  MLLM_INFO("Remaining capacity: {}", chunk_size - current_seq_len);
  std::wcout << L"\n";

  return 0;
})
