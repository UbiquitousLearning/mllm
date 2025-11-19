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

  // cache has been updated due to trace, clear cache
  model.model.clearKVCache();

  auto raw_input_tokens = qwen_tokenizer.convertMessage({.prompt = "提示:海洋世界里，鲸鱼是地球上体型最为庞大的哺乳动物，它们拥有流线型的身躯，主要通过头顶的喷水孔进行呼吸。与终生生活在水下并利用鱼鳃从水中提取溶解氧的鱼类有着本质区别。鲸鱼无法在水下直接呼吸氧气，因此它们需要耗费大量的体力，定时浮出水面完成一次快速而彻底的换气过程。令人惊奇的是，当它们处于睡眠状态时，为了确保不会因为忘记呼吸而发生危险，它们只会关闭大脑的一半来进行休息，另一半大脑则始终保持清醒和警觉，以便及时引导身体浮上水面。这种独特的生存机制是它们在深海中延续生命的关键。问题：鲸鱼与鱼类在呼吸方式上的根本区别是什么？它们在睡觉时会采取什么特殊的措施来保证安全和生存？"})["sequence"];
  // auto raw_input_tokens = qwen_tokenizer.convertMessage({.prompt = "提示:海洋世界里，鲸鱼是体型庞大的哺乳动物，它们通过喷水孔呼吸。与鱼类不同，鲸鱼无法在水下直接呼吸氧气。它们会定时浮出水面进行换气，每次换气需要消耗大量的体力。当它们睡觉时，只会关闭大脑的一半，另一半则保持清醒，以确保不忘记浮出水面呼吸。问题：鲸鱼与鱼类在呼吸方式上的根本区别是什么？它们在睡觉时会采取什么特殊的措施来保证安全？"})["sequence"];
  print(raw_input_tokens);
  MLLM_INFO("raw_input_tokens shape: {} {}", raw_input_tokens.shape()[0], raw_input_tokens.shape()[1]);

  const int chunk_size = 128;
  const int eos_token_id = 151645;
  int prompt_tokens = static_cast<int>(raw_input_tokens.shape()[1]);
  if (prompt_tokens <= 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kShapeError, "Prompt sequence length must be positive");
  }

  // Prepare reusable [1, chunk_size] CPU buffer for chunked prefill/decode
  mllm::models::ARGenerationOutputPast chunk_inputs{
      {"sequence", mllm::Tensor::empty({1, chunk_size}, mllm::kInt64, mllm::kCPU).alloc()}};
  auto sequence_tensor = chunk_inputs["sequence"];
  auto sequence_ptr = sequence_tensor.ptr<int64_t>();
  auto input_data = raw_input_tokens.ptr<int64_t>();

  const int prompt_chunks = (prompt_tokens + chunk_size - 1) / chunk_size;
  bool reached_eos = false;
  int total_decode_steps = 0;

  for (int chunk_index = 0; chunk_index < prompt_chunks && !reached_eos; ++chunk_index) {
    const int chunk_start = chunk_index * chunk_size;
    const int chunk_prompt_len = std::min(chunk_size, prompt_tokens - chunk_start);
    const bool is_last_prompt_chunk = (chunk_index == prompt_chunks - 1);

    // Copy current chunk prompt tokens and pad remaining positions with -1
    for (int i = 0; i < chunk_prompt_len; ++i) { sequence_ptr[i] = input_data[chunk_start + i]; }
    for (int i = chunk_prompt_len; i < chunk_size; ++i) { sequence_ptr[i] = -1; }

    // MLLM_INFO("=== Prefill Chunk {} ===", chunk_index);
    // MLLM_INFO("Chunk start: {}, Chunk prompt length: {}", chunk_start, chunk_prompt_len);
    
    // Calculate absolute sequence length from the start of the entire sequence
    const int absolute_seq_len = chunk_start + chunk_prompt_len;
    // MLLM_INFO("Absolute sequence length: {}", absolute_seq_len);

    // Align KV cache so StaticCache writes start at the chunk's absolute offset
    model.setKVCacheSeqCnt(chunk_start);
    // MLLM_INFO("KV cache seq_cnt set to: {}", chunk_start);

    // Generate position_ids starting from chunk_start for multi-chunk scenarios
    auto position_ids_tensor = mllm::Tensor::empty({1, chunk_size}, mllm::kInt64, mllm::kCPU).alloc();
    auto position_ids_ptr = position_ids_tensor.ptr<int64_t>();
    for (int i = 0; i < chunk_size; ++i) {
      position_ids_ptr[i] = chunk_start + i;
    }
    
    // Prepare input with correct position_ids
    mllm::models::ARGenerationOutputPast prefill_inputs{
        {"sequence", sequence_tensor},
        {"position_ids", position_ids_tensor}};

    // real_seq should be the effective length in the current input tensor (relative position)
    // hidden_states shape is [1, chunk_size, hidden_size], we need to index it with chunk_prompt_len - 1
    auto chunk_output =
        model.forward(prefill_inputs, {{"seq_len", mllm::AnyValue(mllm::any_copy_tag, chunk_prompt_len)}});
    auto& chunk_logits = chunk_output["sequence"];

    // auto tmp_next_token = model.sampleGreedy(chunk_logits);
    // std::wcout << qwen_tokenizer.detokenize(tmp_next_token) << "\n";
    // std::wcout << qwen_tokenizer.detokenize(sequence_ptr[chunk_start + chunk_prompt_len]) << "\n";

    if (!is_last_prompt_chunk) {
      // MLLM_INFO("Chunk {} processed as prompt only, moving to next chunk", chunk_index);
      chunk_logits.delete_();
      chunk_output.clear();
      continue;
    }

    if (chunk_prompt_len >= chunk_size) {
      MLLM_WARN("Last chunk is fully occupied by prompt tokens; no padding for decode");
      chunk_logits.delete_();
      chunk_output.clear();
      break;
    }

    // MLLM_INFO("=== Decode Phase (Chunk {}) ===", chunk_index);

    // Use the prefill logits as the first decode step
    auto next_token = model.sampleGreedy(chunk_logits);
    chunk_logits.delete_();
    
    // Keep full-length position_ids tensor aligned with chunk buffer
    auto position_ids = position_ids_tensor;

    chunk_output.clear();

    auto emit_token = [&](int64_t token_id) {
      std::wcout << qwen_tokenizer.detokenize(token_id) << std::flush;
      if (token_id == eos_token_id) {
        MLLM_INFO("EOS token detected, stopping decode");
        reached_eos = true;
      }
    };

    int current_chunk_len = chunk_prompt_len;
    emit_token(next_token);
    if (reached_eos) { break; }

    sequence_ptr[current_chunk_len] = next_token;
    current_chunk_len++;

    while (!reached_eos && current_chunk_len < chunk_size) {
      total_decode_steps++;
      
      // Calculate absolute sequence length from the start of the entire sequence
      const int absolute_seq_len = chunk_start + current_chunk_len;
      
      // MLLM_INFO("--- Chunk {} Decode Step {} ---", chunk_index, total_decode_steps);
      // MLLM_INFO("Current chunk length: {} (relative), Absolute sequence length: {} (absolute)", current_chunk_len, absolute_seq_len);

      // Keep padding clean for the remaining area
      for (int i = current_chunk_len; i < chunk_size; ++i) { sequence_ptr[i] = -1; }

      // Set KV cache to absolute sequence length (where the next token will be written)
      // [Maybe Wrong]
      model.setKVCacheSeqCnt(chunk_start);
      // MLLM_INFO("KV cache seq_cnt set to: {} (relative position)", chunk_start);
      
      // Prepare decode input with position_ids from previous step
      mllm::models::ARGenerationOutputPast decode_inputs{
          {"sequence", sequence_tensor},
          {"position_ids", position_ids}};
      
      // real_seq should be the effective length in the current input tensor (relative position)
      // hidden_states shape is [1, chunk_size, hidden_size], we need to index it with current_chunk_len - 1
      auto decode_output = model.forward(
          decode_inputs, {{"seq_len", mllm::AnyValue(mllm::any_copy_tag, current_chunk_len)}});
      
      auto& decode_logits = decode_output["sequence"];
      next_token = model.sampleGreedy(decode_logits);
      decode_logits.delete_();
      decode_output.erase("sequence");
      decode_output.clear();

      emit_token(next_token);
      if (reached_eos) { break; }

      sequence_ptr[current_chunk_len] = next_token;
      current_chunk_len++;
    }

    // MLLM_INFO("=== Chunk {} Decode Complete ===", chunk_index);
    // MLLM_INFO("Chunk final length: {}", current_chunk_len);
    // MLLM_INFO("Remaining capacity: {}", chunk_size - current_chunk_len);
  }

  std::wcout << L"\n";

  return 0;
})
