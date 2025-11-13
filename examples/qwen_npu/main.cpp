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

  auto raw_input_tokens = qwen_tokenizer.convertMessage({.prompt = "How are you?"})["sequence"];
  print(raw_input_tokens);
  MLLM_INFO("raw_input_tokens shape: {} {}", raw_input_tokens.shape()[0], raw_input_tokens.shape()[1]);

  const int chunk_size = 128;
  int real_seq = static_cast<int>(raw_input_tokens.shape()[1]);
  const int eos_token_id = 151645;
  if (real_seq <= 0 || real_seq >= chunk_size) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kShapeError, "Invalid input length {} for chunk size {}", real_seq, chunk_size);
  }

  // manually set input data as fill op is not supported in QNN
  auto ptr = inputs["sequence"].ptr<int64_t>();
  auto input_data = raw_input_tokens.ptr<int64_t>();
  for (int i = 0; i < real_seq; ++i) { ptr[i] = input_data[i]; }
  for (int i = real_seq; i < chunk_size; ++i) { ptr[i] = -1; }

  // Prefill
  MLLM_INFO("=== Prefill Phase ===");
  MLLM_INFO("Input sequence length: {}", real_seq);
  auto prefill_output = model.forward(inputs, {{"seq_len", mllm::AnyValue(real_seq)}});
  auto& prefill_logits = prefill_output["sequence"];
  auto sampled = model.sampleGreedy(prefill_logits);
  prefill_logits.delete_();
  prefill_output.erase("sequence");
  MLLM_INFO("Prefill generated token id: {}", sampled);
  std::wcout << qwen_tokenizer.detokenize(sampled);

  // Decode loop
  int current_seq_len = real_seq;
  auto& sequence_tensor = inputs["sequence"];
  auto sequence_ptr = sequence_tensor.ptr<int64_t>();

  // write first token into padding
  sequence_ptr[current_seq_len] = sampled;
  current_seq_len++;

  // carry past (position_ids) from prefill
  mllm::models::ARGenerationOutputPast past{{"position_ids", prefill_output["position_ids"]}};
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
  mllm::Tensor decode_token_tensor = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
  mllm::models::ARGenerationOutputPast decode_input{
      {"sequence", decode_token_tensor},
      {"position_ids", past["position_ids"]},
  };

  while (current_seq_len < chunk_size) {
    decode_step++;
    MLLM_INFO("--- Decode Step {} ---", decode_step);
    MLLM_INFO("Current sequence length: {}", current_seq_len);

    // update KV cache sequence length across layers
    model.setKVCacheSeqCnt(current_seq_len);

    // reuse a single CPU tensor for decode token to avoid repeated QNN allocations
    decode_token_tensor.ptr<int64_t>()[0] = sequence_ptr[current_seq_len - 1];

    // pass through latest position ids returned from previous forward call
    decode_input["position_ids"] = past["position_ids"];
    MLLM_INFO("Decode input token: {}", sequence_ptr[current_seq_len - 1]);

    // forward for next token logits
    auto decode_output = model.forward(decode_input, {{"seq_len", mllm::AnyValue(current_seq_len)}});

    // sample next token
    auto& decode_logits = decode_output["sequence"];
    auto next_token = model.sampleGreedy(decode_logits);
    MLLM_INFO("Generated token id: {}", next_token);
    std::wcout << qwen_tokenizer.detokenize(next_token);

    if (next_token == eos_token_id) {
      MLLM_INFO("EOS token detected, stopping decode");
      break;
    }

    // write token into sequence buffer
    sequence_ptr[current_seq_len] = next_token;
    current_seq_len++;
    MLLM_INFO("Updated sequence length: {}", current_seq_len);

    // carry past (only keep position_ids to avoid leaking QNN buffers)
    decode_logits.delete_();
    decode_output.erase("sequence");
    auto position_ids = decode_output["position_ids"];
    decode_output.erase("position_ids");
    past = {{"position_ids", position_ids}};
    decode_input["position_ids"] = past["position_ids"];
  }

  MLLM_INFO("=== Decode Complete ===");
  MLLM_INFO("Total decode steps: {}", decode_step);
  MLLM_INFO("Final sequence length: {}", current_seq_len);
  MLLM_INFO("Remaining capacity: {}", chunk_size - current_seq_len);
  std::wcout << L"\n";

  return 0;
})
