#include <fmt/core.h>
#include <vector>
#include <chrono>
#include <string>

#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>
#include <mllm/core/DataTypes.hpp>

#include "mllm/models/qwen_npu/tokenization_qwen.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu_cpu.hpp"  // CPU Qwen model for decode
#include "mllm/models/qwen_npu/modeling_qwen_npu.hpp"      // NPU Qwen model for prefill
#include "mllm/nn/lmcache/StaticCache.hpp"                 // For StaticCache type
#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/core/SlicePrimitives.hpp"  // for kAll
#include "mllm/utils/Log.hpp"

using mllm::Argparse;

int main(int argc, char** argv) {
#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif
  ::mllm::initializeContext();

  // Mixed inference: QNN prefill + CPU decode
  const bool use_npu_prefill = true;

  if (use_npu_prefill) {
    MLLM_INFO("Mixed inference mode: NPU prefill + CPU decode");
  } else {
    MLLM_INFO("Pure CPU inference mode");
  }

  const std::string config_path = "config_1.8B_w8a16_qnn.json";
  const std::string npu_model_path = "/data/local/tmp/zhanghao/models/qwen1.5-1.8b-chat-rot-qnn.mllm";
  const std::string cpu_decode_model_path = "/data/local/tmp/zhanghao/models/qwen1.5-1.8b-chat-rot_q4_0.mllm";

  auto qwen_tokenizer = mllm::models::qwen_npu::QwenTokenizer("tokenizer.json", "qwen_merges.txt");

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;

  auto cpu_cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path);

  std::unique_ptr<mllm::nn::StaticCache> shared_kv_cache = nullptr;
  std::unique_ptr<mllm::models::qwen_npu::QwenForCausalLM> npu_model;
  int32_t prefill_seq_len = 0;

  auto cpu_decode_model = mllm::models::qwen_npu::QwenForCausalLMCPU("", cpu_cfg);
  auto cpu_param = mllm::load(cpu_decode_model_path, file_version);
  cpu_decode_model.load(cpu_param);
  MLLM_INFO("CPU decode model loaded from: {}", cpu_decode_model_path);

  // Setup shared KV cache for QNN prefill + CPU decode
  if (use_npu_prefill) {
    MLLM_INFO("Loading QNN model for prefill...");
    mllm::initQnnBackend();
    auto npu_cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path);

    shared_kv_cache = std::make_unique<mllm::nn::StaticCache>(
        cpu_cfg.max_cache_length, cpu_cfg.num_hidden_layers, cpu_cfg.num_attention_heads, cpu_cfg.num_key_value_heads,
        cpu_cfg.head_dim, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
    MLLM_INFO("Created shared StaticCache with {} layers", cpu_cfg.num_hidden_layers);

    npu_model = std::make_unique<mllm::models::qwen_npu::QwenForCausalLM>("", npu_cfg);
    auto npu_param = mllm::load(npu_model_path, file_version);
    npu_model->load(npu_param);
    MLLM_INFO("QNN prefill model loaded from: {}", npu_model_path);

    // Configure QNN model KVCache layers to use shared StaticCache
    auto& npu_decode_blocks = npu_model->model.decode_blocks();
    for (int32_t layer_idx = 0; layer_idx < cpu_cfg.num_hidden_layers; ++layer_idx) {
      auto& npu_block = npu_decode_blocks.list()[layer_idx];
      auto& npu_kv_cache = npu_block.getKVCache();
      npu_kv_cache.setStaticCache(shared_kv_cache.get());
      npu_kv_cache.setLayerIndex(layer_idx);
    }
    MLLM_INFO("Configured {} QNN KVCache layers to use shared StaticCache", cpu_cfg.num_hidden_layers);

    // Configure CPU model KVCache layers to use shared StaticCache
    auto& cpu_decode_blocks = cpu_decode_model.text_model().decode_blocks();
    for (int32_t layer_idx = 0; layer_idx < cpu_cfg.num_hidden_layers; ++layer_idx) {
      auto& cpu_block = cpu_decode_blocks.list()[layer_idx];
      auto& cpu_kv_cache = cpu_block.getKVCache();
      cpu_kv_cache.setStaticCache(shared_kv_cache.get());
      cpu_kv_cache.setLayerIndex(layer_idx);
    }
    MLLM_INFO("Configured {} CPU KVCache layers to use shared StaticCache", cpu_cfg.num_hidden_layers);
  }

  auto raw_input_tokens = qwen_tokenizer.convertMessage(
      {.prompt =
           "提示:"
           "海洋世界里，鲸鱼是地球上体型最为庞大的哺乳动物，它们拥有流线型的身躯，主要通过头顶的喷水孔进行呼吸。与终生生活在水"
           "下并利用鱼鳃从水中提取溶解氧的鱼类有着本质区别。鲸鱼无法在水下直接呼吸氧气，因此它们需要耗费大量的体力，定时浮出水"
           "面完成一次快速而彻底的换气过程。令人惊奇的是，当它们处于睡眠状态时，为了确保不会因为忘记呼吸而发生危险，它们只会关"
           "闭大脑的一半来进行休息，另一半大脑则始终保持清醒和警觉，以便及时引导身体浮上水面。这种独特的生存机制是它们在深海中"
           "延续生命的关键。问题：鲸鱼与鱼类在呼吸方式上的根本区别是什么？它们在睡觉时会采取什么特殊的措施来保证安全和生存？"})
                              ["sequence"];
  MLLM_INFO("Input tokens: {} tokens", raw_input_tokens.shape()[1]);

  const int64_t eos_token_id = cpu_cfg.eos_token_id;
  const int max_new_tokens = 512;

  mllm::models::ARGenerationOutputPast past{{"sequence", raw_input_tokens}};
  mllm::models::ARGenerationArgs args;
  args["debug_layer_outputs"] = false;

  // Prefill phase
  if (!use_npu_prefill) {
    MLLM_INFO("Starting CPU prefill...");
    auto prefill_start = std::chrono::high_resolution_clock::now();

    prefill_seq_len = static_cast<int32_t>(raw_input_tokens.shape()[1]);
    mllm::models::ARGenerationArgs prefill_args;
    prefill_args["seq_len"] = prefill_seq_len;
    auto prefill_output = cpu_decode_model.forward(past, prefill_args);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start);
    MLLM_INFO("CPU prefill completed: seq_len={}, time={:.2f}s", prefill_seq_len, prefill_duration.count() / 1000.0);

    auto prefill_logits = prefill_output["sequence"];
    auto prefill_logits_f = prefill_logits.to(mllm::kFloat32);
    const auto& prefill_shape = prefill_logits_f.shape();

    if (prefill_shape.size() == 3 && prefill_shape[1] > 0) {
      auto last_logits = prefill_logits_f[{mllm::kAll, {prefill_shape[1] - 1}, mllm::kAll}];
      last_logits = last_logits.view({1, 1, prefill_shape[2]});

      const int64_t vocab_size = prefill_shape[2];
      auto* logits_data = last_logits.ptr<float>();
      int first_next_id = 0;
      float max_logit = logits_data[0];
      for (int64_t i = 1; i < vocab_size; ++i) {
        if (logits_data[i] > max_logit) {
          max_logit = logits_data[i];
          first_next_id = static_cast<int>(i);
        }
      }

      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(first_next_id);
      past["position_ids"] = prefill_output["position_ids"];

      auto first_token_str = qwen_tokenizer.detokenize(static_cast<int64_t>(first_next_id));
      std::wcout << first_token_str << std::flush;
    } else {
      auto last_token_idx = raw_input_tokens.shape()[1] - 1;
      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = raw_input_tokens.at<mllm::mllm_int64_t>({0, last_token_idx});
      past["position_ids"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["position_ids"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(prefill_seq_len);
    }
  }

  if (use_npu_prefill && npu_model) {
    MLLM_INFO("Starting QNN prefill...");
    auto prefill_start = std::chrono::high_resolution_clock::now();

    auto irs = npu_model->trace(past, {});

    mllm::ir::PassManager rewritePM(irs["model"]);
    rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
    rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
    rewritePM.run();

    mllm::ir::PassManager graphBuildPM(irs["model"]);
    graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
    graphBuildPM.run();

    npu_model->model.clearKVCache();

    prefill_seq_len = static_cast<int32_t>(raw_input_tokens.shape()[1]);
    mllm::models::ARGenerationArgs prefill_args;
    prefill_args["seq_len"] = prefill_seq_len;
    auto prefill_output = npu_model->forward(past, prefill_args);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start);
    MLLM_INFO("QNN prefill completed: seq_len={}, time={:.2f}s", prefill_seq_len, prefill_duration.count() / 1000.0);

    auto prefill_logits = prefill_output["sequence"];
    auto prefill_logits_f = prefill_logits.to(mllm::kFloat32);
    const auto& prefill_shape = prefill_logits_f.shape();

    if (prefill_shape.size() == 3 && prefill_shape[1] > 0) {
      auto last_logits = prefill_logits_f[{mllm::kAll, {prefill_shape[1] - 1}, mllm::kAll}];
      last_logits = last_logits.view({1, 1, prefill_shape[2]});

      const int64_t vocab_size = prefill_shape[2];
      auto* logits_data = last_logits.ptr<float>();
      int first_next_id = 0;
      float max_logit = logits_data[0];
      for (int64_t i = 1; i < vocab_size; ++i) {
        if (logits_data[i] > max_logit) {
          max_logit = logits_data[i];
          first_next_id = static_cast<int>(i);
        }
      }

      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(first_next_id);
      past["position_ids"] = prefill_output["position_ids"];

      auto first_token_str = qwen_tokenizer.detokenize(static_cast<int64_t>(first_next_id));
      std::wcout << first_token_str << std::flush;
    } else {
      auto last_token_idx = raw_input_tokens.shape()[1] - 1;
      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = raw_input_tokens.at<mllm::mllm_int64_t>({0, last_token_idx});
      past["position_ids"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["position_ids"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(prefill_seq_len);
    }
  }

  // Decode phase
  MLLM_INFO("Starting CPU decode...");
  int decode_start_step = 1;

  double total_decode_time_ms = 0.0;
  int decode_count = 0;

  for (int step = decode_start_step; step < max_new_tokens; ++step) {
    auto step_start = std::chrono::high_resolution_clock::now();
    auto output = cpu_decode_model.forward(past, args);
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
    double step_time_ms = step_duration.count() / 1000.0;

    total_decode_time_ms += step_time_ms;
    decode_count++;

    mllm::Tensor logits = output["sequence"];
    mllm::Tensor logits_f = logits.to(mllm::kFloat32);

    const auto& shape = logits_f.shape();
    if (shape.size() != 3) {
      MLLM_ERROR("Unexpected logits shape (expected [B,1,V])");
      break;
    }

    const int64_t vocab_size = shape[2];
    auto* data = logits_f.ptr<float>();

    int64_t next_id = 0;
    float max_logit = data[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
      if (data[i] > max_logit) {
        max_logit = data[i];
        next_id = i;
      }
    }
    auto next_token_str = qwen_tokenizer.detokenize(next_id);

    MLLM_INFO("[Decode step {:3d}] time: {:.2f}ms, token: ", step, step_time_ms);
    std::wcout << next_token_str << std::flush;

    past = std::move(output);
    past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, logits_f.device()).alloc();
    past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(next_id);

    if (next_id == eos_token_id) {
      MLLM_INFO("Hit EOS at step {}, stopping decode.", step);
      break;
    }
  }
  std::wcout << L"\n";
  if (decode_count > 0) {
    double avg_time_ms = total_decode_time_ms / decode_count;
    MLLM_INFO(
        "Decode completed: {} tokens, total time: {:.2f}ms ({:.2f}s), avg time: {:.2f}ms/token, throughput: {:.2f} tokens/s",
        decode_count, total_decode_time_ms, total_decode_time_ms / 1000.0, avg_time_ms,
        decode_count / (total_decode_time_ms / 1000.0));
  }
#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen_npu.perfetto");
#endif

  return 0;
}