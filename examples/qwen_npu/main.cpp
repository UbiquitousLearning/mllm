#include <fmt/core.h>
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>
#include <mllm/core/DataTypes.hpp>

#include "mllm/models/qwen_npu/tokenization_qwen.hpp"
#include "mllm/models/qwen_npu/ShadowNPUHeadSchedule.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu_cpu.hpp"  // CPU Qwen model for decode
#include "mllm/models/qwen_npu/modeling_qwen_npu.hpp"      // NPU Qwen model for prefill
#include "mllm/nn/lmcache/StaticCache.hpp"                 // For StaticCache type
#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/cpu/kernels/common/shadownpu/AttentionProfiler.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/core/SlicePrimitives.hpp"  // for kAll
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Log.hpp"

using mllm::Argparse;

namespace {

std::string readPromptFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) { throw std::invalid_argument("cannot open prompt file: " + path); }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (contents.str().empty()) { throw std::invalid_argument("prompt file is empty: " + path); }
  return contents.str();
}

mllm::Tensor readPromptTokens(const std::string& path, int64_t vocabulary_size, int32_t capacity) {
  std::ifstream input(path);
  if (!input) { throw std::invalid_argument("cannot open prompt token ID file: " + path); }
  if (capacity <= 0) { throw std::invalid_argument("prompt token capacity must be positive"); }
  std::vector<int64_t> tokens;
  std::string value;
  while (input >> value) {
    int64_t token = -1;
    const auto end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, token);
    if (parsed.ec != std::errc{} || parsed.ptr != end || token < 0 || token >= vocabulary_size) {
      throw std::invalid_argument("invalid token ID in prompt file: " + path);
    }
    if (tokens.size() >= static_cast<size_t>(capacity)) {
      throw std::invalid_argument("prompt token ID file exceeds cache capacity: " + path);
    }
    tokens.push_back(token);
  }
  if (input.bad() || tokens.empty()) {
    throw std::invalid_argument("empty or unreadable prompt token ID file: " + path);
  }
  auto result = mllm::Tensor::empty({1, static_cast<int32_t>(tokens.size())}, mllm::kInt64, mllm::kCPU).alloc();
  std::copy(tokens.begin(), tokens.end(), result.ptr<int64_t>());
  return result;
}

std::string jsonEscape(const std::string& input) {
  std::ostringstream escaped;
  for (const unsigned char character : input) {
    switch (character) {
      case '\\': escaped << "\\\\"; break;
      case '"': escaped << "\\\""; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (character < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character) << std::dec
                  << std::setfill(' ');
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  return escaped.str();
}

std::string answerBeforeEndMarker(const std::string& text) {
  size_t end = text.size();
  for (const std::string marker : {std::string("\0", 1), std::string("<|im_end|>"), std::string("<|endoftext|>")}) {
    const size_t position = text.find(marker);
    if (position != std::string::npos) { end = std::min(end, position); }
  }
  return text.substr(0, end);
}

std::string extractAccessCode(const std::string& text) {
  static const std::regex code_pattern(R"([A-Z]{3}-[0-9]{4})");
  std::smatch match;
  return std::regex_search(text, match, code_pattern) ? match.str() : std::string();
}

uint64_t hashPromptTokens(const mllm::Tensor& tokens) {
  MLLM_RT_ASSERT(tokens.shape().size() == 2 && tokens.shape()[0] == 1);
  uint64_t hash = 1469598103934665603ULL;
  for (int32_t index = 0; index < tokens.shape()[1]; ++index) {
    uint32_t value = static_cast<uint32_t>(tokens.constAt<mllm::mllm_int64_t>({0, index}));
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<unsigned char>(value & 0xffU);
      hash *= 1099511628211ULL;
      value >>= 8;
    }
  }
  return hash;
}

void printLogitSummary(const char* phase, const float* values, int64_t count) {
  int64_t finite_count = 0;
  int64_t nan_count = 0;
  int64_t positive_infinity_count = 0;
  int64_t negative_infinity_count = 0;
  int64_t finite_argmax = -1;
  float finite_min = std::numeric_limits<float>::infinity();
  float finite_max = -std::numeric_limits<float>::infinity();
  const uint32_t first_bits = count > 0 ? std::bit_cast<uint32_t>(values[0]) : 0;
  for (int64_t index = 0; index < count; ++index) {
    const float value = values[index];
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000U) == 0x7f800000U) {
      if ((bits & 0x007fffffU) != 0) {
        ++nan_count;
      } else if ((bits & 0x80000000U) == 0) {
        ++positive_infinity_count;
      } else {
        ++negative_infinity_count;
      }
    } else {
      ++finite_count;
      if (finite_argmax < 0 || value > finite_max) {
        finite_max = value;
        finite_argmax = index;
      }
      if (value < finite_min) { finite_min = value; }
    }
  }
  fmt::print("SHADOWNPU_LOGITS phase={} vocab={} finite={} nan={} pos_inf={} neg_inf={} finite_min={} finite_max={} "
             "finite_argmax={} first_bits={:#010x}\n",
             phase, count, finite_count, nan_count, positive_infinity_count, negative_infinity_count, finite_min, finite_max,
             finite_argmax, first_bits);
}

void printLayerSummaries(const char* prefix, const std::vector<mllm::Tensor>& tensors, bool include_last_token) {
  for (size_t index = 0; index < tensors.size(); ++index) {
    auto tensor = tensors[index];
    tensor = tensor.to(mllm::kFloat32).contiguous();
    std::string phase;
    if (index == 0) {
      phase = fmt::format("{}_embedding", prefix);
    } else if (index + 1 == tensors.size()) {
      phase = fmt::format("{}_final_norm", prefix);
    } else {
      phase = fmt::format("{}_layer_{:02d}", prefix, index - 1);
    }
    printLogitSummary(phase.c_str(), tensor.ptr<float>(), static_cast<int64_t>(tensor.numel()));
    if (include_last_token) {
      const auto& shape = tensor.shape();
      mllm::Tensor last_token;
      if (shape.size() == 3 && shape[1] > 0) {
        last_token = tensor[{mllm::kAll, {shape[1] - 1}, mllm::kAll}].contiguous();
      } else if (shape.size() == 4 && shape[2] > 0) {
        last_token = tensor[{mllm::kAll, mllm::kAll, {shape[2] - 1}, mllm::kAll}].contiguous();
      }
      if (!last_token.isNil()) {
        const auto last_token_phase = phase + "_last_token";
        printLogitSummary(last_token_phase.c_str(), last_token.ptr<float>(), static_cast<int64_t>(last_token.numel()));
      }
    }
  }
}

void printLayerBoundarySummaries(size_t layer_index, const std::vector<mllm::Tensor>& tensors) {
  const std::vector<std::string> phases = {
      fmt::format("qnn_layer{}_q_proj", layer_index),
      fmt::format("qnn_layer{}_k_proj", layer_index),
      fmt::format("qnn_layer{}_v_proj", layer_index),
      fmt::format("qnn_layer{}_attention_output", layer_index),
  };
  MLLM_RT_ASSERT(tensors.size() == phases.size());
  for (size_t index = 0; index < tensors.size(); ++index) {
    auto tensor = tensors[index];
    tensor = tensor.to(mllm::kFloat32).contiguous();
    printLogitSummary(phases[index].c_str(), tensor.ptr<float>(), static_cast<int64_t>(tensor.numel()));
    const auto& shape = tensor.shape();
    mllm::Tensor last_token;
    if (shape.size() == 3 && shape[1] > 0) {
      last_token = tensor[{mllm::kAll, {shape[1] - 1}, mllm::kAll}].contiguous();
    } else if (shape.size() == 4 && shape[2] > 0) {
      last_token = tensor[{mllm::kAll, mllm::kAll, {shape[2] - 1}, mllm::kAll}].contiguous();
    }
    if (!last_token.isNil()) {
      const auto last_token_phase = phases[index] + "_last_token";
      printLogitSummary(last_token_phase.c_str(), last_token.ptr<float>(), static_cast<int64_t>(last_token.numel()));
    }
  }
}

std::string shapeString(const mllm::Tensor::shape_t& shape) {
  std::ostringstream output;
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) { output << 'x'; }
    output << shape[index];
  }
  return output.str();
}

void dumpQNNBoundaryTensors(const std::filesystem::path& root,
                            mllm::models::qwen_npu::QwenForCausalLM& model) {
  std::filesystem::create_directories(root);
  std::ofstream metadata(root / "metadata.tsv", std::ios::out | std::ios::trunc);
  if (!metadata) { throw std::runtime_error("cannot create QNN boundary metadata under " + root.string()); }
  metadata << "layer\tboundary\tshape\tnumel\tbytes\tfile\n";

  const auto dump_tensor = [&](const std::string& layer, const std::string& boundary, const std::string& filename,
                               mllm::Tensor tensor) {
    tensor = tensor.to(mllm::kFloat32).contiguous();
    const auto path = root / filename;
    std::ofstream output(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot create QNN boundary dump " + path.string()); }
    const auto bytes = static_cast<std::streamsize>(tensor.numel() * sizeof(float));
    output.write(reinterpret_cast<const char*>(tensor.ptr<float>()), bytes);
    if (!output) { throw std::runtime_error("failed to write QNN boundary dump " + path.string()); }
    metadata << layer << '\t' << boundary << '\t' << shapeString(tensor.shape()) << '\t' << tensor.numel() << '\t' << bytes
             << '\t' << filename << '\n';
    fmt::print("SHADOWNPU_BOUNDARY_DUMP layer={} boundary={} shape={} numel={} bytes={} path={}\n", layer, boundary,
               shapeString(tensor.shape()), tensor.numel(), bytes, path.string());
  };

  const std::vector<std::string> boundary_names = {"q_proj", "k_proj", "v_proj", "attention_output"};
  for (size_t layer_index = 0; layer_index < model.capturedBoundaryLayerCount(); ++layer_index) {
    const auto& boundaries = model.capturedLayerBoundaries(layer_index);
    MLLM_RT_ASSERT(boundaries.size() == boundary_names.size());
    for (size_t boundary_index = 0; boundary_index < boundaries.size(); ++boundary_index) {
      const auto filename = fmt::format("layer-{:02d}-{}.f32", layer_index, boundary_names[boundary_index]);
      dump_tensor(std::to_string(layer_index), boundary_names[boundary_index], filename, boundaries[boundary_index]);
    }
  }

  const auto& layer_outputs = model.capturedLayerOutputs();
  MLLM_RT_ASSERT(layer_outputs.size() == model.capturedBoundaryLayerCount() + 2);
  dump_tensor("-", "embedding", "embedding.f32", layer_outputs.front());
  for (size_t output_index = 1; output_index + 2 < layer_outputs.size(); ++output_index) {
    const auto layer_index = output_index - 1;
    dump_tensor(std::to_string(layer_index), "residual", fmt::format("layer-{:02d}-residual.f32", layer_index),
                layer_outputs[output_index]);
  }
  dump_tensor("-", "final_norm", "final_norm.f32", layer_outputs.back());
}

}  // namespace

int main(int argc, char** argv) {
#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif
  ::mllm::initializeContext();

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Qwen runtime config JSON");
  auto& npu_model_path =
      Argparse::add<std::string>("--npu-model|--qnn-model").help("QNN prefill model file (required unless --cpu-prefill)");
  auto& cpu_model_path = Argparse::add<std::string>("--cpu-model|--decoding-model").help("CPU decode model file");
  auto& tokenizer_path =
      Argparse::add<std::string>("--tokenizer|--vocab").help("Qwen tokenizer JSON or converted vocabulary file");
  auto& merges_path = Argparse::add<std::string>("--merges|--merge").help("Qwen BPE merges file");
  auto& model_version = Argparse::add<std::string>("-mv|--model-version").help("Model file version: v1 or v2");
  auto& attention_mode =
      Argparse::add<std::string>("--attention-mode")
          .help("Prefill attention mode: dense, f32-topk, or hmx-topk")
          .def("dense");
  auto& head_profile =
      Argparse::add<std::string>("--head-retain-profile")
          .help("Exact layer-major per-head retention profile for f32-topk or hmx-topk");
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Prompt text");
  auto& prompt_file = Argparse::add<std::string>("--prompt-file").help("Read the prompt from a file");
  auto& prompt_ids = Argparse::add<std::string>("--prompt-token-ids-file")
                         .help("Read prepared prompt token IDs in decimal, separated by whitespace");
  auto& max_new_tokens = Argparse::add<int32_t>("-g|--max-new-tokens").help("Maximum generated tokens").def(8);
  auto& expected_prompt_tokens =
      Argparse::add<int32_t>("--expected-prompt-tokens").help("Fail if tokenized prompt length differs");
  auto& expected_prompt_token_fnv64 =
      Argparse::add<std::string>("--expected-prompt-token-fnv64")
          .help("Fail unless PR #703's FNV-1a hash of little-endian Uint32 token IDs matches");
  auto& expected_access_code = Argparse::add<std::string>("--expected-access-code")
                                   .help("Report PR #703 strict retrieval for an [A-Z]{3}-[0-9]{4} access code")
                                   .def("");
  auto& prefill_chunk =
      Argparse::add<int32_t>("--prefill-chunk").help("Logical QNN prefill chunk size; 0 runs one full prompt").def(0);
  auto& qnn_sequence_tile =
      Argparse::add<int32_t>("--qnn-sequence-tile").help("QNN graph sequence tile compiled by trace; 0 disables").def(0);
  auto& qnn_context_path =
      Argparse::add<std::string>("--qnn-context").help("Load an existing QNN context binary before graph binding");
  auto& prepare_qnn_context = Argparse::add<std::string>("--prepare-qnn-context")
                                  .help("Build a prepare-only QNN context, save it to this new path, and exit");
  auto& qnn_qkv_bias_after_dequant = Argparse::add<bool>("--qnn-qkv-bias-after-dequant")
                                       .help("Use pre-bias INT16 ranges and add Q/K/V biases after dequantization");
  auto& cpu_prefill = Argparse::add<bool>("--cpu-prefill").help("Use CPU for prefill instead of QNN");
  auto& prefill_only = Argparse::add<bool>("--prefill-only")
                           .help("Run QNN prefill and report its first token without loading the CPU decode model");
  auto& debug_qnn_layer_outputs =
      Argparse::add<bool>("--debug-qnn-layer-outputs").help("Copy and summarize every QNN decoder residual output");
  auto& debug_qnn_boundary_dir = Argparse::add<std::string>("--debug-qnn-boundary-dir")
                                      .help("Dump every QNN layer Q/K/V and CPU attention output as raw Float32");
  auto& dump_qnn_prefill_logits = Argparse::add<std::string>("--dump-qnn-prefill-logits")
                                      .help("Write the last prompt token's QNN prefill logits as raw Float32 to this file");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    ::mllm::shutdownContext();
    return 0;
  }

  const bool use_npu_prefill = !cpu_prefill.isSet();
  const bool prepare_context = prepare_qnn_context.isSet();
  const bool run_prefill_only = prefill_only.isSet();
  if (dump_qnn_prefill_logits.isSet() && (!use_npu_prefill || prepare_context)) {
    fmt::print(stderr, "Error: --dump-qnn-prefill-logits requires a QNN prefill run.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (!use_npu_prefill && qnn_qkv_bias_after_dequant.isSet()) {
    fmt::print(stderr, "Error: --qnn-qkv-bias-after-dequant requires QNN prefill.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  const auto missing_required = [&]() {
    return !config_path.isSet() || (!run_prefill_only && !cpu_model_path.isSet()) || !tokenizer_path.isSet()
           || !merges_path.isSet() || !model_version.isSet() || (use_npu_prefill && !npu_model_path.isSet());
  };
  if (missing_required()) {
    fmt::print(stderr, "Error: --config, --tokenizer, --merges, and --model-version are required; --cpu-model is required "
                       "unless --prefill-only is set, and --npu-model is required unless --cpu-prefill is set.\n");
    Argparse::printHelp();
    ::mllm::shutdownContext();
    return 2;
  }
  const int prompt_sources = static_cast<int>(prompt.isSet()) + static_cast<int>(prompt_file.isSet())
                             + static_cast<int>(prompt_ids.isSet());
  if (prompt_sources != 1) {
    fmt::print(stderr, "Error: set exactly one of --prompt, --prompt-file, and --prompt-token-ids-file.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (prepare_context && !use_npu_prefill) {
    fmt::print(stderr, "Error: --prepare-qnn-context requires QNN prefill.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (run_prefill_only && !use_npu_prefill) {
    fmt::print(stderr, "Error: --prefill-only requires QNN prefill.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (prepare_context && qnn_context_path.isSet()) {
    fmt::print(stderr, "Error: --prepare-qnn-context and --qnn-context are mutually exclusive.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (qnn_context_path.isSet() && !std::filesystem::is_regular_file(qnn_context_path.get())) {
    fmt::print(stderr, "Error: --qnn-context must name an existing regular file: {}.\n", qnn_context_path.get());
    ::mllm::shutdownContext();
    return 2;
  }
  if (prepare_context && std::filesystem::exists(prepare_qnn_context.get())) {
    fmt::print(stderr, "Error: refusing to overwrite existing QNN context output: {}.\n", prepare_qnn_context.get());
    ::mllm::shutdownContext();
    return 2;
  }
  if (max_new_tokens.get() <= 0) {
    fmt::print(stderr, "Error: --max-new-tokens must be positive.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (expected_prompt_tokens.isSet() && expected_prompt_tokens.get() <= 0) {
    fmt::print(stderr, "Error: --expected-prompt-tokens must be positive.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  std::optional<uint64_t> expected_prompt_hash;
  if (expected_prompt_token_fnv64.isSet()) {
    try {
      size_t consumed = 0;
      expected_prompt_hash = std::stoull(expected_prompt_token_fnv64.get(), &consumed, 0);
      if (consumed != expected_prompt_token_fnv64.get().size()) { throw std::invalid_argument("trailing characters"); }
    } catch (const std::exception&) {
      fmt::print(stderr, "Error: --expected-prompt-token-fnv64 must be a Uint64 decimal or 0x-prefixed value.\n");
      ::mllm::shutdownContext();
      return 2;
    }
  }
  if (!expected_access_code.get().empty()
      && !std::regex_match(expected_access_code.get(), std::regex(R"([A-Z]{3}-[0-9]{4})"))) {
    fmt::print(stderr, "Error: --expected-access-code must match [A-Z]{{3}}-[0-9]{{4}}.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (prefill_chunk.get() < 0 || qnn_sequence_tile.get() < 0) {
    fmt::print(stderr, "Error: --prefill-chunk and --qnn-sequence-tile cannot be negative.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (!use_npu_prefill && (prefill_chunk.get() != 0 || qnn_sequence_tile.get() != 0)) {
    fmt::print(stderr, "Error: QNN prefill chunking/tiling cannot be used with --cpu-prefill.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (qnn_sequence_tile.get() > 0
      && (prefill_chunk.get() <= qnn_sequence_tile.get() || prefill_chunk.get() % qnn_sequence_tile.get() != 0)) {
    fmt::print(stderr, "Error: --prefill-chunk must be an exact, larger multiple of --qnn-sequence-tile.\n");
    ::mllm::shutdownContext();
    return 2;
  }

  mllm::ModelFileVersion file_version;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  } else {
    fmt::print(stderr, "Error: unsupported --model-version '{}'; expected v1 or v2.\n", model_version.get());
    ::mllm::shutdownContext();
    return 2;
  }

  const bool f32_topk = attention_mode.get() == "f32-topk";
  const bool hmx_topk = attention_mode.get() == "hmx-topk";
  const bool sparse_topk = f32_topk || hmx_topk;
  if (attention_mode.get() != "dense" && !sparse_topk) {
    fmt::print(stderr, "Error: unsupported --attention-mode '{}'; expected dense, f32-topk, or hmx-topk.\n",
               attention_mode.get());
    ::mllm::shutdownContext();
    return 2;
  }
  if (sparse_topk && (!use_npu_prefill || !head_profile.isSet())) {
    fmt::print(stderr, "Error: {} requires QNN prefill and --head-retain-profile.\n", attention_mode.get());
    ::mllm::shutdownContext();
    return 2;
  }
  if (attention_mode.get() == "dense" && head_profile.isSet()) {
    fmt::print(stderr, "Error: --head-retain-profile is only valid with --attention-mode f32-topk or hmx-topk.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (sparse_topk && (prefill_chunk.get() != 512 || qnn_sequence_tile.get() != 256)) {
    fmt::print(stderr, "Error: the packaged Qwen2.5 sparse-attention contract requires --prefill-chunk 512 "
                       "and --qnn-sequence-tile 256.\n");
    ::mllm::shutdownContext();
    return 2;
  }
  if (sparse_topk && !prepare_context && !qnn_context_path.isSet()) {
    fmt::print(stderr, "Error: {} execution requires an existing --qnn-context generated for the M256 trace.\n",
               attention_mode.get());
    ::mllm::shutdownContext();
    return 2;
  }

  if (run_prefill_only) {
    MLLM_INFO("NPU prefill-only diagnostic mode");
  } else if (use_npu_prefill) {
    MLLM_INFO("Mixed inference mode: NPU prefill + CPU decode");
  } else {
    MLLM_INFO("Pure CPU inference mode");
  }

  auto cpu_cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path.get());
  cpu_cfg.qnn_qkv_bias_after_dequant = qnn_qkv_bias_after_dequant.isSet();
  if (cpu_cfg.qnn_cross_layer_graphs) {
    if (!use_npu_prefill || !cpu_cfg.qnn_qkv_bias_after_dequant) {
      fmt::print(stderr, "Error: cross-layer QNN graphs require QNN prefill with --qnn-qkv-bias-after-dequant.\n");
      ::mllm::shutdownContext();
      return 2;
    }
    const auto shadow_boundaries = std::count_if(cpu_cfg.shadow_down_proj_layers.begin(), cpu_cfg.shadow_down_proj_layers.end(),
                                                 [&](int32_t layer) { return layer + 1 < cpu_cfg.num_hidden_layers; });
    fmt::print("SHADOWNPU_QNN_CROSS_LAYER_GRAPHS layers={} graph_count={}\n", cpu_cfg.num_hidden_layers,
               cpu_cfg.num_hidden_layers + 1 + shadow_boundaries);
  }
  if (cpu_cfg.qnn_qkv_bias_after_dequant) {
    fmt::print("SHADOWNPU_QKV_BIAS_ORDER mode=after_dequantization\n");
  }
  float average_retention = 1.0F;
  int32_t dense_heads = cpu_cfg.num_hidden_layers * cpu_cfg.num_attention_heads;
  std::string dense_layers = "all";
  if (sparse_topk) {
    const auto schedule = mllm::models::qwen_npu::loadShadowNPUHeadSchedule(head_profile.get(), cpu_cfg.num_hidden_layers,
                                                                            cpu_cfg.num_attention_heads);
    cpu_cfg.prefill_attention_sparsity = 1.0F - schedule.average_retention;
    cpu_cfg.prefill_attention_pattern_sparse = hmx_topk;
    cpu_cfg.prefill_attention_hmx_selector = hmx_topk;
    cpu_cfg.prefill_attention_layer_sparsities.clear();
    cpu_cfg.prefill_attention_head_retentions = schedule.retentions;
    average_retention = schedule.average_retention;
    dense_heads = schedule.dense_heads;
    dense_layers = schedule.normalized_dense_layers;
  } else {
    cpu_cfg.prefill_attention_sparsity = 0.0F;
    cpu_cfg.prefill_attention_pattern_sparse = false;
    cpu_cfg.prefill_attention_hmx_selector = false;
    cpu_cfg.prefill_attention_layer_sparsities.clear();
    cpu_cfg.prefill_attention_head_retentions.clear();
  }

  const std::string prompt_text =
      prompt_ids.isSet() ? std::string{} : (prompt.isSet() ? prompt.get() : readPromptFile(prompt_file.get()));
  auto qwen_tokenizer = mllm::models::qwen_npu::QwenTokenizer(tokenizer_path.get(), merges_path.get());

  const std::string context_mode = prepare_context ? "prepare" : (qnn_context_path.isSet() ? "load" : "live-build");
  const std::string context_receipt =
      prepare_context ? prepare_qnn_context.get() : (qnn_context_path.isSet() ? qnn_context_path.get() : "none");
  fmt::print("SHADOWNPU_RUN_CONFIG attention_mode={} prefill_backend={} prefill_only={} model_version={} config={} "
             "npu_model={} cpu_model={} "
             "tokenizer={} merges={} head_profile={} average_retention={:.8f} dense_heads={} dense_layers={} max_new_tokens={} "
             "prefill_chunk={} qnn_sequence_tile={} qnn_context={} qnn_context_mode={}\n",
             attention_mode.get(), use_npu_prefill ? "qnn" : "cpu", run_prefill_only, model_version.get(), config_path.get(),
             use_npu_prefill ? npu_model_path.get() : "none", cpu_model_path.isSet() ? cpu_model_path.get() : "none",
             tokenizer_path.get(), merges_path.get(), head_profile.isSet() ? head_profile.get() : "none", average_retention,
             dense_heads, dense_layers, max_new_tokens.get(), prefill_chunk.get(), qnn_sequence_tile.get(), context_receipt,
             context_mode);

  std::unique_ptr<mllm::nn::StaticCache> shared_kv_cache = nullptr;
  std::unique_ptr<mllm::models::qwen_npu::QwenForCausalLM> npu_model;
  std::shared_ptr<mllm::qnn::QNNBackend> qnn_backend;
  int32_t prefill_seq_len = 0;
  int32_t prefill_chunk_count = 1;

  std::optional<mllm::models::qwen_npu::QwenForCausalLMCPU> cpu_decode_model;
  if (!run_prefill_only) {
    cpu_decode_model.emplace("", cpu_cfg);
    auto cpu_param = mllm::load(cpu_model_path.get(), file_version);
    cpu_decode_model->load(cpu_param);
    MLLM_INFO("CPU decode model loaded from: {}", cpu_model_path.get());
  }

  // Setup shared KV cache for QNN prefill + CPU decode
  if (use_npu_prefill) {
    MLLM_INFO("Loading QNN model for prefill...");
    mllm::initQnnBackend(qnn_context_path.isSet() ? qnn_context_path.get() : std::string(), prepare_context);
    qnn_backend = std::dynamic_pointer_cast<mllm::qnn::QNNBackend>(mllm::Context::instance().getBackend(mllm::kQNN));
    if (!qnn_backend) { throw std::runtime_error("QNN backend initialization did not register QNNBackend"); }
    qnn_backend->setSequenceTile(qnn_sequence_tile.get());

    shared_kv_cache = std::make_unique<mllm::nn::StaticCache>(
        cpu_cfg.max_cache_length, cpu_cfg.num_hidden_layers, cpu_cfg.num_attention_heads, cpu_cfg.num_key_value_heads,
        cpu_cfg.head_dim, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
    MLLM_INFO("Created shared StaticCache with {} layers (K=F32, V=F32; HMX keys use per-call F16 staging)",
              cpu_cfg.num_hidden_layers);

    npu_model = std::make_unique<mllm::models::qwen_npu::QwenForCausalLM>("", cpu_cfg);
    auto npu_param = mllm::load(npu_model_path.get(), file_version);
    npu_model->load(npu_param);
    MLLM_INFO("QNN prefill model loaded from: {}", npu_model_path.get());

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
    if (cpu_decode_model) {
      auto& cpu_decode_blocks = cpu_decode_model->text_model().decode_blocks();
      for (int32_t layer_idx = 0; layer_idx < cpu_cfg.num_hidden_layers; ++layer_idx) {
        auto& cpu_block = cpu_decode_blocks.list()[layer_idx];
        auto& cpu_kv_cache = cpu_block.getKVCache();
        cpu_kv_cache.setStaticCache(shared_kv_cache.get());
        cpu_kv_cache.setLayerIndex(layer_idx);
      }
      MLLM_INFO("Configured {} CPU KVCache layers to use shared StaticCache", cpu_cfg.num_hidden_layers);
    }
  }

  auto raw_input_tokens = prompt_ids.isSet()
                              ? readPromptTokens(prompt_ids.get(), cpu_cfg.vocab_size, cpu_cfg.max_cache_length)
                              : qwen_tokenizer.convertMessage({.prompt = prompt_text})["sequence"];
  MLLM_INFO("Input tokens: {} tokens", raw_input_tokens.shape()[1]);
  if (expected_prompt_tokens.isSet() && raw_input_tokens.shape()[1] != expected_prompt_tokens.get()) {
    fmt::print(stderr, "Error: tokenized prompt length {} does not match expected {}.\n", raw_input_tokens.shape()[1],
               expected_prompt_tokens.get());
    ::mllm::shutdownContext();
    return 2;
  }
  const uint64_t prompt_token_hash = hashPromptTokens(raw_input_tokens);
  fmt::print("SHADOWNPU_INPUT_TOKEN_HASH algorithm=fnv1a64-le-u32 value={} hex={:#018x}\n", prompt_token_hash,
             prompt_token_hash);
  if (expected_prompt_hash && prompt_token_hash != *expected_prompt_hash) {
    fmt::print(stderr, "Error: prompt token FNV-1a hash {} ({:#018x}) does not match expected {} ({:#018x}).\n",
               prompt_token_hash, prompt_token_hash, *expected_prompt_hash, *expected_prompt_hash);
    ::mllm::shutdownContext();
    return 2;
  }
  if (use_npu_prefill && prefill_chunk.get() > 0 && raw_input_tokens.shape()[1] % prefill_chunk.get() != 0) {
    fmt::print(stderr, "Error: tokenized prompt length {} is not an exact multiple of --prefill-chunk {}.\n",
               raw_input_tokens.shape()[1], prefill_chunk.get());
    ::mllm::shutdownContext();
    return 2;
  }

  const int64_t eos_token_id = cpu_cfg.eos_token_id;
  const int max_new_tokens_value = max_new_tokens.get();
  double prefill_time_ms = 0.0;
  std::vector<int64_t> generated_token_ids;
  std::string generated_text;
  bool prefill_hit_eos = false;
  bool logits_dump_failed = false;

  mllm::models::ARGenerationOutputPast past{{"sequence", raw_input_tokens}};
  mllm::models::ARGenerationArgs args;
  args["debug_layer_outputs"] = true;
  args["return_hidden_states"] = true;

  // Give the pure CPU path an explicitly sized cache. The implicit per-layer
  // cache has a fixed 1024-token capacity and cannot serve long prompts.
  if (!use_npu_prefill) {
    const int64_t requested_cache_length =
        static_cast<int64_t>(raw_input_tokens.shape()[1]) + max_new_tokens_value;
    if (requested_cache_length <= 0 || requested_cache_length > cpu_cfg.max_cache_length) {
      fmt::print(stderr, "Error: CPU prompt and generation require {} cache slots; configured capacity is {}.\n",
                 requested_cache_length, cpu_cfg.max_cache_length);
      ::mllm::shutdownContext();
      return 2;
    }
    shared_kv_cache = std::make_unique<mllm::nn::StaticCache>(
        static_cast<int32_t>(requested_cache_length), cpu_cfg.num_hidden_layers, cpu_cfg.num_attention_heads,
        cpu_cfg.num_key_value_heads, cpu_cfg.head_dim, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
    auto& cpu_decode_blocks = cpu_decode_model->text_model().decode_blocks();
    for (int32_t layer_idx = 0; layer_idx < cpu_cfg.num_hidden_layers; ++layer_idx) {
      auto& cache = cpu_decode_blocks.list()[layer_idx].getKVCache();
      cache.setStaticCache(shared_kv_cache.get());
      cache.setLayerIndex(layer_idx);
    }
    fmt::print("SHADOWNPU_CPU_CACHE capacity={} layers={}\n", requested_cache_length, cpu_cfg.num_hidden_layers);
  }

  // Prefill phase
  if (!use_npu_prefill) {
    MLLM_INFO("Starting CPU prefill...");
    mllm::cpu::shadownpu::CPUAttentionProfiler::reset();
    mllm::cpu::shadownpu::CPUSparseSelectionStats::reset();
    auto prefill_start = std::chrono::high_resolution_clock::now();

    prefill_seq_len = static_cast<int32_t>(raw_input_tokens.shape()[1]);
    mllm::models::ARGenerationArgs prefill_args;
    prefill_args["seq_len"] = prefill_seq_len;
    prefill_args["debug_layer_outputs"] = true;
    prefill_args["return_hidden_states"] = true;
    auto prefill_output = cpu_decode_model->forward(past, prefill_args);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    const auto prefill_duration = std::chrono::duration<double, std::milli>(prefill_end - prefill_start);
    prefill_time_ms = prefill_duration.count();
    MLLM_INFO("CPU prefill completed: seq_len={}, time={:.2f}s", prefill_seq_len, prefill_duration.count() / 1000.0);

    // Keep CPU diagnostics observational: a last-token Slice/Contiguous submission here can
    // perturb the eager executor before the next decode step. The full CPU tensor already
    // covers every value, and decode tensors have sequence length one.
    printLayerSummaries("cpu_prefill", cpu_decode_model->capturedLayerOutputs(), false);
    auto prefill_hidden_states = prefill_output.at("hidden_states").to(mllm::kFloat32).contiguous();
    printLogitSummary("cpu_prefill_hidden", prefill_hidden_states.ptr<float>(), prefill_hidden_states.numel());
    auto prefill_logits = prefill_output["sequence"];
    auto prefill_logits_f = prefill_logits.to(mllm::kFloat32);
    const auto& prefill_shape = prefill_logits_f.shape();

    if (prefill_shape.size() == 3 && prefill_shape[1] > 0) {
      auto last_logits = prefill_logits_f[{mllm::kAll, {prefill_shape[1] - 1}, mllm::kAll}];
      last_logits = last_logits.view({1, 1, prefill_shape[2]});

      const int64_t vocab_size = prefill_shape[2];
      auto* logits_data = last_logits.ptr<float>();
      printLogitSummary("cpu_prefill", logits_data, vocab_size);
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
      generated_token_ids.push_back(first_next_id);

      auto first_token_str = qwen_tokenizer.detokenize(static_cast<int64_t>(first_next_id));
      generated_text += mllm::preprocessor::wideString2Utf8String(first_token_str);
      std::wcout << first_token_str << std::flush;
      prefill_hit_eos = first_next_id == eos_token_id;
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

    prefill_seq_len = static_cast<int32_t>(raw_input_tokens.shape()[1]);
    const int32_t logical_chunk = prefill_chunk.get() > 0 ? prefill_chunk.get() : prefill_seq_len;
    const int32_t trace_sequence = qnn_sequence_tile.get() > 0 ? qnn_sequence_tile.get() : logical_chunk;
    auto trace_tokens = raw_input_tokens[{mllm::kAll, {0, trace_sequence}}].contiguous();
    mllm::models::ARGenerationOutputPast trace_past{{"sequence", trace_tokens}};
    auto irs = npu_model->trace(trace_past, {});

    mllm::ir::PassManager rewritePM(irs["model"]);
    rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
    rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
    if (!rewritePM.run()) { throw std::runtime_error("QNN graph rewrite pipeline failed"); }

    mllm::ir::PassManager graphBuildPM(irs["model"]);
    graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
    if (!graphBuildPM.run()) { throw std::runtime_error("QNN graph build pipeline failed"); }

    if (prepare_context) {
      const auto& output_path = prepare_qnn_context.get();
      if (!qnn_backend->saveContext(output_path)) {
        throw std::runtime_error("failed to save prepare-only QNN context to " + output_path);
      }
      const auto context_bytes = std::filesystem::file_size(output_path);
      fmt::print("SHADOWNPU_QNN_CONTEXT_GENERATED path={} bytes={} trace_sequence={} logical_chunk={} attention_mode={}\n",
                 output_path, context_bytes, trace_sequence, logical_chunk, attention_mode.get());
      std::cout.flush();
      std::wcout.flush();
      npu_model.reset();
      qnn_backend.reset();
      ::mllm::shutdownContext();
      return 0;
    }

    npu_model->model.clearKVCache();

    mllm::cpu::shadownpu::CPUAttentionProfiler::reset();
    mllm::cpu::shadownpu::CPUSparseSelectionStats::reset();
    auto prefill_start = std::chrono::steady_clock::now();
    mllm::models::ARGenerationOutputPast prefill_output;
    prefill_chunk_count = prefill_seq_len / logical_chunk;
    for (int32_t chunk_index = 0; chunk_index < prefill_chunk_count; ++chunk_index) {
      const int32_t chunk_start = chunk_index * logical_chunk;
      auto chunk_tokens = raw_input_tokens[{mllm::kAll, {chunk_start, chunk_start + logical_chunk}}].contiguous();
      auto chunk_positions = mllm::Tensor::empty({1, logical_chunk}, mllm::kInt64, mllm::kCPU).alloc();
      for (int32_t position = 0; position < logical_chunk; ++position) {
        chunk_positions.at<mllm::mllm_int64_t>({0, position}) = static_cast<mllm::mllm_int64_t>(chunk_start + position);
      }
      mllm::models::ARGenerationOutputPast chunk_past{
          {"sequence", std::move(chunk_tokens)},
          {"position_ids", std::move(chunk_positions)},
      };
      mllm::models::ARGenerationArgs prefill_args;
      prefill_args["seq_len"] = logical_chunk;
      if (debug_qnn_layer_outputs.isSet() || debug_qnn_boundary_dir.isSet()) {
        prefill_args["debug_layer_outputs"] = true;
        prefill_args["return_hidden_states"] = true;
      }
      prefill_output = npu_model->forward(chunk_past, prefill_args);
    }

    auto prefill_end = std::chrono::steady_clock::now();
    const auto prefill_duration = std::chrono::duration<double, std::milli>(prefill_end - prefill_start);
    prefill_time_ms = prefill_duration.count();

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

      const auto first_token_ready = std::chrono::steady_clock::now();
      const auto ttft = std::chrono::duration<double, std::milli>(first_token_ready - prefill_start);
      fmt::print("SHADOWNPU_TTFT basis=prefill_start_to_first_token attention_mode={} prompt_tokens={} "
                 "first_token_id={} ttft_ms={:.3f}\n",
                 attention_mode.get(), prefill_seq_len, first_next_id, ttft.count());
      MLLM_INFO("QNN prefill completed: seq_len={}, chunks={}, logical_chunk={}, qnn_tile={}, time={:.2f}s", prefill_seq_len,
                prefill_chunk_count, logical_chunk, qnn_sequence_tile.get(), prefill_duration.count() / 1000.0);

      if (debug_qnn_layer_outputs.isSet() || debug_qnn_boundary_dir.isSet()) {
        printLayerSummaries("qnn_prefill", npu_model->capturedLayerOutputs(), true);
        for (size_t layer_index = 0; layer_index < npu_model->capturedBoundaryLayerCount(); ++layer_index) {
          printLayerBoundarySummaries(layer_index, npu_model->capturedLayerBoundaries(layer_index));
        }
        if (debug_qnn_boundary_dir.isSet()) {
          dumpQNNBoundaryTensors(debug_qnn_boundary_dir.get(), *npu_model);
        }
        auto prefill_hidden_states = prefill_output.at("hidden_states").to(mllm::kFloat32).contiguous();
        printLogitSummary("qnn_prefill_hidden", prefill_hidden_states.ptr<float>(), prefill_hidden_states.numel());
      }

      printLogitSummary("qnn_prefill", logits_data, vocab_size);
      if (dump_qnn_prefill_logits.isSet()) {
        const auto& path = dump_qnn_prefill_logits.get();
        std::ofstream dump(path, std::ios::binary);
        dump.write(reinterpret_cast<const char*>(logits_data), static_cast<std::streamsize>(vocab_size * sizeof(float)));
        dump.close();
        logits_dump_failed = !dump;
        if (logits_dump_failed) {
          fmt::print(stderr, "Error: could not write QNN prefill logits to {}.\n", path);
        } else {
          fmt::print("SHADOWNPU_PREFILL_LOGITS_DUMP values={} path={}\n", vocab_size, path);
        }
      }

      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(first_next_id);
      past["position_ids"] = prefill_output["position_ids"];
      generated_token_ids.push_back(first_next_id);

      auto first_token_str = qwen_tokenizer.detokenize(static_cast<int64_t>(first_next_id));
      generated_text += mllm::preprocessor::wideString2Utf8String(first_token_str);
      std::wcout << first_token_str << std::flush;
      prefill_hit_eos = first_next_id == eos_token_id;
    } else {
      if (dump_qnn_prefill_logits.isSet()) {
        fmt::print(stderr, "Error: no final QNN prefill logits are available to dump.\n");
        logits_dump_failed = true;
      }
      auto last_token_idx = raw_input_tokens.shape()[1] - 1;
      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = raw_input_tokens.at<mllm::mllm_int64_t>({0, last_token_idx});
      past["position_ids"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["position_ids"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(prefill_seq_len);
    }
  }

  double total_decode_time_ms = 0.0;
  int decode_count = 0;
  if (!run_prefill_only && !prefill_hit_eos && !logits_dump_failed) {
    // Decode phase
    MLLM_INFO("Starting CPU decode...");
    constexpr int decode_start_step = 1;

    for (int step = decode_start_step; step < max_new_tokens_value; ++step) {
      auto step_start = std::chrono::high_resolution_clock::now();
      auto output = cpu_decode_model->forward(past, args);
      auto step_end = std::chrono::high_resolution_clock::now();
      auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
      double step_time_ms = step_duration.count() / 1000.0;

      total_decode_time_ms += step_time_ms;
      decode_count++;

      mllm::Tensor logits = output["sequence"];
      mllm::Tensor logits_f = logits.to(mllm::kFloat32);
      auto decode_hidden_states = output.at("hidden_states").to(mllm::kFloat32).contiguous();
      printLayerSummaries("cpu_decode", cpu_decode_model->capturedLayerOutputs(), false);
      printLogitSummary("cpu_decode_hidden", decode_hidden_states.ptr<float>(), decode_hidden_states.numel());

      const auto& shape = logits_f.shape();
      if (shape.size() != 3) {
        MLLM_ERROR("Unexpected logits shape (expected [B,1,V])");
        break;
      }

      const int64_t vocab_size = shape[2];
      auto* data = logits_f.ptr<float>();
      printLogitSummary("cpu_decode", data, vocab_size);

      int64_t next_id = 0;
      float max_logit = data[0];
      for (int64_t i = 1; i < vocab_size; ++i) {
        if (data[i] > max_logit) {
          max_logit = data[i];
          next_id = i;
        }
      }
      auto next_token_str = qwen_tokenizer.detokenize(next_id);
      generated_token_ids.push_back(next_id);
      generated_text += mllm::preprocessor::wideString2Utf8String(next_token_str);

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
  }
  std::wcout << L"\n";
  if (decode_count > 0) {
    double avg_time_ms = total_decode_time_ms / decode_count;
    MLLM_INFO(
        "Decode completed: {} tokens, total time: {:.2f}ms ({:.2f}s), avg time: {:.2f}ms/token, throughput: {:.2f} tokens/s",
        decode_count, total_decode_time_ms, total_decode_time_ms / 1000.0, avg_time_ms,
        decode_count / (total_decode_time_ms / 1000.0));
  }
  const auto attention_profile = mllm::cpu::shadownpu::CPUAttentionProfiler::snapshot();
  const auto selection = mllm::cpu::shadownpu::CPUSparseSelectionStats::snapshot();
  const double actual_retention =
      selection.eligible == 0 ? 1.0 : static_cast<double>(selection.retained) / static_cast<double>(selection.eligible);
  const auto qnn_tile_profile = qnn_backend ? qnn_backend->sequenceTileProfile() : mllm::qnn::QNNBackend::SequenceTileProfile{};
  fmt::print(
      "SHADOWNPU_RESULT attention_mode={} prompt_tokens={} prefill_ms={:.3f} generated_tokens={} actual_retention={:.8f} "
      "hmx_select_ms={:.3f} sparse_total_ms={:.3f} prefill_chunks={} qnn_tile_sequence={} qnn_graph_calls={} "
      "qnn_tile_calls={} qnn_input_copy_ms={:.3f} qnn_execute_ms={:.3f} qnn_output_copy_ms={:.3f} "
      "qnn_scratch_bytes={}\n",
      attention_mode.get(), raw_input_tokens.shape()[1], prefill_time_ms, generated_token_ids.size(), actual_retention,
      attention_profile.milliseconds(mllm::cpu::shadownpu::AttentionProfileStage::HMX_SELECT),
      attention_profile.milliseconds(mllm::cpu::shadownpu::AttentionProfileStage::SPARSE_TOTAL), prefill_chunk_count,
      qnn_backend ? qnn_backend->sequenceTile() : 0, qnn_tile_profile.graph_calls, qnn_tile_profile.tile_calls,
      qnn_tile_profile.input_copy_us / 1000.0, qnn_tile_profile.execute_us / 1000.0, qnn_tile_profile.output_copy_us / 1000.0,
      qnn_tile_profile.scratch_bytes);
  fmt::print("SHADOWNPU_TOKEN_IDS ids=");
  for (size_t index = 0; index < generated_token_ids.size(); ++index) {
    if (index != 0) { fmt::print(","); }
    fmt::print("{}", generated_token_ids[index]);
  }
  fmt::print("\n");
  fmt::print("SHADOWNPU_GENERATED_TEXT text=\"{}\"\n", jsonEscape(generated_text));
  if (!expected_access_code.get().empty()) {
    const std::string quality_answer = answerBeforeEndMarker(generated_text);
    const std::string extracted_code = extractAccessCode(quality_answer);
    const bool retrieval_exact = extracted_code == expected_access_code.get();
    fmt::print("QUALITY_RESULT expected_code={} extracted_code={} retrieval_exact={}\n", expected_access_code.get(),
               extracted_code.empty() ? "none" : extracted_code, retrieval_exact);
  }
  if (run_prefill_only) {
    const int64_t top1_token_id = generated_token_ids.empty() ? -1 : generated_token_ids.front();
    fmt::print("SHADOWNPU_PREFILL_ONLY_RESULT attention_mode={} prompt_tokens={} top1_token_id={} prefill_ms={:.3f}\n",
               attention_mode.get(), raw_input_tokens.shape()[1], top1_token_id, prefill_time_ms);
  }
#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen_npu.perfetto");
#endif

  // Tear down QNN while the process is still under application control.
  // Leaving the backend alive until C runtime finalization lets vendor DSO
  // destructors run after their dependent QNN state and can abort in
  // libQnnHtpPrepare. Release the model first, drop our backend reference,
  // then let shutdownContext remove the Context-owned reference.
  std::cout.flush();
  std::wcout.flush();
  npu_model.reset();
  qnn_backend.reset();
  ::mllm::shutdownContext();

  return logits_dump_failed ? 2 : 0;
}
