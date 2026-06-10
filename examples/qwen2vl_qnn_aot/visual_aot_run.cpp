// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTModule.hpp>
#include <mllm/core/DataTypes.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>

using mllm::Argparse;
using mllm::Tensor;
using mllm::models::qwen2vl::Qwen2VLConfig;
using mllm::models::qwen2vl::Qwen2VLTokenizer;
using mllm::qnn::aot::QnnAOTModule;

namespace {

bool g_verbose = false;

struct TimedResult {
  std::string name;
  Tensor output;
  Tensor compare_output;
  double seconds = 0.0;
};

struct CompareStats {
  int64_t numel = 0;
  double cosine = 0.0;
  double l2 = 0.0;
  double ref_l2 = 0.0;
  double l2_rel = 0.0;
  double qnn_l2 = 0.0;
  double norm_ratio = 0.0;
  float max_abs_diff = 0.0f;
  float qnn_at_max = 0.0f;
  float ref_at_max = 0.0f;
  int64_t max_abs_index = -1;
  int64_t nonfinite_qnn = 0;
  int64_t nonfinite_ref = 0;
};

Tensor makeQnnTensor(const std::vector<int32_t>& shape, mllm::DataTypes dtype, const std::string& name) {
  if (g_verbose) {
    fmt::print("[visual-aot] alloc QNN tensor {} shape=[", name);
    for (size_t i = 0; i < shape.size(); ++i) { fmt::print("{}{}", i == 0 ? "" : ", ", shape[i]); }
    fmt::print("] dtype={}\n", dtype);
    std::cout << std::flush;
  }
  auto tensor = Tensor::empty(shape, dtype, mllm::kQNN).setName(name).alloc();
  if (g_verbose) {
    fmt::print("[visual-aot] alloc QNN tensor {} done, bytes={}, ptr={}\n", name, tensor.bytes(), tensor.ptr<void>());
    std::cout << std::flush;
  }
  return tensor;
}

Tensor copyToQnn(const Tensor& cpu_tensor, const std::string& name) {
  if (g_verbose) {
    fmt::print("[visual-aot] copy {} CPU->QNN begin, bytes={}\n", name, cpu_tensor.bytes());
    std::cout << std::flush;
  }
  auto qnn_tensor = makeQnnTensor(cpu_tensor.shape(), cpu_tensor.dtype(), name);
  if (g_verbose) {
    fmt::print("[visual-aot] memcpy {} CPU ptr={} -> QNN ptr={} bytes={}\n", name, cpu_tensor.ptr<void>(), qnn_tensor.ptr<void>(),
               cpu_tensor.bytes());
    std::cout << std::flush;
  }
  std::memcpy(qnn_tensor.ptr<void>(), cpu_tensor.ptr<void>(), cpu_tensor.bytes());
  if (g_verbose) {
    fmt::print("[visual-aot] copy {} CPU->QNN done\n", name);
    std::cout << std::flush;
  }
  return qnn_tensor;
}

TimedResult runVisualGraph(const std::string& graph_name,
                           Tensor hidden,
                           Tensor visual_embedding_sin,
                           Tensor visual_embedding_cos,
                           Tensor output,
                           bool snapshot_output) {
  fmt::print("[visual-aot] execute {} ...\n", graph_name);
  std::cout << std::flush;

  QnnAOTModule module(graph_name);
  module.to(mllm::kQNN);
  module.setOutputTensors({output});

  auto inputs = std::vector<Tensor>{hidden, visual_embedding_sin, visual_embedding_cos};
  const auto start = std::chrono::high_resolution_clock::now();
  auto outputs = module(inputs);
  const auto end = std::chrono::high_resolution_clock::now();

  MLLM_RT_ASSERT_EQ(outputs.size(), 1);
  fmt::print("[visual-aot] execute {} done\n", graph_name);
  std::cout << std::flush;
  Tensor compare_output = Tensor::nil();
  if (snapshot_output) {
    compare_output = outputs[0].to(mllm::kCPU).clone();
  }

  return {
      .name = graph_name,
      .output = outputs[0],
      .compare_output = compare_output,
      .seconds = std::chrono::duration<double>(end - start).count(),
  };
}

const Tensor& compareTensor(const TimedResult& result) {
  return result.compare_output.isNil() ? result.output : result.compare_output;
}

void printShape(const std::string& name, const Tensor& tensor) {
  fmt::print("{} shape=[", name);
  const auto& shape = tensor.shape();
  for (size_t i = 0; i < shape.size(); ++i) { fmt::print("{}{}", i == 0 ? "" : ", ", shape[i]); }
  fmt::print("] dtype={} device={}\n", tensor.dtype(), tensor.device());
}

bool sameShape(const Tensor& lhs, const Tensor& rhs) { return lhs.shape() == rhs.shape(); }

CompareStats compareFloatTensors(const Tensor& qnn_tensor, const Tensor& ref_tensor) {
  MLLM_RT_ASSERT_EQ(qnn_tensor.dtype(), mllm::kFloat32);
  MLLM_RT_ASSERT_EQ(ref_tensor.dtype(), mllm::kFloat32);
  MLLM_RT_ASSERT(sameShape(qnn_tensor, ref_tensor));

  CompareStats stats;
  stats.numel = qnn_tensor.numel();
  const auto* qnn = qnn_tensor.ptr<float>();
  const auto* ref = ref_tensor.ptr<float>();

  double dot = 0.0;
  double qnn_norm2 = 0.0;
  double ref_norm2 = 0.0;
  double diff_norm2 = 0.0;
  float max_abs = -1.0f;

  for (int64_t i = 0; i < stats.numel; ++i) {
    const float q = qnn[i];
    const float r = ref[i];
    if (!std::isfinite(q)) { ++stats.nonfinite_qnn; }
    if (!std::isfinite(r)) { ++stats.nonfinite_ref; }
    const double qd = static_cast<double>(q);
    const double rd = static_cast<double>(r);
    const double diff = qd - rd;
    dot += qd * rd;
    qnn_norm2 += qd * qd;
    ref_norm2 += rd * rd;
    diff_norm2 += diff * diff;
    const float abs_diff = std::abs(q - r);
    if (abs_diff > max_abs) {
      max_abs = abs_diff;
      stats.max_abs_index = i;
      stats.qnn_at_max = q;
      stats.ref_at_max = r;
    }
  }

  constexpr double eps = 1e-12;
  stats.qnn_l2 = std::sqrt(qnn_norm2);
  stats.ref_l2 = std::sqrt(ref_norm2);
  stats.l2 = std::sqrt(diff_norm2);
  stats.cosine = dot / (stats.qnn_l2 * stats.ref_l2 + eps);
  stats.l2_rel = stats.l2 / (stats.ref_l2 + eps);
  stats.norm_ratio = stats.qnn_l2 / (stats.ref_l2 + eps);
  stats.max_abs_diff = std::max(0.0f, max_abs);
  return stats;
}

void printCompareStats(const CompareStats& stats) {
  fmt::print(fg(fmt::color::cyan), "\n{:=^58}\n", " Visual QNN vs Reference ");
  fmt::print("numel               {:>16}\n", stats.numel);
  fmt::print("cosine              {:>16.8f}\n", stats.cosine);
  fmt::print("L2 diff             {:>16.6f}\n", stats.l2);
  fmt::print("ref L2              {:>16.6f}\n", stats.ref_l2);
  fmt::print("relative L2         {:>16.8f}\n", stats.l2_rel);
  fmt::print("norm ratio qnn/ref  {:>16.8f}\n", stats.norm_ratio);
  fmt::print("max abs diff        {:>16.6f} @ {}\n", stats.max_abs_diff, stats.max_abs_index);
  fmt::print("qnn/ref at max      {:>16.6f} / {:.6f}\n", stats.qnn_at_max, stats.ref_at_max);
  fmt::print("nonfinite qnn/ref   {:>16} / {}\n", stats.nonfinite_qnn, stats.nonfinite_ref);
  fmt::print(fg(fmt::color::cyan), "{:=^58}\n", "");
}

Tensor runReferenceVisual(const std::string& model_path,
                          const std::string& model_version,
                          const std::string& config_path,
                          Tensor img,
                          Tensor visual_embedding_sin,
                          Tensor visual_embedding_cos) {
  fmt::print("[visual-aot] run reference visual model ...\n");
  std::cout << std::flush;

  auto cfg = Qwen2VLConfig(config_path);
  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV2;
  if (model_version == "v1") { file_version = mllm::ModelFileVersion::kV1; }

  auto params = mllm::load(model_path, file_version);
  auto visual = mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel("visual", cfg);
  visual.load(params);

  const auto start = std::chrono::high_resolution_clock::now();
  auto output = visual(img, visual_embedding_sin, visual_embedding_cos)[0];
  const auto end = std::chrono::high_resolution_clock::now();
  fmt::print("[visual-aot] reference visual done, time cost: {:.3f} s\n",
             std::chrono::duration<double>(end - start).count());
  printShape("reference_visual_embeddings", output);
  std::cout << std::flush;
  return output;
}

std::vector<TimedResult> runReferenceVisualBundle(const std::string& model_path,
                                                  const std::string& model_version,
                                                  const std::string& config_path,
                                                  Tensor img,
                                                  Tensor visual_embedding_sin,
                                                  Tensor visual_embedding_cos,
                                                  const std::string& bundle_layout) {
  fmt::print("[visual-aot] run reference visual bundle ...\n");
  std::cout << std::flush;

  auto cfg = Qwen2VLConfig(config_path);
  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV2;
  if (model_version == "v1") { file_version = mllm::ModelFileVersion::kV1; }

  auto params = mllm::load(model_path, file_version);
  auto patch_embed = mllm::models::qwen2vl::PatchEmbed("visual.patch_embed", cfg);
  auto patch_merger = mllm::models::qwen2vl::PatchMerger("visual.merger", cfg);
  auto blocks = mllm::nn::ModuleList<mllm::models::qwen2vl::Qwen2VLVisionBlock>("visual.blocks", cfg.visual_depth, cfg);
  patch_embed.load(params);
  patch_merger.load(params);
  blocks.load(params);

  std::vector<TimedResult> results;
  results.reserve(bundle_layout == "tail4" ? 8 : bundle_layout == "early2" ? 9 : bundle_layout == "block1" ? 34 : 6);

  auto push_stage = [&](const std::string& name, const Tensor& output) {
    results.push_back({.name = name, .output = output, .seconds = 0.0});
  };

  auto hidden = patch_embed(img)[0];
  push_stage("visual_patch_embed", hidden);

  auto run_block_range = [&](int32_t begin, int32_t end, const std::string& stage_name) {
    for (int32_t i = begin; i < end; ++i) {
      hidden = blocks.list()[i](hidden, visual_embedding_sin, visual_embedding_cos)[0];
    }
    push_stage(stage_name, hidden);
  };

  if (bundle_layout == "6x8") {
    run_block_range(0, 8, "visual_blocks_0_8");
    run_block_range(8, 16, "visual_blocks_8_16");
    run_block_range(16, 24, "visual_blocks_16_24");
    run_block_range(24, 32, "visual_blocks_24_32");
  } else if (bundle_layout == "tail4") {
    run_block_range(0, 8, "visual_blocks_0_8");
    run_block_range(8, 16, "visual_blocks_8_16");
    run_block_range(16, 20, "visual_blocks_16_20");
    run_block_range(20, 24, "visual_blocks_20_24");
    run_block_range(24, 28, "visual_blocks_24_28");
    run_block_range(28, 32, "visual_blocks_28_32");
  } else if (bundle_layout == "early2") {
    run_block_range(0, 2, "visual_blocks_0_2");
    run_block_range(2, 4, "visual_blocks_2_4");
    run_block_range(4, 6, "visual_blocks_4_6");
    run_block_range(6, 8, "visual_blocks_6_8");
    run_block_range(8, 16, "visual_blocks_8_16");
    run_block_range(16, 24, "visual_blocks_16_24");
    run_block_range(24, 32, "visual_blocks_24_32");
  } else if (bundle_layout == "block1") {
    for (int32_t i = 0; i < 32; ++i) {
      run_block_range(i, i + 1, "visual_blocks_" + std::to_string(i) + "_" + std::to_string(i + 1));
    }
  } else {
    std::cerr << "--bundle_layout must be 6x8, tail4, early2 or block1\n";
    return {};
  }

  auto merged = patch_merger(hidden)[0];
  push_stage("visual_merger", merged);
  std::cout << std::flush;
  return results;
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("-m|--model").help("Qwen2-VL visual QNN AOT context .bin path").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Qwen2-VL config path").required(true);
  auto& image_path = Argparse::add<std::string>("-i|--image").help("input image path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("prompt text").def("describe this picture");
  auto& ref_model_path =
      Argparse::add<std::string>("--ref_model").help("Optional FP32/W4A32 .mllm model used to compare visual output.").def("");
  auto& ref_model_version =
      Argparse::add<std::string>("--ref_model_version").help("reference model file version: v1/v2").def("v2");
  auto& ref_config_path =
      Argparse::add<std::string>("--ref_config").help("Optional reference model config path. Defaults to --config.").def("");
  auto& compare_segments =
      Argparse::add<bool>("--compare_segments").help("Compare each visual bundle segment against the CPU/KAI reference.");
  auto& compare_segments_oracle = Argparse::add<bool>("--compare_segments_oracle")
                                      .help("Run each visual QNN segment with the matching FP32 reference segment input.");
  auto& bundle_layout =
      Argparse::add<std::string>("--bundle_layout").help("visual bundle layout: 6x8, tail4, early2 or block1").def("6x8");
  auto& max_graphs = Argparse::add<int>("--max_graphs")
                         .help("Execute only the first N visual graphs for diagnostics. -1 executes the full bundle; 0 only prepares inputs.")
                         .def(-1);
  auto& verbose = Argparse::add<bool>("--verbose").help("Print detailed allocation/copy diagnostics.");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  g_verbose = verbose.isSet();

  mllm::initQnnBackend(context_path.get());

  auto cfg = Qwen2VLConfig(config_path.get());
  auto tokenizer = Qwen2VLTokenizer(tokenizer_path.get());
  auto inputs = tokenizer.convertMessage({.prompt = prompt.get(), .img_file_path = image_path.get()});
  auto img = inputs.at("img");
  auto grid_thw = inputs.at("grid_thw");

  auto inv_freq = mllm::models::qwen2vl::makeVisualRoPEInvFreq(cfg.visual_embed_dim / cfg.visual_num_heads, 10000.0);
  auto visual_pos_ids = mllm::models::qwen2vl::makeVisualRotaryPosEmbIds(grid_thw, cfg.visual_spatial_merge_size);
  auto rotary_pos_emb_full = mllm::models::qwen2vl::makeVisualRotaryPosEmbFull(inv_freq, img.shape()[0]);
  auto pos_emb = mllm::models::qwen2vl::makeVisualRotaryPosEmb(rotary_pos_emb_full, visual_pos_ids, grid_thw);
  auto [visual_embedding_sin, visual_embedding_cos] = mllm::models::qwen2vl::makeVisualRotarySinCos(pos_emb);

  const int32_t half_dim = cfg.visual_embed_dim / cfg.visual_num_heads / 2;
  visual_embedding_sin = visual_embedding_sin.view({1, -1, 1, half_dim}, false);
  visual_embedding_cos = visual_embedding_cos.view({1, -1, 1, half_dim}, false);

  const int32_t visual_patch_tokens = img.shape()[0];
  const int32_t patch_dim = img.shape()[1];
  const int32_t merged_tokens = visual_patch_tokens / (cfg.visual_spatial_merge_size * cfg.visual_spatial_merge_size);

  fmt::print("Qwen2-VL visual QNN AOT runner\n");
  printShape("img", img);
  printShape("grid_thw", grid_thw);
  printShape("visual_embedding_sin", visual_embedding_sin);
  printShape("visual_embedding_cos", visual_embedding_cos);
  std::cout << std::flush;

  if (g_verbose) {
    fmt::print("[visual-aot] copy CPU inputs to QNN shared buffers ...\n");
    std::cout << std::flush;
  }
  auto img_qnn = copyToQnn(img, "visual_img");
  auto sin_qnn = copyToQnn(visual_embedding_sin, "visual_embedding_sin");
  auto cos_qnn = copyToQnn(visual_embedding_cos, "visual_embedding_cos");
  if (g_verbose) {
    fmt::print("[visual-aot] copy CPU inputs to QNN shared buffers done\n");
    std::cout << std::flush;
  }

  auto hidden_a = makeQnnTensor({visual_patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, "visual_hidden_a");
  auto hidden_b = makeQnnTensor({visual_patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, "visual_hidden_b");
  auto visual_embeddings = makeQnnTensor({merged_tokens, cfg.hidden_size}, mllm::kFloat32, "visual_embeddings");

  std::vector<TimedResult> results;
  if (bundle_layout.get() == "6x8") {
    results.reserve(6);
    const int32_t graph_limit = max_graphs.get() < 0 ? 6 : std::max(0, std::min(max_graphs.get(), 6));
    if (graph_limit >= 1) {
      results.push_back(runVisualGraph("visual_patch_embed", img_qnn, sin_qnn, cos_qnn, hidden_a, compare_segments.isSet()));
    }
    if (graph_limit >= 2) {
      results.push_back(runVisualGraph("visual_blocks_0_8", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 3) {
      results.push_back(runVisualGraph("visual_blocks_8_16", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 4) {
      results.push_back(runVisualGraph("visual_blocks_16_24", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 5) {
      results.push_back(runVisualGraph("visual_blocks_24_32", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 6) {
      results.push_back(runVisualGraph("visual_merger", results.back().output, sin_qnn, cos_qnn, visual_embeddings,
                                       compare_segments.isSet()));
    }
  } else if (bundle_layout.get() == "tail4") {
    results.reserve(8);
    const int32_t graph_limit = max_graphs.get() < 0 ? 8 : std::max(0, std::min(max_graphs.get(), 8));
    if (graph_limit >= 1) {
      results.push_back(runVisualGraph("visual_patch_embed", img_qnn, sin_qnn, cos_qnn, hidden_a, compare_segments.isSet()));
    }
    if (graph_limit >= 2) {
      results.push_back(runVisualGraph("visual_blocks_0_8", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 3) {
      results.push_back(runVisualGraph("visual_blocks_8_16", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 4) {
      results.push_back(runVisualGraph("visual_blocks_16_20", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 5) {
      results.push_back(runVisualGraph("visual_blocks_20_24", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 6) {
      results.push_back(runVisualGraph("visual_blocks_24_28", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 7) {
      results.push_back(runVisualGraph("visual_blocks_28_32", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 8) {
      results.push_back(runVisualGraph("visual_merger", results.back().output, sin_qnn, cos_qnn, visual_embeddings,
                                       compare_segments.isSet()));
    }
  } else if (bundle_layout.get() == "early2") {
    results.reserve(9);
    const int32_t graph_limit = max_graphs.get() < 0 ? 9 : std::max(0, std::min(max_graphs.get(), 9));
    if (graph_limit >= 1) {
      results.push_back(runVisualGraph("visual_patch_embed", img_qnn, sin_qnn, cos_qnn, hidden_a, compare_segments.isSet()));
    }
    if (graph_limit >= 2) {
      results.push_back(runVisualGraph("visual_blocks_0_2", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 3) {
      results.push_back(runVisualGraph("visual_blocks_2_4", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 4) {
      results.push_back(runVisualGraph("visual_blocks_4_6", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 5) {
      results.push_back(runVisualGraph("visual_blocks_6_8", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 6) {
      results.push_back(runVisualGraph("visual_blocks_8_16", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 7) {
      results.push_back(runVisualGraph("visual_blocks_16_24", results.back().output, sin_qnn, cos_qnn, hidden_a,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 8) {
      results.push_back(runVisualGraph("visual_blocks_24_32", results.back().output, sin_qnn, cos_qnn, hidden_b,
                                       compare_segments.isSet()));
    }
    if (graph_limit >= 9) {
      results.push_back(runVisualGraph("visual_merger", results.back().output, sin_qnn, cos_qnn, visual_embeddings,
                                       compare_segments.isSet()));
    }
  } else if (bundle_layout.get() == "block1") {
    results.reserve(34);
    const int32_t graph_limit = max_graphs.get() < 0 ? 34 : std::max(0, std::min(max_graphs.get(), 34));
    if (graph_limit >= 1) {
      results.push_back(runVisualGraph("visual_patch_embed", img_qnn, sin_qnn, cos_qnn, hidden_a, compare_segments.isSet()));
    }
    for (int32_t i = 0; i < 32 && graph_limit >= i + 2; ++i) {
      auto& output = (i % 2 == 0) ? hidden_b : hidden_a;
      results.push_back(runVisualGraph("visual_blocks_" + std::to_string(i) + "_" + std::to_string(i + 1), results.back().output,
                                       sin_qnn, cos_qnn, output, compare_segments.isSet()));
    }
    if (graph_limit >= 34) {
      results.push_back(runVisualGraph("visual_merger", results.back().output, sin_qnn, cos_qnn, visual_embeddings,
                                       compare_segments.isSet()));
    }
  } else {
    std::cerr << "--bundle_layout must be 6x8, tail4, early2 or block1\n";
    return 1;
  }

  double total_seconds = 0.0;
  fmt::print(fg(fmt::color::cyan), "\n{:=^58}\n", " Visual QNN AOT Performance ");
  for (const auto& result : results) {
    total_seconds += result.seconds;
    fmt::print("{:<20} {:>10.3f} s   ", result.name, result.seconds);
    printShape("output", result.output);
  }
  fmt::print("Visual total         {:>10.3f} s\n", total_seconds);
  fmt::print(fg(fmt::color::cyan), "{:=^58}\n", "");

  if (!ref_model_path.get().empty()) {
    const bool bundle_ok = (bundle_layout.get() == "6x8" && results.size() == 6 && results.back().name == "visual_merger")
                           || (bundle_layout.get() == "tail4" && results.size() == 8 && results.back().name == "visual_merger")
                           || (bundle_layout.get() == "early2" && results.size() == 9 && results.back().name == "visual_merger")
                           || (bundle_layout.get() == "block1" && results.size() == 34 && results.back().name == "visual_merger");
    if (!bundle_ok) {
      fmt::print("[visual-aot] skip reference comparison: --ref_model requires full bundle execution.\n");
    } else {
      const std::string ref_cfg = ref_config_path.get().empty() ? config_path.get() : ref_config_path.get();
      if (compare_segments.isSet() || compare_segments_oracle.isSet()) {
        auto ref_segments = runReferenceVisualBundle(ref_model_path.get(),
                                                     ref_model_version.get(),
                                                     ref_cfg,
                                                     img,
                                                     visual_embedding_sin,
                                                     visual_embedding_cos,
                                                     bundle_layout.get());
        if (ref_segments.size() == results.size()) {
          if (compare_segments.isSet()) {
            fmt::print("\n{:=^72}\n", " Visual Segment vs Reference ");
            for (size_t i = 0; i < results.size(); ++i) {
              const auto& qnn_stage = results[i];
              const auto& ref_stage = ref_segments[i];
              if (qnn_stage.name != ref_stage.name) {
                fmt::print("{:<24} vs {:<24} name mismatch, skip\n", qnn_stage.name, ref_stage.name);
                continue;
              }
              const auto& qnn_compare = compareTensor(qnn_stage);
              if (!sameShape(qnn_compare, ref_stage.output)) {
                fmt::print("{:<24} shape mismatch, skip\n", qnn_stage.name);
                printShape("qnn_stage", qnn_compare);
                printShape("ref_stage", ref_stage.output);
                continue;
              }
              auto stats = compareFloatTensors(qnn_compare, ref_stage.output);
              fmt::print("{:<24} cosine={:>10.8f}  rel_l2={:>10.8f}  norm_ratio={:>10.8f}\n",
                         qnn_stage.name, stats.cosine, stats.l2_rel, stats.norm_ratio);
            }
            fmt::print("{:=^72}\n", "");
          }

          if (compare_segments_oracle.isSet()) {
            fmt::print("\n{:=^84}\n", " Visual Segment Oracle Input vs Reference ");
            for (size_t i = 0; i < ref_segments.size(); ++i) {
              const auto& ref_stage = ref_segments[i];
              Tensor qnn_input = Tensor::nil();
              if (i == 0) {
                qnn_input = img_qnn;
              } else {
                qnn_input = copyToQnn(ref_segments[i - 1].output, "oracle_input_" + std::to_string(i));
              }
              auto qnn_output =
                  makeQnnTensor(ref_stage.output.shape(), ref_stage.output.dtype(), "oracle_output_" + std::to_string(i));
              auto qnn_stage = runVisualGraph(ref_stage.name, qnn_input, sin_qnn, cos_qnn, qnn_output, true);
              const auto& qnn_compare = compareTensor(qnn_stage);
              auto stats = compareFloatTensors(qnn_compare, ref_stage.output);
              fmt::print("{:<24} cosine={:>10.8f}  rel_l2={:>10.8f}  norm_ratio={:>10.8f}  time={:>8.3f}s\n",
                         ref_stage.name, stats.cosine, stats.l2_rel, stats.norm_ratio, qnn_stage.seconds);
            }
            fmt::print("{:=^84}\n", "");
          }
        } else {
          fmt::print("[visual-aot] skip segment comparison: reference segment count mismatch.\n");
        }
      }

      auto ref_visual_embeddings =
          runReferenceVisual(ref_model_path.get(), ref_model_version.get(), ref_cfg, img, visual_embedding_sin, visual_embedding_cos);
      if (!sameShape(results.back().output, ref_visual_embeddings)) {
        fmt::print("[visual-aot] skip comparison: shape mismatch between QNN output and reference output.\n");
        printShape("qnn_visual_embeddings", results.back().output);
        printShape("reference_visual_embeddings", ref_visual_embeddings);
      } else {
        printCompareStats(compareFloatTensors(results.back().output, ref_visual_embeddings));
      }
    }
  }

  mllm::memoryReport();
  return 0;
});
