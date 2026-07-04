// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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

struct Grid {
  int32_t t = 1;
  int32_t h = 0;
  int32_t w = 0;

  [[nodiscard]] int32_t patchTokens() const { return t * h * w; }
};

struct StageResult {
  std::string name;
  Tensor output;
  double seconds = 0.0;
  bool merged = false;
};

struct CompareStats {
  int64_t numel = 0;
  int64_t finite_count = 0;
  int64_t nonfinite_test = 0;
  int64_t nonfinite_ref = 0;
  double cosine = 0.0;
  double l2 = 0.0;
  double ref_l2 = 0.0;
  double rel_l2 = 0.0;
  double norm_ratio = 0.0;
  float max_abs_diff = 0.0f;
  int64_t max_abs_index = -1;
};

Grid parseGrid(const std::string& text) {
  auto sep = text.find('x');
  if (sep == std::string::npos) { sep = text.find('X'); }
  if (sep == std::string::npos) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid grid '{}', expected HxW.", text);
  }
  Grid grid{1, std::stoi(text.substr(0, sep)), std::stoi(text.substr(sep + 1))};
  if (grid.h <= 0 || grid.w <= 0 || grid.h % 2 != 0 || grid.w % 2 != 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid grid '{}': H/W must be positive even numbers.", text);
  }
  return grid;
}

std::string resolveSuffix(const std::string& value, int32_t patch_tokens) {
  if (value == "auto") { return "_s" + std::to_string(patch_tokens); }
  if (value == "none") { return ""; }
  return value;
}

Tensor makeGridThwTensor(const Grid& grid) {
  auto t = Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  t.ptr<int32_t>()[0] = grid.t;
  t.ptr<int32_t>()[1] = grid.h;
  t.ptr<int32_t>()[2] = grid.w;
  return t;
}

Grid readGrid(Tensor grid_thw) {
  const auto* g = grid_thw.ptr<int32_t>();
  return {.t = g[0], .h = g[1], .w = g[2]};
}

int64_t visualPatchIndex(int32_t t, int32_t h, int32_t w, int32_t grid_h, int32_t grid_w, int32_t merge_size) {
  const int32_t h_blocks = grid_h / merge_size;
  const int32_t w_blocks = grid_w / merge_size;
  return (((static_cast<int64_t>(t) * h_blocks + h / merge_size) * w_blocks + w / merge_size) * merge_size + h % merge_size)
             * merge_size
         + w % merge_size;
}

int64_t visualMergedIndex(int32_t t, int32_t h, int32_t w, int32_t grid_h, int32_t grid_w, int32_t merge_size) {
  const int32_t h_blocks = grid_h / merge_size;
  const int32_t w_blocks = grid_w / merge_size;
  return (static_cast<int64_t>(t) * h_blocks + h) * w_blocks + w;
}

Tensor padVisualPatchesToBucket(Tensor img, Tensor grid_thw, const Grid& bucket, const Qwen2VLConfig& cfg) {
  MLLM_RT_ASSERT_EQ(img.dtype(), mllm::kFloat32);
  const auto original = readGrid(grid_thw);
  if (bucket.t != original.t) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Only t=1 visual buckets are supported in this diagnostic.");
  }
  if (bucket.h < original.h || bucket.w < original.w) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "Bucket {}x{} cannot cover original grid {}x{}.",
                    bucket.h,
                    bucket.w,
                    original.h,
                    original.w);
  }
  if (bucket.h == original.h && bucket.w == original.w) { return img; }

  const int32_t patch_dim = img.shape()[1];
  auto padded = Tensor::empty({bucket.patchTokens(), patch_dim}, img.dtype(), mllm::kCPU).alloc();
  std::fill(padded.ptr<float>(), padded.ptr<float>() + padded.numel(), 0.0f);

  for (int32_t t = 0; t < original.t; ++t) {
    for (int32_t h = 0; h < original.h; ++h) {
      for (int32_t w = 0; w < original.w; ++w) {
        const auto src_idx = visualPatchIndex(t, h, w, original.h, original.w, cfg.visual_spatial_merge_size);
        const auto dst_idx = visualPatchIndex(t, h, w, bucket.h, bucket.w, cfg.visual_spatial_merge_size);
        std::memcpy(padded.offsettedPtr<float>({static_cast<int32_t>(dst_idx), 0}),
                    img.offsettedPtr<float>({static_cast<int32_t>(src_idx), 0}),
                    static_cast<size_t>(patch_dim) * sizeof(float));
      }
    }
  }
  return padded;
}

Tensor makeVisualAttentionMaskForBucket(Tensor original_grid_thw, const Grid& bucket, const Qwen2VLConfig& cfg) {
  const auto original = readGrid(original_grid_thw);
  if (bucket.t != original.t) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Only t=1 visual buckets are supported in this diagnostic.");
  }
  if (bucket.h < original.h || bucket.w < original.w) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "Bucket {}x{} cannot cover original grid {}x{}.",
                    bucket.h,
                    bucket.w,
                    original.h,
                    original.w);
  }

  auto mask = Tensor::empty({1, 1, 1, bucket.patchTokens()}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(mask.ptr<float>(), mask.ptr<float>() + mask.numel(), -10000.0f);
  for (int32_t t = 0; t < original.t; ++t) {
    for (int32_t h = 0; h < original.h; ++h) {
      for (int32_t w = 0; w < original.w; ++w) {
        const auto idx = visualPatchIndex(t, h, w, bucket.h, bucket.w, cfg.visual_spatial_merge_size);
        *mask.offsettedPtr<float>({0, 0, 0, static_cast<int32_t>(idx)}) = 0.0f;
      }
    }
  }
  return mask;
}

Tensor makeAllValidVisualAttentionMask(int32_t patch_tokens) {
  auto mask = Tensor::empty({1, 1, 1, patch_tokens}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(mask.ptr<float>(), mask.ptr<float>() + mask.numel(), 0.0f);
  return mask;
}

Tensor cropPatchTensorFromBucket(Tensor bucket_tensor, Tensor original_grid_thw, const Grid& bucket, const Qwen2VLConfig& cfg) {
  const auto original = readGrid(original_grid_thw);
  if (bucket.h == original.h && bucket.w == original.w) { return bucket_tensor; }

  const int32_t hidden_size = bucket_tensor.shape()[1];
  auto cropped = Tensor::empty({original.patchTokens(), hidden_size}, bucket_tensor.dtype(), mllm::kCPU).alloc();
  for (int32_t t = 0; t < original.t; ++t) {
    for (int32_t h = 0; h < original.h; ++h) {
      for (int32_t w = 0; w < original.w; ++w) {
        const auto src_idx = visualPatchIndex(t, h, w, bucket.h, bucket.w, cfg.visual_spatial_merge_size);
        const auto dst_idx = visualPatchIndex(t, h, w, original.h, original.w, cfg.visual_spatial_merge_size);
        std::memcpy(cropped.offsettedPtr<float>({static_cast<int32_t>(dst_idx), 0}),
                    bucket_tensor.offsettedPtr<float>({static_cast<int32_t>(src_idx), 0}),
                    static_cast<size_t>(hidden_size) * sizeof(float));
      }
    }
  }
  return cropped;
}

Tensor cropMergedTensorFromBucket(Tensor bucket_tensor, Tensor original_grid_thw, const Grid& bucket, const Qwen2VLConfig& cfg) {
  const auto original = readGrid(original_grid_thw);
  if (bucket.h == original.h && bucket.w == original.w) { return bucket_tensor; }

  const int32_t merge = cfg.visual_spatial_merge_size;
  const int32_t hidden_size = bucket_tensor.shape()[1];
  const int32_t original_h = original.h / merge;
  const int32_t original_w = original.w / merge;
  auto cropped = Tensor::empty({original.t * original_h * original_w, hidden_size}, bucket_tensor.dtype(), mllm::kCPU).alloc();

  for (int32_t t = 0; t < original.t; ++t) {
    for (int32_t h = 0; h < original_h; ++h) {
      for (int32_t w = 0; w < original_w; ++w) {
        const auto src_idx = visualMergedIndex(t, h, w, bucket.h, bucket.w, merge);
        const auto dst_idx = visualMergedIndex(t, h, w, original.h, original.w, merge);
        std::memcpy(cropped.offsettedPtr<float>({static_cast<int32_t>(dst_idx), 0}),
                    bucket_tensor.offsettedPtr<float>({static_cast<int32_t>(src_idx), 0}),
                    static_cast<size_t>(hidden_size) * sizeof(float));
      }
    }
  }
  return cropped;
}

std::pair<Tensor, Tensor> makeVisualSinCos(const Qwen2VLConfig& cfg, Tensor grid_thw, int32_t patch_tokens) {
  auto inv_freq = mllm::models::qwen2vl::makeVisualRoPEInvFreq(cfg.visual_embed_dim / cfg.visual_num_heads, 10000.0);
  auto pos_ids = mllm::models::qwen2vl::makeVisualRotaryPosEmbIds(grid_thw, cfg.visual_spatial_merge_size);
  auto rotary_pos_emb_full = mllm::models::qwen2vl::makeVisualRotaryPosEmbFull(inv_freq, patch_tokens);
  auto pos_emb = mllm::models::qwen2vl::makeVisualRotaryPosEmb(rotary_pos_emb_full, pos_ids, grid_thw);
  auto [sin, cos] = mllm::models::qwen2vl::makeVisualRotarySinCos(pos_emb);
  const int32_t half_dim = cfg.visual_embed_dim / cfg.visual_num_heads / 2;
  return {sin.view({1, -1, 1, half_dim}, false), cos.view({1, -1, 1, half_dim}, false)};
}

Tensor makeQnnTensor(const std::vector<int32_t>& shape, mllm::DataTypes dtype, const std::string& name) {
  return Tensor::empty(shape, dtype, mllm::kQNN).setName(name).alloc();
}

Tensor copyToQnn(const Tensor& cpu_tensor, const std::string& name) {
  auto qnn_tensor = makeQnnTensor(cpu_tensor.shape(), cpu_tensor.dtype(), name);
  std::memcpy(qnn_tensor.ptr<void>(), cpu_tensor.ptr<void>(), cpu_tensor.bytes());
  return qnn_tensor;
}

StageResult runVisualGraph(const std::string& graph_name,
                           Tensor hidden,
                           Tensor visual_embedding_sin,
                           Tensor visual_embedding_cos,
                           Tensor visual_attention_mask,
                           Tensor output,
                           bool merged) {
  QnnAOTModule module(graph_name);
  module.to(mllm::kQNN);
  module.setOutputTensors({output});
  auto inputs = visual_attention_mask.isNil() ? std::vector<Tensor>{hidden, visual_embedding_sin, visual_embedding_cos}
                                              : std::vector<Tensor>{hidden,
                                                                    visual_embedding_sin,
                                                                    visual_embedding_cos,
                                                                    visual_attention_mask};
  const auto start = std::chrono::high_resolution_clock::now();
  auto outputs = module(inputs);
  const auto end = std::chrono::high_resolution_clock::now();
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);
  return {
      .name = graph_name,
      .output = outputs[0].to(mllm::kCPU).clone(),
      .seconds = std::chrono::duration<double>(end - start).count(),
      .merged = merged,
  };
}

std::vector<std::string> graphNamesForLayout(const std::string& layout) {
  if (layout == "6x8") {
    return {
        "visual_patch_embed",
        "visual_blocks_0_8",
        "visual_blocks_8_16",
        "visual_blocks_16_24",
        "visual_blocks_24_32",
        "visual_merger",
    };
  }
  if (layout == "early2") {
    return {
        "visual_patch_embed",
        "visual_blocks_0_2",
        "visual_blocks_2_4",
        "visual_blocks_4_6",
        "visual_blocks_6_8",
        "visual_blocks_8_16",
        "visual_blocks_16_24",
        "visual_blocks_24_32",
        "visual_merger",
    };
  }
  if (layout == "tail4") {
    return {
        "visual_patch_embed",
        "visual_blocks_0_8",
        "visual_blocks_8_16",
        "visual_blocks_16_20",
        "visual_blocks_20_24",
        "visual_blocks_24_28",
        "visual_blocks_28_32",
        "visual_merger",
    };
  }
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--bundle_layout must be 6x8, early2 or tail4.");
}

std::vector<StageResult> runVisualBundle(Tensor img,
                                         Tensor sin,
                                         Tensor cos,
                                         Tensor attention_mask,
                                         const Qwen2VLConfig& cfg,
                                         const std::string& layout,
                                         const std::string& suffix,
                                         const std::string& tag) {
  const int32_t patch_tokens = img.shape()[0];
  const int32_t merged_tokens = patch_tokens / (cfg.visual_spatial_merge_size * cfg.visual_spatial_merge_size);
  auto img_qnn = copyToQnn(img, tag + "_img");
  auto sin_qnn = copyToQnn(sin, tag + "_sin");
  auto cos_qnn = copyToQnn(cos, tag + "_cos");
  auto mask_qnn = copyToQnn(attention_mask, tag + "_attention_mask");

  auto hidden_a = makeQnnTensor({patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, tag + "_hidden_a");
  auto hidden_b = makeQnnTensor({patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, tag + "_hidden_b");
  auto merged = makeQnnTensor({merged_tokens, cfg.hidden_size}, mllm::kFloat32, tag + "_merged");

  std::vector<StageResult> results;
  const auto graph_names = graphNamesForLayout(layout);
  Tensor current = img_qnn;
  for (size_t i = 0; i < graph_names.size(); ++i) {
    const bool is_last = i + 1 == graph_names.size();
    Tensor output = is_last ? merged : (i % 2 == 0 ? hidden_a : hidden_b);
    const auto graph_name = graph_names[i] + suffix;
    fmt::print("[{}] execute {} ...\n", tag, graph_name);
    const bool is_block = graph_names[i].find("visual_blocks_") == 0;
    current = runVisualGraph(graph_name, current, sin_qnn, cos_qnn, is_block ? mask_qnn : Tensor::nil(), output, is_last).output;
    results.push_back({.name = graph_names[i], .output = current, .seconds = 0.0, .merged = is_last});
    if (!is_last) {
      current = copyToQnn(results.back().output, tag + "_stage_" + std::to_string(i));
    }
  }
  return results;
}

CompareStats compareFloatTensors(const Tensor& test_tensor, const Tensor& ref_tensor) {
  MLLM_RT_ASSERT(test_tensor.shape() == ref_tensor.shape());
  MLLM_RT_ASSERT_EQ(test_tensor.dtype(), mllm::kFloat32);
  MLLM_RT_ASSERT_EQ(ref_tensor.dtype(), mllm::kFloat32);

  CompareStats stats;
  stats.numel = test_tensor.numel();
  const auto* test = test_tensor.ptr<float>();
  const auto* ref = ref_tensor.ptr<float>();
  double dot = 0.0;
  double test_norm2 = 0.0;
  double ref_norm2 = 0.0;
  double diff_norm2 = 0.0;
  float max_abs = -1.0f;

  for (int64_t i = 0; i < stats.numel; ++i) {
    const float a = test[i];
    const float b = ref[i];
    if (!std::isfinite(a)) {
      ++stats.nonfinite_test;
      continue;
    }
    if (!std::isfinite(b)) {
      ++stats.nonfinite_ref;
      continue;
    }
    ++stats.finite_count;
    const double ad = static_cast<double>(a);
    const double bd = static_cast<double>(b);
    const double diff = ad - bd;
    dot += ad * bd;
    test_norm2 += ad * ad;
    ref_norm2 += bd * bd;
    diff_norm2 += diff * diff;
    const float abs_diff = std::abs(a - b);
    if (abs_diff > max_abs) {
      max_abs = abs_diff;
      stats.max_abs_index = i;
    }
  }

  constexpr double eps = 1e-12;
  const double test_l2 = std::sqrt(test_norm2);
  stats.ref_l2 = std::sqrt(ref_norm2);
  stats.l2 = std::sqrt(diff_norm2);
  stats.cosine = dot / (test_l2 * stats.ref_l2 + eps);
  stats.rel_l2 = stats.l2 / (stats.ref_l2 + eps);
  stats.norm_ratio = test_l2 / (stats.ref_l2 + eps);
  stats.max_abs_diff = std::max(0.0f, max_abs);
  return stats;
}

void printShape(const std::string& name, const Tensor& tensor) {
  fmt::print("{} dtype={} shape=[", name, mllm::nameOfType(tensor.dtype()));
  const auto& shape = tensor.shape();
  for (size_t i = 0; i < shape.size(); ++i) { fmt::print("{}{}", i == 0 ? "" : ", ", shape[i]); }
  fmt::print("]\n");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("-m|--model").help("QNN AOT context containing both native and bucket visual graphs.").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Qwen2-VL visual config path").required(true);
  auto& image_path = Argparse::add<std::string>("-i|--image").help("input image path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("prompt text").def("describe this picture");
  auto& bucket_grid_arg = Argparse::add<std::string>("--bucket_grid").help("Padded bucket grid HxW, e.g. 36x26.").required(true);
  auto& native_suffix_arg = Argparse::add<std::string>("--native_suffix")
                                .help("Native graph suffix. Use auto for _s{native_tokens}, or none for unsuffixed graphs.")
                                .def("auto");
  auto& padded_suffix_arg = Argparse::add<std::string>("--padded_suffix")
                                .help("Padded graph suffix. Use auto for _s{bucket_tokens}, or none for unsuffixed graphs.")
                                .def("auto");
  auto& bundle_layout = Argparse::add<std::string>("--bundle_layout").help("visual bundle layout: 6x8, early2 or tail4").def("6x8");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(context_path.get());

  auto cfg = Qwen2VLConfig(config_path.get());
  auto tokenizer = Qwen2VLTokenizer(tokenizer_path.get());
  auto inputs = tokenizer.convertMessage({.prompt = prompt.get(), .img_file_path = image_path.get()});
  auto img = inputs.at("img");
  auto grid_thw = inputs.at("grid_thw");
  const auto original_grid = readGrid(grid_thw);
  const auto bucket_grid = parseGrid(bucket_grid_arg.get());
  auto bucket_grid_thw = makeGridThwTensor(bucket_grid);
  auto padded_img = padVisualPatchesToBucket(img, grid_thw, bucket_grid, cfg);

  const auto native_suffix = resolveSuffix(native_suffix_arg.get(), img.shape()[0]);
  const auto padded_suffix = resolveSuffix(padded_suffix_arg.get(), padded_img.shape()[0]);

  fmt::print("Qwen2-VL visual padding A/B diagnostic\n");
  fmt::print("original grid : {}x{} patches={} suffix='{}'\n", original_grid.h, original_grid.w, img.shape()[0], native_suffix);
  fmt::print("bucket grid   : {}x{} patches={} suffix='{}'\n", bucket_grid.h, bucket_grid.w, padded_img.shape()[0], padded_suffix);
  printShape("img", img);
  printShape("padded_img", padded_img);

  auto [native_sin, native_cos] = makeVisualSinCos(cfg, grid_thw, img.shape()[0]);
  auto [padded_sin, padded_cos] = makeVisualSinCos(cfg, bucket_grid_thw, padded_img.shape()[0]);
  auto native_mask = makeAllValidVisualAttentionMask(img.shape()[0]);
  auto padded_mask = makeVisualAttentionMaskForBucket(grid_thw, bucket_grid, cfg);

  auto native_results = runVisualBundle(img, native_sin, native_cos, native_mask, cfg, bundle_layout.get(), native_suffix, "native");
  auto padded_results =
      runVisualBundle(padded_img, padded_sin, padded_cos, padded_mask, cfg, bundle_layout.get(), padded_suffix, "padded");
  MLLM_RT_ASSERT_EQ(native_results.size(), padded_results.size());

  fmt::print(fg(fmt::color::cyan), "\n{:=^96}\n", " Native QNN vs Padded QNN Crop ");
  fmt::print("{:<24} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}\n",
             "stage",
             "cosine",
             "rel_l2",
             "norm_ratio",
             "max_abs",
             "nonfin_t",
             "nonfin_r");
  for (size_t i = 0; i < native_results.size(); ++i) {
    const auto& native = native_results[i];
    const auto& padded = padded_results[i];
    MLLM_RT_ASSERT(native.name == padded.name);
    auto cropped = padded.merged ? cropMergedTensorFromBucket(padded.output, grid_thw, bucket_grid, cfg)
                                 : cropPatchTensorFromBucket(padded.output, grid_thw, bucket_grid, cfg);
    auto stats = compareFloatTensors(cropped, native.output);
    fmt::print("{:<24} {:>12.8f} {:>12.8f} {:>12.8f} {:>12.6f} {:>10} {:>10}\n",
               native.name,
               stats.cosine,
               stats.rel_l2,
               stats.norm_ratio,
               stats.max_abs_diff,
               stats.nonfinite_test,
               stats.nonfinite_ref);
  }
  fmt::print(fg(fmt::color::cyan), "{:=^96}\n", "");

  mllm::memoryReport();
  return 0;
});
