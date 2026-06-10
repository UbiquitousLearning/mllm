// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include <mllm/core/DataTypes.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>

using mllm::Argparse;
using mllm::Tensor;
using mllm::models::qwen2vl::Qwen2VLConfig;
using mllm::models::qwen2vl::Qwen2VLTokenizer;

namespace {

struct CompareStats {
  int64_t numel = 0;
  int64_t nan_count = 0;
  int64_t inf_count = 0;
  int64_t finite_count = 0;
  double cosine = 0.0;
  double l2 = 0.0;
  double ref_l2 = 0.0;
  double rel_l2 = 0.0;
  double norm_ratio = 0.0;
  float max_abs_diff = 0.0f;
};

struct BucketGrid {
  int32_t grid_h = 0;
  int32_t grid_w = 0;
};

BucketGrid parseBucketGrid(const std::string& text) {
  auto sep = text.find('x');
  if (sep == std::string::npos) { sep = text.find('X'); }
  if (sep == std::string::npos) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid --bucket_grid '{}', expected HxW.", text);
  }
  BucketGrid grid{std::stoi(text.substr(0, sep)), std::stoi(text.substr(sep + 1))};
  if (grid.grid_h <= 0 || grid.grid_w <= 0 || grid.grid_h % 2 != 0 || grid.grid_w % 2 != 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid --bucket_grid '{}': values must be positive even numbers.", text);
  }
  return grid;
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

Tensor makeGridThwTensor(int32_t grid_t, int32_t grid_h, int32_t grid_w) {
  auto grid = Tensor::empty({1, 3}, mllm::kInt32, mllm::kCPU).alloc();
  grid.ptr<int32_t>()[0] = grid_t;
  grid.ptr<int32_t>()[1] = grid_h;
  grid.ptr<int32_t>()[2] = grid_w;
  return grid;
}

Tensor padVisualPatchesToBucket(Tensor img, Tensor grid_thw, const BucketGrid& bucket, const Qwen2VLConfig& cfg) {
  const auto* grid = grid_thw.ptr<int32_t>();
  const int32_t grid_t = grid[0];
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];
  if (bucket.grid_h < grid_h || bucket.grid_w < grid_w) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "Bucket {}x{} cannot cover original grid {}x{}.",
                    bucket.grid_h,
                    bucket.grid_w,
                    grid_h,
                    grid_w);
  }
  if (grid_h == bucket.grid_h && grid_w == bucket.grid_w) { return img; }

  const int32_t patch_dim = img.shape()[1];
  const auto elem_bytes = mllm::bytesOfType(img.dtype());
  auto padded = Tensor::empty({grid_t * bucket.grid_h * bucket.grid_w, patch_dim}, img.dtype(), mllm::kCPU).alloc();
  std::memset(padded.ptr<char>(), 0, padded.bytes());

  for (int32_t t = 0; t < grid_t; ++t) {
    for (int32_t h = 0; h < grid_h; ++h) {
      for (int32_t w = 0; w < grid_w; ++w) {
        const auto src_idx = visualPatchIndex(t, h, w, grid_h, grid_w, cfg.visual_spatial_merge_size);
        const auto dst_idx = visualPatchIndex(t, h, w, bucket.grid_h, bucket.grid_w, cfg.visual_spatial_merge_size);
        std::memcpy(padded.offsettedPtr<char>({static_cast<int32_t>(dst_idx), 0}),
                    img.offsettedPtr<char>({static_cast<int32_t>(src_idx), 0}),
                    static_cast<size_t>(patch_dim) * elem_bytes);
      }
    }
  }
  return padded;
}

Tensor cropVisualEmbeddingsFromBucket(Tensor bucket_embeddings,
                                      Tensor original_grid_thw,
                                      Tensor bucket_grid_thw,
                                      const Qwen2VLConfig& cfg) {
  const auto* original_grid = original_grid_thw.ptr<int32_t>();
  const auto* bucket_grid = bucket_grid_thw.ptr<int32_t>();
  const int32_t grid_t = original_grid[0];
  const int32_t original_h = original_grid[1];
  const int32_t original_w = original_grid[2];
  const int32_t bucket_h = bucket_grid[1];
  const int32_t bucket_w = bucket_grid[2];
  if (original_h == bucket_h && original_w == bucket_w) { return bucket_embeddings; }

  const int32_t merge = cfg.visual_spatial_merge_size;
  const int32_t hidden_size = bucket_embeddings.shape()[1];
  const auto elem_bytes = mllm::bytesOfType(bucket_embeddings.dtype());
  const int32_t original_merged_h = original_h / merge;
  const int32_t original_merged_w = original_w / merge;
  auto cropped = Tensor::empty({grid_t * original_merged_h * original_merged_w, hidden_size}, bucket_embeddings.dtype(), mllm::kCPU).alloc();

  for (int32_t t = 0; t < grid_t; ++t) {
    for (int32_t h = 0; h < original_merged_h; ++h) {
      for (int32_t w = 0; w < original_merged_w; ++w) {
        const auto src_idx = visualMergedIndex(t, h, w, bucket_h, bucket_w, merge);
        const auto dst_idx = visualMergedIndex(t, h, w, original_h, original_w, merge);
        std::memcpy(cropped.offsettedPtr<char>({static_cast<int32_t>(dst_idx), 0}),
                    bucket_embeddings.offsettedPtr<char>({static_cast<int32_t>(src_idx), 0}),
                    static_cast<size_t>(hidden_size) * elem_bytes);
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
  return mllm::models::qwen2vl::makeVisualRotarySinCos(pos_emb);
}

Tensor runVisual(const std::string& model_path,
                 const std::string& model_version,
                 const Qwen2VLConfig& cfg,
                 Tensor img,
                 Tensor sin,
                 Tensor cos) {
  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV2;
  if (model_version == "v1") { file_version = mllm::ModelFileVersion::kV1; }
  auto params = mllm::load(model_path, file_version);
  auto visual = mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel("visual", cfg);
  visual.load(params);
  return visual(img, sin, cos)[0];
}

CompareStats compareFloatTensors(Tensor test_tensor, Tensor ref_tensor) {
  MLLM_RT_ASSERT(test_tensor.shape() == ref_tensor.shape());
  MLLM_RT_ASSERT(test_tensor.dtype() == ref_tensor.dtype());
  CompareStats stats;
  stats.numel = test_tensor.numel();
  double dot = 0.0;
  double test_norm2 = 0.0;
  double ref_norm2 = 0.0;
  double diff_norm2 = 0.0;
  float max_abs = 0.0f;

  auto accumulate = [&](auto* test, auto* ref) {
    for (int64_t i = 0; i < stats.numel; ++i) {
      const double a = static_cast<double>(test[i]);
      const double b = static_cast<double>(ref[i]);
      if (std::isnan(a) || std::isnan(b)) {
        ++stats.nan_count;
        continue;
      }
      if (std::isinf(a) || std::isinf(b)) {
        ++stats.inf_count;
        continue;
      }
      ++stats.finite_count;
      const double diff = a - b;
      dot += a * b;
      test_norm2 += a * a;
      ref_norm2 += b * b;
      diff_norm2 += diff * diff;
      max_abs = std::max(max_abs, static_cast<float>(std::abs(diff)));
    }
  };

  switch (test_tensor.dtype()) {
    case mllm::kFloat32: accumulate(test_tensor.ptr<float>(), ref_tensor.ptr<float>()); break;
    case mllm::kFloat16: accumulate(test_tensor.ptr<mllm::mllm_fp16_t>(), ref_tensor.ptr<mllm::mllm_fp16_t>()); break;
    default:
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Unsupported comparison dtype: {}", mllm::nameOfType(test_tensor.dtype()));
  }

  if (test_norm2 == 0.0 && ref_norm2 == 0.0 && diff_norm2 == 0.0) {
    stats.cosine = 1.0;
    stats.rel_l2 = 0.0;
    stats.norm_ratio = 1.0;
    stats.max_abs_diff = 0.0f;
    return stats;
  }

  constexpr double eps = 1e-12;
  const double test_l2 = std::sqrt(test_norm2);
  stats.ref_l2 = std::sqrt(ref_norm2);
  stats.l2 = std::sqrt(diff_norm2);
  stats.cosine = dot / (test_l2 * stats.ref_l2 + eps);
  stats.rel_l2 = stats.l2 / (stats.ref_l2 + eps);
  stats.norm_ratio = test_l2 / (stats.ref_l2 + eps);
  stats.max_abs_diff = max_abs;
  return stats;
}

void printShape(const std::string& name, const Tensor& tensor) {
  fmt::print("{} dtype={} shape=[", name, mllm::nameOfType(tensor.dtype()));
  for (size_t i = 0; i < tensor.shape().size(); ++i) { fmt::print("{}{}", i == 0 ? "" : ", ", tensor.shape()[i]); }
  fmt::print("]\n");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("visual-capable .mllm model").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("model file version: v1/v2").def("v2");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Qwen2-VL config path").required(true);
  auto& image_path = Argparse::add<std::string>("-i|--image").help("input image path").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("prompt text").def("describe this picture");
  auto& bucket_grid = Argparse::add<std::string>("--bucket_grid").help("bucket patch grid HxW, e.g. 12x16").required(true);

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  auto cfg = Qwen2VLConfig(config_path.get());
  auto tokenizer = Qwen2VLTokenizer(tokenizer_path.get());
  auto inputs = tokenizer.convertMessage({.prompt = prompt.get(), .img_file_path = image_path.get()});
  auto img = inputs.at("img");
  auto grid_thw = inputs.at("grid_thw");
  const auto* grid = grid_thw.ptr<int32_t>();
  const auto bucket = parseBucketGrid(bucket_grid.get());
  auto bucket_grid_thw = makeGridThwTensor(grid[0], bucket.grid_h, bucket.grid_w);
  auto padded_img = padVisualPatchesToBucket(img, grid_thw, bucket, cfg);

  fmt::print("Qwen2-VL visual padding diagnostic\n");
  fmt::print("original grid: t={} h={} w={} patch_tokens={}\n", grid[0], grid[1], grid[2], img.shape()[0]);
  fmt::print("bucket grid  : t={} h={} w={} patch_tokens={}\n", grid[0], bucket.grid_h, bucket.grid_w, padded_img.shape()[0]);
  printShape("img", img);
  printShape("padded_img", padded_img);

  auto [sin, cos] = makeVisualSinCos(cfg, grid_thw, img.shape()[0]);
  auto [bucket_sin, bucket_cos] = makeVisualSinCos(cfg, bucket_grid_thw, padded_img.shape()[0]);

  auto ref_embeddings = runVisual(model_path.get(), model_version.get(), cfg, img, sin, cos);
  auto bucket_embeddings = runVisual(model_path.get(), model_version.get(), cfg, padded_img, bucket_sin, bucket_cos);
  auto cropped_embeddings = cropVisualEmbeddingsFromBucket(bucket_embeddings, grid_thw, bucket_grid_thw, cfg);

  printShape("ref_embeddings", ref_embeddings);
  printShape("bucket_embeddings", bucket_embeddings);
  printShape("cropped_embeddings", cropped_embeddings);

  auto stats = compareFloatTensors(cropped_embeddings, ref_embeddings);
  fmt::print(fg(fmt::color::cyan), "\n{:=^58}\n", " Padding Contamination ");
  fmt::print("numel               {:>16}\n", stats.numel);
  fmt::print("finite count        {:>16}\n", stats.finite_count);
  fmt::print("NaN count           {:>16}\n", stats.nan_count);
  fmt::print("Inf count           {:>16}\n", stats.inf_count);
  fmt::print("cosine              {:>16.8f}\n", stats.cosine);
  fmt::print("relative L2         {:>16.8f}\n", stats.rel_l2);
  fmt::print("norm ratio          {:>16.8f}\n", stats.norm_ratio);
  fmt::print("ref L2              {:>16.8f}\n", stats.ref_l2);
  fmt::print("diff L2             {:>16.8f}\n", stats.l2);
  fmt::print("max abs diff        {:>16.6f}\n", stats.max_abs_diff);
  fmt::print(fg(fmt::color::cyan), "{:=^58}\n", "");
  return 0;
});
