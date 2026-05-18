// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot_rt/KVCacheManager.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTModule.hpp>
#include <mllm/core/DataTypes.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/preprocessor/tokenizers/Unicode.hpp>

using mllm::Argparse;
using mllm::Tensor;
using mllm::models::qwen2vl::Qwen2VLConfig;
using mllm::models::qwen2vl::Qwen2VLTokenizer;
using mllm::qnn::aot::KVCacheManager;
using mllm::qnn::aot::QnnAOTConfig;
using mllm::qnn::aot::QnnAOTModule;

namespace {

constexpr float kDefaultInputEmbeddingScale = 0.002563515f;
constexpr int32_t kDefaultInputEmbeddingZeroPoint = 15604;

struct QuantParams {
  float scale = 1.f;
  int32_t zero_point = 0;
};

struct LogitEntry {
  int32_t token_id = 0;
  uint16_t raw = 0;
  float value = 0.f;
};

struct RuntimeIO {
  int32_t ar_len = 1;
  std::unique_ptr<QnnAOTModule> module;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  bool dump_block_outputs = false;
  bool dump_layer0_outputs = false;
};

using QnnAOTModuleCache = std::unordered_map<std::string, std::unique_ptr<QnnAOTModule>>;

struct PromptFeatures {
  Tensor sequence;
  Tensor position_ids;
  Tensor llm_embedding_sin;
  Tensor llm_embedding_cos;
  Tensor visual_embeddings;
  int32_t vision_token_start = -1;
};

struct VisualBucketGrid {
  int32_t grid_h = 0;
  int32_t grid_w = 0;

  [[nodiscard]] int32_t patchTokens() const { return grid_h * grid_w; }
};

struct VisualResizeOverride {
  bool enabled = false;
  int32_t native_grid_h = 0;
  int32_t native_grid_w = 0;
  int32_t resize_grid_h = 0;
  int32_t resize_grid_w = 0;
  VisualBucketGrid bucket;
};

std::string trimInteractiveLine(std::string line) {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || std::isspace(static_cast<unsigned char>(line.back())))) {
    line.pop_back();
  }
  size_t start = 0;
  while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) { ++start; }
  if (start > 0) { line.erase(0, start); }
  return line;
}

float readScalarFloat(const mllm::ParameterFile::ptr_t& params, const std::string& name) {
  return params->pull(name).ptr<float>()[0];
}

int32_t readScalarInt32(const mllm::ParameterFile::ptr_t& params, const std::string& name) {
  return params->pull(name).ptr<int32_t>()[0];
}

QuantParams readQuantParams(const mllm::ParameterFile::ptr_t& params,
                            const std::string& scale_name,
                            const std::string& zp_name) {
  return {.scale = readScalarFloat(params, scale_name), .zero_point = readScalarInt32(params, zp_name)};
}

uint16_t quantizeUInt16(float value, const QuantParams& qp) {
  auto q = static_cast<int64_t>(std::llround(value / qp.scale) + qp.zero_point);
  q = std::max<int64_t>(0, std::min<int64_t>(65535, q));
  return static_cast<uint16_t>(q);
}

bool shouldDumpQwen2VLAOTStats() {
  const char* flag = std::getenv("MLLM_QWEN2VL_AOT_DUMP_STATS");
  return flag != nullptr && std::string(flag) != "0";
}

std::vector<VisualBucketGrid> parseVisualBucketGrids(const std::string& text) {
  std::vector<VisualBucketGrid> buckets;
  if (text.empty()) { return buckets; }

  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
    if (item.empty()) { continue; }

    auto sep = item.find('x');
    if (sep == std::string::npos) { sep = item.find('X'); }
    if (sep == std::string::npos) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid visual bucket grid '{}', expected HxW.", item);
    }

    const int32_t grid_h = std::stoi(item.substr(0, sep));
    const int32_t grid_w = std::stoi(item.substr(sep + 1));
    if (grid_h <= 0 || grid_w <= 0 || grid_h % 2 != 0 || grid_w % 2 != 0) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                      "Invalid visual bucket grid '{}': grid_h/grid_w must be positive even numbers.",
                      item);
    }
    buckets.push_back({grid_h, grid_w});
  }
  return buckets;
}

std::string visualGraphSuffixForPatchTokens(int32_t patch_tokens) { return "_s" + std::to_string(patch_tokens); }

int32_t findCoveringVisualBucketIndex(int32_t grid_h, int32_t grid_w, const std::vector<VisualBucketGrid>& buckets) {
  int32_t best_idx = -1;
  int64_t best_area = 0;
  int64_t best_padding = 0;
  for (int32_t i = 0; i < static_cast<int32_t>(buckets.size()); ++i) {
    const auto& bucket = buckets[i];
    if (bucket.grid_h < grid_h || bucket.grid_w < grid_w) { continue; }
    const int64_t area = static_cast<int64_t>(bucket.grid_h) * bucket.grid_w;
    const int64_t padding = area - static_cast<int64_t>(grid_h) * grid_w;
    if (best_idx < 0 || area < best_area || (area == best_area && padding < best_padding)) {
      best_idx = i;
      best_area = area;
      best_padding = padding;
    }
  }
  return best_idx;
}

int32_t floorEvenGrid(double value) {
  int32_t grid = static_cast<int32_t>(std::floor(value + 1e-6));
  grid -= grid % 2;
  return std::max(4, grid);
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

VisualBucketGrid selectVisualBucket(Tensor grid_thw, const std::vector<VisualBucketGrid>& buckets) {
  const auto* grid = grid_thw.ptr<int32_t>();
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];

  const int32_t best_idx = findCoveringVisualBucketIndex(grid_h, grid_w, buckets);

  if (best_idx < 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "No visual bucket can cover grid {}x{}. Add a larger --visual_bucket_grids entry.",
                    grid_h,
                    grid_w);
  }
  return buckets[best_idx];
}

VisualResizeOverride chooseVisualResizeOverrideForOversizedImage(const std::string& image_path,
                                                                 const std::vector<VisualBucketGrid>& buckets) {
  VisualResizeOverride override;
  if (buckets.empty()) { return override; }

  auto image = mllm::Image::open(image_path);
  mllm::models::qwen2vl::Qwen2VLImagePreprocessor image_preprocessor(56 * 56, 28 * 28 * 256);
  auto [native_h_px, native_w_px] = image_preprocessor.smartResize(image.h(), image.w(), 28, 56 * 56, 28 * 28 * 256);
  const int32_t native_grid_h = native_h_px / 14;
  const int32_t native_grid_w = native_w_px / 14;
  if (findCoveringVisualBucketIndex(native_grid_h, native_grid_w, buckets) >= 0) { return override; }

  int32_t max_patch_tokens = 0;
  for (const auto& bucket : buckets) { max_patch_tokens = std::max(max_patch_tokens, bucket.patchTokens()); }

  int32_t best_idx = -1;
  double best_scale = -1.0;
  double best_aspect_error = 0.0;
  const double native_aspect = static_cast<double>(native_grid_h) / native_grid_w;
  for (int32_t i = 0; i < static_cast<int32_t>(buckets.size()); ++i) {
    const auto& bucket = buckets[i];
    if (bucket.patchTokens() != max_patch_tokens) { continue; }
    const double scale = std::min(static_cast<double>(bucket.grid_h) / native_grid_h,
                                  static_cast<double>(bucket.grid_w) / native_grid_w);
    const double bucket_aspect = static_cast<double>(bucket.grid_h) / bucket.grid_w;
    const double aspect_error = std::abs(std::log(bucket_aspect / native_aspect));
    if (best_idx < 0 || scale > best_scale || (scale == best_scale && aspect_error < best_aspect_error)) {
      best_idx = i;
      best_scale = scale;
      best_aspect_error = aspect_error;
    }
  }

  if (best_idx < 0 || best_scale <= 0.0) { return override; }

  const auto& bucket = buckets[best_idx];
  int32_t resize_grid_h = floorEvenGrid(native_grid_h * best_scale);
  int32_t resize_grid_w = floorEvenGrid(native_grid_w * best_scale);
  while (resize_grid_h > bucket.grid_h) { resize_grid_h -= 2; }
  while (resize_grid_w > bucket.grid_w) { resize_grid_w -= 2; }
  resize_grid_h = std::max(4, resize_grid_h);
  resize_grid_w = std::max(4, resize_grid_w);

  override.enabled = true;
  override.native_grid_h = native_grid_h;
  override.native_grid_w = native_grid_w;
  override.resize_grid_h = resize_grid_h;
  override.resize_grid_w = resize_grid_w;
  override.bucket = bucket;
  return override;
}

Tensor padVisualPatchesToBucket(Tensor img, Tensor grid_thw, const VisualBucketGrid& bucket, const Qwen2VLConfig& cfg) {
  const auto* grid = grid_thw.ptr<int32_t>();
  const int32_t grid_t = grid[0];
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];
  if (grid_h == bucket.grid_h && grid_w == bucket.grid_w) { return img; }

  const int32_t patch_dim = img.shape()[1];
  auto padded = Tensor::empty({grid_t * bucket.grid_h * bucket.grid_w, patch_dim}, img.dtype(), mllm::kCPU).alloc();
  std::fill(padded.ptr<float>(), padded.ptr<float>() + padded.numel(), 0.0f);

  for (int32_t t = 0; t < grid_t; ++t) {
    for (int32_t h = 0; h < grid_h; ++h) {
      for (int32_t w = 0; w < grid_w; ++w) {
        const auto src_idx = visualPatchIndex(t, h, w, grid_h, grid_w, cfg.visual_spatial_merge_size);
        const auto dst_idx = visualPatchIndex(t, h, w, bucket.grid_h, bucket.grid_w, cfg.visual_spatial_merge_size);
        std::memcpy(padded.offsettedPtr<float>({static_cast<int32_t>(dst_idx), 0}),
                    img.offsettedPtr<float>({static_cast<int32_t>(src_idx), 0}),
                    patch_dim * sizeof(float));
      }
    }
  }
  return padded;
}

Tensor makeVisualAttentionMaskForBucket(Tensor original_grid_thw, const VisualBucketGrid& bucket, const Qwen2VLConfig& cfg) {
  const auto* grid = original_grid_thw.ptr<int32_t>();
  const int32_t grid_t = grid[0];
  const int32_t grid_h = grid[1];
  const int32_t grid_w = grid[2];
  const int32_t bucket_tokens = grid_t * bucket.grid_h * bucket.grid_w;

  auto mask = Tensor::empty({1, 1, 1, bucket_tokens}, mllm::kFloat32, mllm::kCPU).alloc();
  std::fill(mask.ptr<float>(), mask.ptr<float>() + mask.numel(), -10000.0f);

  for (int32_t t = 0; t < grid_t; ++t) {
    for (int32_t h = 0; h < grid_h; ++h) {
      for (int32_t w = 0; w < grid_w; ++w) {
        const auto idx = visualPatchIndex(t, h, w, bucket.grid_h, bucket.grid_w, cfg.visual_spatial_merge_size);
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
  const int32_t original_merged_h = original_h / merge;
  const int32_t original_merged_w = original_w / merge;
  auto cropped = Tensor::empty({grid_t * original_merged_h * original_merged_w, hidden_size}, bucket_embeddings.dtype(), mllm::kCPU).alloc();

  for (int32_t t = 0; t < grid_t; ++t) {
    for (int32_t h = 0; h < original_merged_h; ++h) {
      for (int32_t w = 0; w < original_merged_w; ++w) {
        const auto src_idx = visualMergedIndex(t, h, w, bucket_h, bucket_w, merge);
        const auto dst_idx = visualMergedIndex(t, h, w, original_h, original_w, merge);
        std::memcpy(cropped.offsettedPtr<float>({static_cast<int32_t>(dst_idx), 0}),
                    bucket_embeddings.offsettedPtr<float>({static_cast<int32_t>(src_idx), 0}),
                    hidden_size * sizeof(float));
      }
    }
  }
  return cropped;
}

Tensor makeQnnTensor(const std::vector<int32_t>& shape, mllm::DataTypes dtype, const std::string& name) {
  auto tensor = Tensor::empty(shape, dtype, mllm::kQNN).setName(name).alloc();
  return tensor;
}

Tensor copyToQnn(const Tensor& cpu_tensor, const std::string& name) {
  auto qnn_tensor = makeQnnTensor(cpu_tensor.shape(), cpu_tensor.dtype(), name);
  std::memcpy(qnn_tensor.ptr<void>(), cpu_tensor.ptr<void>(), cpu_tensor.bytes());
  return qnn_tensor;
}

QnnAOTModule& getOrCreateCachedModule(QnnAOTModuleCache& cache, const std::string& graph_name) {
  auto it = cache.find(graph_name);
  if (it != cache.end()) { return *it->second; }

  auto module = std::make_unique<QnnAOTModule>(graph_name);
  module->to(mllm::kQNN);
  auto [inserted, _] = cache.emplace(graph_name, std::move(module));
  return *inserted->second;
}

std::vector<std::string> visualGraphNamesForLayout(const std::string& bundle_layout, const std::string& graph_suffix) {
  if (bundle_layout == "single") { return {"visual_full" + graph_suffix}; }
  if (bundle_layout == "6x8") {
    return {"visual_patch_embed" + graph_suffix,
            "visual_blocks_0_8" + graph_suffix,
            "visual_blocks_8_16" + graph_suffix,
            "visual_blocks_16_24" + graph_suffix,
            "visual_blocks_24_32" + graph_suffix,
            "visual_merger" + graph_suffix};
  }
  if (bundle_layout == "early2") {
    return {"visual_patch_embed" + graph_suffix,
            "visual_blocks_0_2" + graph_suffix,
            "visual_blocks_2_4" + graph_suffix,
            "visual_blocks_4_6" + graph_suffix,
            "visual_blocks_6_8" + graph_suffix,
            "visual_blocks_8_16" + graph_suffix,
            "visual_blocks_16_24" + graph_suffix,
            "visual_blocks_24_32" + graph_suffix,
            "visual_merger" + graph_suffix};
  }
  if (bundle_layout == "tail4") {
    return {"visual_patch_embed" + graph_suffix,
            "visual_blocks_0_8" + graph_suffix,
            "visual_blocks_8_16" + graph_suffix,
            "visual_blocks_16_20" + graph_suffix,
            "visual_blocks_20_24" + graph_suffix,
            "visual_blocks_24_28" + graph_suffix,
            "visual_blocks_28_32" + graph_suffix,
            "visual_merger" + graph_suffix};
  }

  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--visual_bundle_layout must be single, 6x8, early2 or tail4.");
}

void initializeVisualQnnModules(const std::string& bundle_layout,
                                const std::vector<VisualBucketGrid>& buckets,
                                QnnAOTModuleCache& cache) {
  std::vector<std::string> suffixes;
  if (buckets.empty()) {
    suffixes.push_back("");
  } else {
    std::unordered_set<int32_t> seen_patch_tokens;
    for (const auto& bucket : buckets) {
      if (seen_patch_tokens.insert(bucket.patchTokens()).second) {
        suffixes.push_back(visualGraphSuffixForPatchTokens(bucket.patchTokens()));
      }
    }
  }

  const auto before = cache.size();
  for (const auto& suffix : suffixes) {
    for (const auto& graph_name : visualGraphNamesForLayout(bundle_layout, suffix)) {
      (void)getOrCreateCachedModule(cache, graph_name);
    }
  }
  fmt::print("[Qwen2VL AOT] initialized {} visual QNN module wrappers (layout={}, bucket_suffixes={})\n",
             cache.size() - before,
             bundle_layout,
             suffixes.size());
}

Tensor runQnnGraph1(const std::string& graph_name,
                    Tensor hidden,
                    Tensor visual_embedding_sin,
                    Tensor visual_embedding_cos,
                    Tensor visual_attention_mask,
                    Tensor output,
                    QnnAOTModuleCache* module_cache = nullptr) {
  std::unique_ptr<QnnAOTModule> local_module;
  QnnAOTModule* module = nullptr;
  if (module_cache != nullptr) {
    module = &getOrCreateCachedModule(*module_cache, graph_name);
  } else {
    local_module = std::make_unique<QnnAOTModule>(graph_name);
    local_module->to(mllm::kQNN);
    module = local_module.get();
  }

  module->setOutputTensors({output});
  auto inputs = visual_attention_mask.isNil() ? std::vector<Tensor>{hidden, visual_embedding_sin, visual_embedding_cos}
                                              : std::vector<Tensor>{hidden,
                                                                    visual_embedding_sin,
                                                                    visual_embedding_cos,
                                                                    visual_attention_mask};
  auto outputs = (*module)(inputs);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);
  return outputs[0];
}

Tensor runVisualQnnBundle(Tensor img,
                          Tensor visual_embedding_sin,
                          Tensor visual_embedding_cos,
                          Tensor visual_attention_mask,
                          const Qwen2VLConfig& cfg,
                          const std::string& bundle_layout,
                          const std::string& graph_suffix,
                          QnnAOTModuleCache* module_cache = nullptr) {
  const int32_t visual_patch_tokens = img.shape()[0];
  const int32_t merged_tokens = visual_patch_tokens / (cfg.visual_spatial_merge_size * cfg.visual_spatial_merge_size);

  auto img_qnn = copyToQnn(img, "visual_img");
  auto sin_qnn = copyToQnn(visual_embedding_sin, "visual_embedding_sin");
  auto cos_qnn = copyToQnn(visual_embedding_cos, "visual_embedding_cos");
  auto mask_qnn = copyToQnn(visual_attention_mask, "visual_attention_mask");

  auto hidden_a = makeQnnTensor({visual_patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, "visual_hidden_a");
  auto hidden_b = makeQnnTensor({visual_patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, "visual_hidden_b");
  auto visual_embeddings = makeQnnTensor({merged_tokens, cfg.hidden_size}, mllm::kFloat32, "visual_embeddings");

  Tensor hidden = Tensor::nil();
  if (bundle_layout == "single") {
    auto output =
        runQnnGraph1("visual_full" + graph_suffix, img_qnn, sin_qnn, cos_qnn, mask_qnn, visual_embeddings, module_cache);
    return output.to(mllm::kCPU).clone();
  } else if (bundle_layout == "6x8") {
    hidden = runQnnGraph1("visual_patch_embed" + graph_suffix, img_qnn, sin_qnn, cos_qnn, Tensor::nil(), hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_0_8" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_8_16" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_16_24" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_24_32" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
  } else if (bundle_layout == "early2") {
    hidden = runQnnGraph1("visual_patch_embed" + graph_suffix, img_qnn, sin_qnn, cos_qnn, Tensor::nil(), hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_0_2" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_2_4" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_4_6" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_6_8" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_8_16" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_16_24" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_24_32" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
  } else if (bundle_layout == "tail4") {
    hidden = runQnnGraph1("visual_patch_embed" + graph_suffix, img_qnn, sin_qnn, cos_qnn, Tensor::nil(), hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_0_8" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_8_16" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_16_20" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_20_24" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
    hidden = runQnnGraph1("visual_blocks_24_28" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_b, module_cache);
    hidden = runQnnGraph1("visual_blocks_28_32" + graph_suffix, hidden, sin_qnn, cos_qnn, mask_qnn, hidden_a, module_cache);
  } else {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--visual_bundle_layout must be single, 6x8, early2 or tail4 for full runner.");
  }

  auto output = runQnnGraph1("visual_merger" + graph_suffix,
                             hidden,
                             sin_qnn,
                             cos_qnn,
                             Tensor::nil(),
                             visual_embeddings,
                             module_cache);
  return output.to(mllm::kCPU).clone();
}

int32_t findVisionTokenStart(const Tensor& sequence, int32_t vision_token_id) {
  auto* ids = sequence.ptr<int64_t>();
  for (int32_t i = 0; i < sequence.shape()[1]; ++i) {
    if (ids[i] == vision_token_id) { return i; }
  }
  return -1;
}

Tensor makePositionIdsPrefill(const Tensor& sequence, const Tensor& image_grid_thw, const Qwen2VLConfig& cfg) {
  MLLM_RT_ASSERT_EQ(sequence.shape().size(), 2);
  MLLM_RT_ASSERT_EQ(image_grid_thw.shape().size(), 2);
  MLLM_RT_ASSERT_EQ(sequence.shape()[0], 1);

  const int32_t seq_len = sequence.shape()[1];
  Tensor position_ids = Tensor::empty({3, 1, seq_len}, mllm::kInt64, mllm::kCPU).alloc();

  const int32_t vision_pad_start = findVisionTokenStart(sequence, cfg.vision_token_id);
  MLLM_RT_ASSERT(vision_pad_start >= 0);

  const int* grid = image_grid_thw.ptr<int32_t>();
  const int32_t img_t = grid[0];
  const int32_t img_h = grid[1];
  const int32_t img_w = grid[2];
  const int32_t inputs_t = img_t;
  const int32_t inputs_h = img_h / cfg.visual_spatial_merge_size;
  const int32_t inputs_w = img_w / cfg.visual_spatial_merge_size;
  const int32_t vision_tokens = inputs_t * inputs_h * inputs_w;

  int64_t current_max_position_id = 0;
  for (int32_t d = 0; d < 3; ++d) {
    auto* ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
    for (int64_t k = 0; k < vision_pad_start; ++k) { ptr[k] = k; }
  }
  current_max_position_id = vision_pad_start - 1;

  int32_t cnt = 0;
  const int64_t vision_start_id = current_max_position_id + 1;
  for (int32_t ti = 0; ti < inputs_t; ++ti) {
    for (int32_t hi = 0; hi < inputs_h; ++hi) {
      for (int32_t wi = 0; wi < inputs_w; ++wi) {
        *position_ids.offsettedPtr<int64_t>({0, 0, vision_pad_start + cnt}) = vision_start_id + ti;
        *position_ids.offsettedPtr<int64_t>({1, 0, vision_pad_start + cnt}) = vision_start_id + hi;
        *position_ids.offsettedPtr<int64_t>({2, 0, vision_pad_start + cnt}) = vision_start_id + wi;
        ++cnt;
      }
    }
  }

  const int64_t dim_0_tail = *position_ids.offsettedPtr<int64_t>({0, 0, vision_pad_start + vision_tokens - 1});
  const int64_t dim_1_tail = *position_ids.offsettedPtr<int64_t>({1, 0, vision_pad_start + vision_tokens - 1});
  const int64_t dim_2_tail = *position_ids.offsettedPtr<int64_t>({2, 0, vision_pad_start + vision_tokens - 1});
  current_max_position_id = std::max({dim_0_tail, dim_1_tail, dim_2_tail});

  const int64_t trailing_text_start = vision_pad_start + vision_tokens;
  const int64_t trailing_text_count = seq_len - trailing_text_start;
  if (trailing_text_count > 0) {
    const int64_t start_id = current_max_position_id + 1;
    for (int32_t d = 0; d < 3; ++d) {
      auto* ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
      for (int64_t k = 0; k < trailing_text_count; ++k) { ptr[trailing_text_start + k] = start_id + k; }
    }
  }

  return position_ids;
}

Tensor makePositionIdsDecode(Tensor prev_position_ids) {
  const int32_t last_idx = prev_position_ids.shape()[2] - 1;
  const auto last_pos = *prev_position_ids.offsettedPtr<int64_t>({0, 0, last_idx});
  auto ret = Tensor::empty({3, 1, 1}, mllm::kInt64, mllm::kCPU).alloc();
  for (int32_t d = 0; d < 3; ++d) { *ret.offsettedPtr<int64_t>({d, 0, 0}) = last_pos + 1; }
  return ret;
}

void fillQuantizedFloatChunk(Tensor src,
                             int32_t src_start,
                             int32_t valid_len,
                             Tensor& dst,
                             const QuantParams& qp,
                             int32_t feature_dim) {
  auto* out = dst.ptr<uint16_t>();
  std::fill(out, out + dst.numel(), quantizeUInt16(0.f, qp));

  for (int32_t s = 0; s < valid_len; ++s) {
    const float* in_row = src.offsettedPtr<float>({src_start + s, 0});
    uint16_t* out_row = out + s * feature_dim;
    for (int32_t d = 0; d < feature_dim; ++d) { out_row[d] = quantizeUInt16(in_row[d], qp); }
  }
}

void fillEmbeddingChunk(Tensor sequence,
                        Tensor embedding_weight,
                        Tensor visual_embeddings,
                        int32_t vision_start,
                        int32_t chunk_start,
                        int32_t valid_len,
                        Tensor& dst,
                        const QuantParams& embedding_weight_qp,
                        const QuantParams& input_embedding_qp,
                        int32_t hidden_size) {
  auto* out = dst.ptr<uint16_t>();
  const auto* ids = sequence.ptr<int64_t>();
  const auto* weight = embedding_weight.ptr<uint16_t>();
  std::fill(out, out + dst.numel(), quantizeUInt16(0.f, input_embedding_qp));

  const bool same_qp = embedding_weight_qp.scale == input_embedding_qp.scale
                       && embedding_weight_qp.zero_point == input_embedding_qp.zero_point;

  for (int32_t s = 0; s < valid_len; ++s) {
    const int32_t seq_idx = chunk_start + s;
    uint16_t* out_row = out + s * hidden_size;
    const int64_t token_id = ids[seq_idx];
    const int32_t visual_idx = seq_idx - vision_start;
    if (!visual_embeddings.isNil() && visual_idx >= 0 && visual_idx < visual_embeddings.shape()[0]) {
      const float* visual_row = visual_embeddings.offsettedPtr<float>({visual_idx, 0});
      for (int32_t d = 0; d < hidden_size; ++d) { out_row[d] = quantizeUInt16(visual_row[d], input_embedding_qp); }
    } else {
      const uint16_t* weight_row = weight + token_id * hidden_size;
      if (same_qp) {
        std::memcpy(out_row, weight_row, hidden_size * sizeof(uint16_t));
      } else {
        for (int32_t d = 0; d < hidden_size; ++d) {
          const float value = (static_cast<int32_t>(weight_row[d]) - embedding_weight_qp.zero_point) * embedding_weight_qp.scale;
          out_row[d] = quantizeUInt16(value, input_embedding_qp);
        }
      }
    }
  }
}

void initQwen2VLIO(RuntimeIO& io,
                   const QnnAOTConfig& cfg,
                   KVCacheManager<uint8_t>& kv_manager_u8,
                   KVCacheManager<uint16_t>* key_cache_manager_u16,
                   bool key_cache_uint16,
                   int32_t hidden_size,
                   int32_t intermediate_size) {
  io.module = std::make_unique<QnnAOTModule>("model.0.s" + std::to_string(io.ar_len));
  io.module->to(mllm::kQNN);

  io.inputs.clear();
  io.outputs.clear();
  io.inputs.reserve(4 + 2 * cfg.num_layers);
  const int32_t attention_heads = hidden_size / cfg.head_dim;
  io.outputs.reserve(1 + 2 * cfg.num_layers + (io.dump_block_outputs ? cfg.num_layers : 0)
                     + (io.dump_layer0_outputs ? 16 + 4 * attention_heads : 0));

  io.inputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy, "input_embeddings"));
  io.inputs.push_back(makeQnnTensor({1, io.ar_len, cfg.head_dim}, mllm::kUInt16PerTensorAsy, "llm_embedding_sin"));
  io.inputs.push_back(makeQnnTensor({1, io.ar_len, cfg.head_dim}, mllm::kUInt16PerTensorAsy, "llm_embedding_cos"));
  io.inputs.push_back(makeQnnTensor({1, 1, io.ar_len, cfg.context_len}, mllm::kUInt16PerTensorAsy, "causal_mask"));

  const auto& v_caches = kv_manager_u8.getVCache();
  if (key_cache_uint16) {
    MLLM_RT_ASSERT(key_cache_manager_u16 != nullptr);
    const auto& k_caches = key_cache_manager_u16->getKCache();
    for (int32_t l = 0; l < cfg.num_layers; ++l) {
      auto k_tensor =
          Tensor::empty({1, cfg.num_heads, cfg.head_dim, cfg.context_len - io.ar_len}, mllm::kUInt16PerTensorSym,
                        mllm::kQNN);
      k_tensor.impl()->storage()->ptr_ = k_caches[l].buffer;
      k_tensor.impl()->storage()->mem_type_ = mllm::kManual;
      k_tensor.setName("past_key_" + std::to_string(l));
      io.inputs.push_back(k_tensor);
    }
  } else {
    const auto& k_caches = kv_manager_u8.getKCache();
    for (int32_t l = 0; l < cfg.num_layers; ++l) {
      auto k_tensor =
          Tensor::empty({1, cfg.num_heads, cfg.head_dim, cfg.context_len - io.ar_len}, mllm::kUInt8, mllm::kQNN);
      k_tensor.impl()->storage()->ptr_ = k_caches[l].buffer;
      k_tensor.impl()->storage()->mem_type_ = mllm::kManual;
      k_tensor.setName("past_key_" + std::to_string(l));
      io.inputs.push_back(k_tensor);
    }
  }
  for (int32_t l = 0; l < cfg.num_layers; ++l) {
    auto v_tensor = Tensor::empty({1, cfg.num_heads, cfg.context_len - io.ar_len, cfg.head_dim}, mllm::kUInt8, mllm::kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].buffer;
    v_tensor.impl()->storage()->mem_type_ = mllm::kManual;
    v_tensor.setName("past_value_" + std::to_string(l));
    io.inputs.push_back(v_tensor);
  }

  io.outputs.push_back(makeQnnTensor({1, 1, io.ar_len, cfg.vocab_size}, mllm::kUInt16PerTensorAsy, "logits"));
  if (key_cache_uint16) {
    MLLM_RT_ASSERT(key_cache_manager_u16 != nullptr);
    const auto& k_caches = key_cache_manager_u16->getKCache();
    for (int32_t l = 0; l < cfg.num_layers; ++l) {
      auto k_tensor =
          Tensor::empty({1, cfg.num_heads, cfg.head_dim, io.ar_len}, mllm::kUInt16PerTensorSym, mllm::kQNN);
      k_tensor.impl()->storage()->ptr_ = k_caches[l].output_buffer;
      k_tensor.impl()->storage()->mem_type_ = mllm::kManual;
      k_tensor.setName("present_key_" + std::to_string(l));
      io.outputs.push_back(k_tensor);
    }
  } else {
    const auto& k_caches = kv_manager_u8.getKCache();
    for (int32_t l = 0; l < cfg.num_layers; ++l) {
      auto k_tensor = Tensor::empty({1, cfg.num_heads, cfg.head_dim, io.ar_len}, mllm::kUInt8, mllm::kQNN);
      k_tensor.impl()->storage()->ptr_ = k_caches[l].output_buffer;
      k_tensor.impl()->storage()->mem_type_ = mllm::kManual;
      k_tensor.setName("present_key_" + std::to_string(l));
      io.outputs.push_back(k_tensor);
    }
  }
  for (int32_t l = 0; l < cfg.num_layers; ++l) {
    auto v_tensor = Tensor::empty({1, cfg.num_heads, io.ar_len, cfg.head_dim}, mllm::kUInt8, mllm::kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].output_buffer;
    v_tensor.impl()->storage()->mem_type_ = mllm::kManual;
    v_tensor.setName("present_value_" + std::to_string(l));
    io.outputs.push_back(v_tensor);
  }
  if (io.dump_block_outputs) {
    for (int32_t l = 0; l < cfg.num_layers; ++l) {
      io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                         "block_out_" + std::to_string(l)));
    }
  }
  if (io.dump_layer0_outputs) {
    const int32_t kv_dim = cfg.num_heads * cfg.head_dim;
    const auto push_attn_head_outputs = [&](const std::string& base_name) {
      for (int32_t h = 0; h < attention_heads; ++h) {
        io.outputs.push_back(makeQnnTensor({1, 1, io.ar_len, cfg.context_len}, mllm::kUInt16PerTensorAsy,
                                           base_name + "_h" + std::to_string(h)));
      }
    };
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_input_layernorm_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_q_proj_out_flat"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, kv_dim}, mllm::kUInt16PerTensorAsy, "layer0_k_proj_out_flat"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, kv_dim}, mllm::kUInt16PerTensorAsy, "layer0_v_proj_out_flat"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, kv_dim}, mllm::kUInt8, "layer0_v_cache_out_flat"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_q_rope_out_flat"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, kv_dim}, mllm::kUInt16PerTensorAsy, "layer0_k_rope_out_flat"));
    push_attn_head_outputs("layer0_qk_matmul_out_flat");
    push_attn_head_outputs("layer0_qk_scaled_out_flat");
    push_attn_head_outputs("layer0_qk_masked_out_flat");
    push_attn_head_outputs("layer0_softmax_out_flat");
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_attn_value_out_flat"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy, "layer0_o_proj_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_post_attention_layernorm_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, intermediate_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_mlp_up_proj_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, intermediate_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_mlp_gate_proj_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, intermediate_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_mlp_gate_act_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, intermediate_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_mlp_down_proj_input"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy,
                                       "layer0_mlp_down_proj_out"));
    io.outputs.push_back(makeQnnTensor({1, io.ar_len, hidden_size}, mllm::kUInt16PerTensorAsy, "layer0_block_out"));
  }
}

int64_t sampleGreedyFromLogitsAt(Tensor& logits, int32_t token_idx) {
  auto logits_cpu = logits.to(mllm::kCPU);
  auto* data = logits_cpu.ptr<uint16_t>();
  const int32_t vocab = logits_cpu.shape().back();
  const auto& shape = logits_cpu.shape();
  if (shape.size() == 4) {
    MLLM_RT_ASSERT(token_idx >= 0 && token_idx < shape[2]);
    data += token_idx * vocab;
  } else if (shape.size() == 3) {
    MLLM_RT_ASSERT(token_idx >= 0 && token_idx < shape[1]);
    data += token_idx * vocab;
  } else {
    MLLM_RT_ASSERT_EQ(logits_cpu.numel(), vocab);
  }
  auto max_it = std::max_element(data, data + vocab);
  return static_cast<int64_t>(std::distance(data, max_it));
}

std::vector<LogitEntry> topKFromLogitsAt(Tensor& logits, int32_t token_idx, int32_t k, const QuantParams& qp) {
  auto logits_cpu = logits.to(mllm::kCPU);
  auto* data = logits_cpu.ptr<uint16_t>();
  const int32_t vocab = logits_cpu.shape().back();
  const auto& shape = logits_cpu.shape();
  if (shape.size() == 4) {
    MLLM_RT_ASSERT(token_idx >= 0 && token_idx < shape[2]);
    data += token_idx * vocab;
  } else if (shape.size() == 3) {
    MLLM_RT_ASSERT(token_idx >= 0 && token_idx < shape[1]);
    data += token_idx * vocab;
  } else {
    MLLM_RT_ASSERT_EQ(logits_cpu.numel(), vocab);
  }

  k = std::max(0, std::min(k, vocab));
  std::vector<int32_t> indices(vocab);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int32_t lhs, int32_t rhs) {
    if (data[lhs] == data[rhs]) { return lhs < rhs; }
    return data[lhs] > data[rhs];
  });

  std::vector<LogitEntry> ret;
  ret.reserve(k);
  for (int32_t i = 0; i < k; ++i) {
    const int32_t token_id = indices[i];
    const uint16_t raw = data[token_id];
    ret.push_back({.token_id = token_id, .raw = raw, .value = (static_cast<int32_t>(raw) - qp.zero_point) * qp.scale});
  }
  return ret;
}

class Qwen2VLAOTRunner {
 public:
  Qwen2VLAOTRunner(const Qwen2VLConfig& qnn_cfg,
                   const Qwen2VLConfig& visual_cfg,
                   const mllm::ParameterFile::ptr_t& qnn_params,
                   int32_t ar_len,
                   int32_t context_len,
                   float input_embedding_scale,
                   int32_t input_embedding_zero_point,
                   bool dump_block_outputs,
                   bool dump_layer0_outputs,
                   bool dump_visual_tokens,
                   bool key_cache_uint16,
                   bool visual_qnn,
                   std::string visual_bundle_layout,
                   std::vector<VisualBucketGrid> visual_bucket_grids,
                   int32_t dump_logits_topk,
                   std::vector<int32_t> dump_token_indices,
                   std::string dump_path)
      : qnn_cfg_(qnn_cfg),
        visual_cfg_(visual_cfg),
        qnn_params_(qnn_params),
        dump_block_outputs_(dump_block_outputs),
        dump_layer0_outputs_(dump_layer0_outputs),
        dump_visual_tokens_(dump_visual_tokens),
        key_cache_uint16_(key_cache_uint16),
        visual_qnn_(visual_qnn),
        visual_bundle_layout_(std::move(visual_bundle_layout)),
        visual_bucket_grids_(std::move(visual_bucket_grids)),
        dump_logits_topk_(dump_logits_topk),
        dump_token_indices_(std::move(dump_token_indices)),
        dump_path_(std::move(dump_path)) {
    config_.num_layers = qnn_cfg_.num_hidden_layers;
    config_.num_heads = qnn_cfg_.num_key_value_heads;
    config_.head_dim = qnn_cfg_.hidden_size / qnn_cfg_.num_attention_heads;
    config_.vocab_size = qnn_cfg_.vocab_size;
    config_.context_len = context_len;
    config_.ar_len = ar_len;
    config_.kv_dtype = mllm::kUInt8;
    config_.max_cache_len = context_len - 1;
    config_.max_ar_len = std::max(1, ar_len);
    config_.sliding_window = context_len;

    embedding_weight_qp_ = readQuantParams(qnn_params_, "model.embed_tokens.scale", "model.embed_tokens.zero_point");
    input_embedding_qp_ = embedding_weight_qp_;
    if (input_embedding_scale > 0.0f && input_embedding_zero_point >= 0) {
      input_embedding_qp_ = {.scale = input_embedding_scale, .zero_point = input_embedding_zero_point};
      fmt::print("[Qwen2VL AOT] override input_embeddings quant params: scale={:.9f}, zero_point={}\n",
                 input_embedding_qp_.scale, input_embedding_qp_.zero_point);
    }
    sin_qp_ = readQuantParams(qnn_params_, "model.sin_embedding_input_qdq.fake_quant.scale",
                              "model.sin_embedding_input_qdq.fake_quant.zero_point");
    cos_qp_ = readQuantParams(qnn_params_, "model.cos_embedding_input_qdq.fake_quant.scale",
                              "model.cos_embedding_input_qdq.fake_quant.zero_point");
    logits_qp_ =
        readQuantParams(qnn_params_, "lm_head_output_qdq.fake_quant.scale", "lm_head_output_qdq.fake_quant.zero_point");
    block_out_qps_.reserve(qnn_cfg_.num_hidden_layers);
    for (int32_t l = 0; l < qnn_cfg_.num_hidden_layers; ++l) {
      if (l + 1 < qnn_cfg_.num_hidden_layers) {
        block_out_qps_.push_back(readQuantParams(
            qnn_params_, "model.layers." + std::to_string(l + 1) + ".input_layernorm_input_qdq.fake_quant.scale",
            "model.layers." + std::to_string(l + 1) + ".input_layernorm_input_qdq.fake_quant.zero_point"));
      } else {
        block_out_qps_.push_back(readQuantParams(qnn_params_, "model.norm_input_qdq.fake_quant.scale",
                                                 "model.norm_input_qdq.fake_quant.zero_point"));
      }
    }
    initLayer0QuantParams();
    embedding_weight_ = qnn_params_->pull("model.embed_tokens.weight");
  }

  bool load() {
    auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
    if (!backend) {
      MLLM_ERROR("QNN Backend not found");
      return false;
    }

    kv_manager_u8_ = std::make_unique<KVCacheManager<uint8_t>>(config_);
    kv_manager_u8_->initCache(backend->allocator().get(), config_.ar_len);
    if (key_cache_uint16_) {
      key_cache_manager_u16_ = std::make_unique<KVCacheManager<uint16_t>>(config_);
      key_cache_manager_u16_->initCache(backend->allocator().get(), config_.ar_len);
      fmt::print("[Qwen2VL AOT] experimental key cache dtype: uint16, value cache dtype: uint8\n");
    }
    prefill_io_.ar_len = config_.ar_len;
    decode_io_.ar_len = 1;
    prefill_io_.dump_block_outputs = dump_block_outputs_;
    decode_io_.dump_block_outputs = dump_block_outputs_;
    prefill_io_.dump_layer0_outputs = dump_layer0_outputs_;
    decode_io_.dump_layer0_outputs = dump_layer0_outputs_;
    initQwen2VLIO(prefill_io_, config_, *kv_manager_u8_, key_cache_manager_u16_.get(), key_cache_uint16_,
                  qnn_cfg_.hidden_size, qnn_cfg_.intermediate_size);
    initQwen2VLIO(decode_io_, config_, *kv_manager_u8_, key_cache_manager_u16_.get(), key_cache_uint16_,
                  qnn_cfg_.hidden_size, qnn_cfg_.intermediate_size);
    if (visual_qnn_) { initializeVisualQnnModules(visual_bundle_layout_, visual_bucket_grids_, visual_modules_); }
    return true;
  }

  void resetStateForRequest() {
    current_pos_ = 0;
    prev_position_ids_ = Tensor::nil();
    prefill_tokens_ = 0;
    generated_tokens_ = 0;
    prompt_token_ids_.clear();
    vision_token_start_ = -1;
    vision_token_count_ = 0;
  }

  PromptFeatures preparePrompt(const mllm::models::ARGenerationOutputPast& inputs,
                               mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel* visual) {
    PromptFeatures features;
    features.sequence = inputs.at("sequence");
    auto img = inputs.at("img");
    auto grid_thw = inputs.at("grid_thw");
    if (dump_block_outputs_ || dump_layer0_outputs_ || dump_visual_tokens_) {
      prompt_token_ids_ = tensorToTokenIds(features.sequence);
      dumpPromptSummary();
    }

    fmt::print("ViT Processing: ...\n");
    mllm::print("Image shape is:", img.shape());
    auto visual_img = img;
    auto visual_grid_thw = grid_thw;
    auto visual_attention_mask = makeAllValidVisualAttentionMask(img.shape()[0]);
    std::string visual_graph_suffix;
    if (visual_qnn_ && !visual_bucket_grids_.empty()) {
      const auto* original_grid = grid_thw.ptr<int32_t>();
      const auto bucket = selectVisualBucket(grid_thw, visual_bucket_grids_);
      visual_img = padVisualPatchesToBucket(img, grid_thw, bucket, visual_cfg_);
      visual_grid_thw = makeGridThwTensor(original_grid[0], bucket.grid_h, bucket.grid_w);
      visual_attention_mask = makeVisualAttentionMaskForBucket(grid_thw, bucket, visual_cfg_);
      visual_graph_suffix = visualGraphSuffixForPatchTokens(bucket.patchTokens());
      fmt::print("[Qwen2VL AOT] visual bucket: original_grid={}x{} patches={} -> bucket_grid={}x{} patches={} graph_suffix={}\n",
                 original_grid[1],
                 original_grid[2],
                 img.shape()[0],
                 bucket.grid_h,
                 bucket.grid_w,
                 visual_img.shape()[0],
                 visual_graph_suffix);
    }
    auto inv_freq = mllm::models::qwen2vl::makeVisualRoPEInvFreq(visual_cfg_.visual_embed_dim / visual_cfg_.visual_num_heads,
                                                                 10000.0);
    auto visual_pos_ids = mllm::models::qwen2vl::makeVisualRotaryPosEmbIds(visual_grid_thw, visual_cfg_.visual_spatial_merge_size);
    auto rotary_pos_emb_full = mllm::models::qwen2vl::makeVisualRotaryPosEmbFull(inv_freq, visual_img.shape()[0]);
    auto pos_emb = mllm::models::qwen2vl::makeVisualRotaryPosEmb(rotary_pos_emb_full, visual_pos_ids, visual_grid_thw);
    auto [visual_embedding_sin, visual_embedding_cos] = mllm::models::qwen2vl::makeVisualRotarySinCos(pos_emb);

    vit_start_ = std::chrono::high_resolution_clock::now();
    if (visual_qnn_) {
      const int32_t half_dim = visual_cfg_.visual_embed_dim / visual_cfg_.visual_num_heads / 2;
      auto visual_sin_4d = visual_embedding_sin.view({1, -1, 1, half_dim}, false);
      auto visual_cos_4d = visual_embedding_cos.view({1, -1, 1, half_dim}, false);
      auto bucket_embeddings =
          runVisualQnnBundle(visual_img,
                             visual_sin_4d,
                             visual_cos_4d,
                             visual_attention_mask,
                             visual_cfg_,
                             visual_bundle_layout_,
                             visual_graph_suffix,
                             &visual_modules_);
      features.visual_embeddings = cropVisualEmbeddingsFromBucket(bucket_embeddings, grid_thw, visual_grid_thw, visual_cfg_);
    } else {
      MLLM_RT_ASSERT(visual != nullptr);
      features.visual_embeddings = (*visual)(img, visual_embedding_sin, visual_embedding_cos)[0];
    }
    vit_end_ = std::chrono::high_resolution_clock::now();
    fmt::print("ViT Processing: done, backend={}, time cost: {:.3f} seconds\n",
               visual_qnn_ ? "QNN AOT" : "CPU/KAI",
               std::chrono::duration<double>(vit_end_ - vit_start_).count());
    if (shouldDumpQwen2VLAOTStats()) {
      const auto* data = features.visual_embeddings.ptr<float>();
      const auto numel = features.visual_embeddings.numel();
      auto [min_it, max_it] = std::minmax_element(data, data + numel);
      const float qmin = (0 - input_embedding_qp_.zero_point) * input_embedding_qp_.scale;
      const float qmax = (65535 - input_embedding_qp_.zero_point) * input_embedding_qp_.scale;
      int64_t clipped = 0;
      for (int64_t i = 0; i < numel; ++i) {
        if (data[i] < qmin || data[i] > qmax) { ++clipped; }
      }
      fmt::print("[Qwen2VL AOT] visual_embeddings shape=[{}, {}], range=[{:.6f}, {:.6f}], embedding_qrange=[{:.6f}, "
                 "{:.6f}], clipped={}/{} ({:.2f}%)\n",
                 features.visual_embeddings.shape()[0], features.visual_embeddings.shape()[1], *min_it, *max_it, qmin, qmax,
                 clipped, numel, numel > 0 ? 100.0 * static_cast<double>(clipped) / static_cast<double>(numel) : 0.0);
    }

    features.vision_token_start = findVisionTokenStart(features.sequence, qnn_cfg_.vision_token_id);
    MLLM_RT_ASSERT(features.vision_token_start >= 0);
    vision_token_start_ = features.vision_token_start;
    vision_token_count_ = features.visual_embeddings.shape()[0];

    features.position_ids = makePositionIdsPrefill(features.sequence, grid_thw, qnn_cfg_);
    auto inv = mllm::models::qwen2vl::makeMultimodalRoPEInvFreq(
        qnn_cfg_.hidden_size / qnn_cfg_.num_attention_heads, qnn_cfg_.rope_theta);
    auto [sin, cos] = mllm::models::qwen2vl::makeMultimodalPositionEmbedding(
        features.position_ids, inv, qnn_cfg_.max_position_embeddings, qnn_cfg_.hidden_size / qnn_cfg_.num_attention_heads,
        qnn_cfg_.mrope_section);
    features.llm_embedding_sin = sin;
    features.llm_embedding_cos = cos;
    return features;
  }

  int64_t prefill(const PromptFeatures& features) {
    const int64_t num_tokens = features.sequence.shape()[1];
    const auto dump_tokens = selectDumpTokens(num_tokens);
    if ((dump_block_outputs_ || dump_layer0_outputs_ || dump_visual_tokens_ || dump_logits_topk_ > 0) && !dump_path_.empty()) {
      writeDumpHeader(num_tokens, dump_tokens);
    }
    int64_t processed_tokens = 0;
    int64_t current_pos = 0;

    rearrangeCache(config_.ar_len);

    std::vector<int32_t> attention_map(config_.ar_len);
    std::iota(attention_map.begin(), attention_map.end(), -1);
    kv_manager_u8_->initAttentionMask(prefill_io_.inputs[3].ptr<uint16_t>(), attention_map, config_.ar_len, 0,
                                      config_.sliding_window);
    prefill_io_.module->setOutputTensors(prefill_io_.outputs);

    int64_t next_token = 0;
    while (processed_tokens < num_tokens) {
      const int32_t chunk = static_cast<int32_t>(std::min<int64_t>(config_.ar_len, num_tokens - processed_tokens));
      fillEmbeddingChunk(features.sequence, embedding_weight_, features.visual_embeddings, features.vision_token_start,
                         processed_tokens, chunk, prefill_io_.inputs[0], embedding_weight_qp_, input_embedding_qp_,
                         qnn_cfg_.hidden_size);
      fillQuantizedFloatChunk(features.llm_embedding_sin, processed_tokens, chunk, prefill_io_.inputs[1], sin_qp_,
                              config_.head_dim);
      fillQuantizedFloatChunk(features.llm_embedding_cos, processed_tokens, chunk, prefill_io_.inputs[2], cos_qp_,
                              config_.head_dim);

      auto module_inputs = prefill_io_.inputs;
      prefill_io_.outputs = (*prefill_io_.module)(module_inputs);
      if (dump_block_outputs_ || dump_layer0_outputs_ || dump_visual_tokens_) {
        dumpPrefillDebugOutputs(prefill_io_.inputs, prefill_io_.outputs, chunk, processed_tokens, dump_tokens);
      }

      updateCache(config_.ar_len, current_pos, chunk);
      kv_manager_u8_->updateAttentionMask(prefill_io_.inputs[3].ptr<uint16_t>(), config_.ar_len, current_pos, chunk,
                                          config_.sliding_window);

      const int32_t logits_idx = static_cast<int32_t>((processed_tokens + chunk - 1) % config_.ar_len);
      next_token = sampleGreedyFromLogitsAt(prefill_io_.outputs[0], logits_idx);
      if (dump_logits_topk_ > 0 && processed_tokens + chunk == num_tokens) {
        dumpLogitsTopK(prefill_io_.outputs[0], logits_idx, static_cast<int32_t>(processed_tokens + chunk - 1), "prefill");
      }

      processed_tokens += chunk;
      current_pos += chunk;
    }

    current_pos_ = current_pos;
    prev_position_ids_ = features.position_ids;
    return next_token;
  }

  int64_t decodeOne(int64_t token) {
    rearrangeCache(1);
    auto position_ids = makePositionIdsDecode(prev_position_ids_);
    prev_position_ids_ = position_ids;

    std::vector<int32_t> attention_map(1);
    std::iota(attention_map.begin(), attention_map.end(), -1);
    kv_manager_u8_->initAttentionMask(decode_io_.inputs[3].ptr<uint16_t>(), attention_map, 1, current_pos_,
                                      config_.sliding_window);

    auto token_tensor = Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
    token_tensor.ptr<int64_t>()[0] = token;
    auto visual_nil = Tensor::nil();
    fillEmbeddingChunk(token_tensor, embedding_weight_, visual_nil, 0, 0, 1, decode_io_.inputs[0], embedding_weight_qp_,
                       input_embedding_qp_, qnn_cfg_.hidden_size);

    auto inv = mllm::models::qwen2vl::makeMultimodalRoPEInvFreq(
        qnn_cfg_.hidden_size / qnn_cfg_.num_attention_heads, qnn_cfg_.rope_theta);
    auto [sin, cos] = mllm::models::qwen2vl::makeMultimodalPositionEmbedding(
        position_ids, inv, qnn_cfg_.max_position_embeddings, qnn_cfg_.hidden_size / qnn_cfg_.num_attention_heads,
        qnn_cfg_.mrope_section);
    fillQuantizedFloatChunk(sin, 0, 1, decode_io_.inputs[1], sin_qp_, config_.head_dim);
    fillQuantizedFloatChunk(cos, 0, 1, decode_io_.inputs[2], cos_qp_, config_.head_dim);

    decode_io_.module->setOutputTensors(decode_io_.outputs);
    auto module_inputs = decode_io_.inputs;
    decode_io_.outputs = (*decode_io_.module)(module_inputs);

    updateCache(1, current_pos_, 1);
    auto next_token = sampleGreedyFromLogitsAt(decode_io_.outputs[0], 0);
    ++current_pos_;
    return next_token;
  }

  void generate(const PromptFeatures& features,
                Qwen2VLTokenizer& tokenizer,
                int32_t gen_len,
                const std::function<void(const std::string&)>& token_callback) {
    prefill_start_ = std::chrono::high_resolution_clock::now();
    int64_t next_token = prefill(features);
    prefill_end_ = std::chrono::high_resolution_clock::now();

    generated_tokens_ = 1;
    emitToken(next_token, tokenizer, token_callback);

    decode_start_ = std::chrono::high_resolution_clock::now();
    for (int32_t i = 1; i < gen_len; ++i) {
      if (current_pos_ >= config_.context_len) { break; }
      if (eos_ids_.count(next_token)) { break; }
      next_token = decodeOne(next_token);
      ++generated_tokens_;
      emitToken(next_token, tokenizer, token_callback);
      if (eos_ids_.count(next_token)) { break; }
    }
    decode_end_ = std::chrono::high_resolution_clock::now();
    prefill_tokens_ = features.sequence.shape()[1];
  }

  void perfSummary() const {
    const auto llm_prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end_ - prefill_start_).count();
    const auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end_ - decode_start_).count();
    const auto vit_us = std::chrono::duration_cast<std::chrono::microseconds>(vit_end_ - vit_start_).count();
    const auto prefill_total_us = vit_us + llm_prefill_us;
    const auto total_us = prefill_total_us + decode_us;

    fmt::print(fg(fmt::color::cyan), "\n{:=^50}\n", " QNN AOT Performance ");
    fmt::print("Total time          : {:10.3f} s\n", total_us / 1000000.0);
    fmt::print("Prefill total time  : {:10.3f} s ({:6.2f} tokens/s)\n", prefill_total_us / 1000000.0,
               prefill_total_us > 0 ? prefill_tokens_ / (prefill_total_us / 1000000.0) : 0.0);
    fmt::print("LLM prefill time    : {:10.3f} s ({:6.2f} tokens/s)\n", llm_prefill_us / 1000000.0,
               llm_prefill_us > 0 ? prefill_tokens_ / (llm_prefill_us / 1000000.0) : 0.0);
    fmt::print("Decode time         : {:10.3f} s ({:6.2f} tokens/s)\n", decode_us / 1000000.0,
               decode_us > 0 ? generated_tokens_ / (decode_us / 1000000.0) : 0.0);
    fmt::print("TTFT                : {:10.3f} s\n", prefill_total_us / 1000000.0);
    fmt::print("Prefill tokens      : {:10d}\n", prefill_tokens_);
    fmt::print("Decode steps        : {:10d}\n", generated_tokens_);
    fmt::print("Avg decode time     : {:10.3f} s/token\n",
               generated_tokens_ > 0 ? (decode_us / 1000000.0) / generated_tokens_ : 0.0);
    fmt::print(fg(fmt::color::cyan), "{:=^50}\n", "");
    fmt::print(fg(fmt::color::magenta), "\n{:=^50}\n", " Custom Events ");
    fmt::print("ViT                 : {:10.2f} μs\n", static_cast<double>(vit_us));
    fmt::print(fg(fmt::color::magenta), "{:=^50}\n", "");
  }

 private:
  void rearrangeCache(int32_t ar_len) {
    if (key_cache_uint16_) {
      MLLM_RT_ASSERT(key_cache_manager_u16_ != nullptr);
      key_cache_manager_u16_->rearrangeCache(ar_len);
    }
    kv_manager_u8_->rearrangeCache(ar_len);
  }

  void updateCache(int32_t ar_len, int32_t current_pos, int32_t chunk) {
    if (key_cache_uint16_) {
      MLLM_RT_ASSERT(key_cache_manager_u16_ != nullptr);
      key_cache_manager_u16_->updateCache(ar_len, current_pos, chunk, {});
    }
    kv_manager_u8_->updateCache(ar_len, current_pos, chunk, {});
  }

  void emitToken(int64_t token,
                 Qwen2VLTokenizer& tokenizer,
                 const std::function<void(const std::string&)>& token_callback) {
    if (!token_callback) { return; }
    auto wstr = tokenizer.detokenize(token);
    token_callback(mllm::preprocessor::wideString2Utf8String(wstr));
  }

  const Qwen2VLConfig& qnn_cfg_;
  const Qwen2VLConfig& visual_cfg_;
  mllm::ParameterFile::ptr_t qnn_params_;
  Tensor embedding_weight_;
  QuantParams embedding_weight_qp_;
  QuantParams input_embedding_qp_;
  QuantParams sin_qp_;
  QuantParams cos_qp_;
  QuantParams logits_qp_;
  std::vector<QuantParams> block_out_qps_;
  std::vector<std::pair<std::string, QuantParams>> layer0_output_qps_;
  QnnAOTConfig config_;
  std::unique_ptr<KVCacheManager<uint8_t>> kv_manager_u8_;
  std::unique_ptr<KVCacheManager<uint16_t>> key_cache_manager_u16_;
  RuntimeIO prefill_io_;
  RuntimeIO decode_io_;
  QnnAOTModuleCache visual_modules_;
  Tensor prev_position_ids_;
  int64_t current_pos_ = 0;
  int32_t prefill_tokens_ = 0;
  int32_t generated_tokens_ = 0;
  bool dump_block_outputs_ = false;
  bool dump_layer0_outputs_ = false;
  bool dump_visual_tokens_ = false;
  bool key_cache_uint16_ = false;
  bool visual_qnn_ = false;
  std::string visual_bundle_layout_ = "6x8";
  std::vector<VisualBucketGrid> visual_bucket_grids_;
  int32_t dump_logits_topk_ = 0;
  std::vector<int32_t> dump_token_indices_;
  std::string dump_path_;
  std::vector<int64_t> prompt_token_ids_;
  int32_t vision_token_start_ = -1;
  int32_t vision_token_count_ = 0;
  std::unordered_set<int64_t> eos_ids_{151643, 151645};
  std::chrono::high_resolution_clock::time_point vit_start_;
  std::chrono::high_resolution_clock::time_point vit_end_;
  std::chrono::high_resolution_clock::time_point prefill_start_;
  std::chrono::high_resolution_clock::time_point prefill_end_;
  std::chrono::high_resolution_clock::time_point decode_start_;
  std::chrono::high_resolution_clock::time_point decode_end_;

  static float dequantizeUInt16(uint16_t value, const QuantParams& qp) {
    return (static_cast<int32_t>(value) - qp.zero_point) * qp.scale;
  }

  static float dequantizeUInt8(uint8_t value, const QuantParams& qp) {
    return (static_cast<int32_t>(value) - qp.zero_point) * qp.scale;
  }

  static std::vector<int64_t> tensorToTokenIds(Tensor sequence) {
    const auto seq_len = sequence.shape()[1];
    const auto* ids = sequence.ptr<int64_t>();
    return std::vector<int64_t>(ids, ids + seq_len);
  }

  void initLayer0QuantParams() {
    const auto p = [&](const std::string& label, const std::string& base) {
      layer0_output_qps_.push_back(
          {label, readQuantParams(qnn_params_, base + ".fake_quant.scale", base + ".fake_quant.zero_point")});
    };
    const auto p_heads = [&](const std::string& label, const std::string& base) {
      for (int32_t h = 0; h < qnn_cfg_.num_attention_heads; ++h) { p(label + "_h" + std::to_string(h), base); }
    };
    p("layer0_input_layernorm_out", "model.layers.0.self_attn.q_proj_input_qdq");
    p("layer0_q_proj_out_flat", "model.layers.0.self_attn.q_proj_output_qdq");
    p("layer0_k_proj_out_flat", "model.layers.0.self_attn.k_proj_output_qdq");
    p("layer0_v_proj_out_flat", "model.layers.0.self_attn.v_cast_to_int16_qdq");
    p("layer0_v_cache_out_flat", "model.layers.0.self_attn.v_cast_to_int8_qdq");
    p("layer0_q_rope_out_flat", "model.layers.0.self_attn.q_rope_add_0_output_qdq");
    p("layer0_k_rope_out_flat", "model.layers.0.self_attn.k_rope_add_0_output_qdq");
    p_heads("layer0_qk_matmul_out_flat", "model.layers.0.self_attn.qk_matmul_output_qdq");
    p_heads("layer0_qk_scaled_out_flat", "model.layers.0.self_attn.mul_0_output_qdq");
    p_heads("layer0_qk_masked_out_flat", "model.layers.0.self_attn.where_attn_qdq");
    p_heads("layer0_softmax_out_flat", "model.layers.0.self_attn.softmax_output_qdq");
    p("layer0_attn_value_out_flat", "model.layers.0.self_attn.attn_value_matmul_output_qdq");
    p("layer0_o_proj_out", "model.layers.0.add_0_lhs_input_qdq");
    p("layer0_post_attention_layernorm_out", "model.layers.0.mlp.up_proj_input_qdq");
    p("layer0_mlp_up_proj_out", "model.layers.0.mlp.up_proj_output_qdq");
    p("layer0_mlp_gate_proj_out", "model.layers.0.mlp.gate_proj_output_qdq");
    p("layer0_mlp_gate_act_out", "model.layers.0.mlp.act_output_qdq");
    p("layer0_mlp_down_proj_input", "model.layers.0.mlp.down_proj_input_qdq");
    p("layer0_mlp_down_proj_out", "model.layers.0.add_1_lhs_input_qdq");
    layer0_output_qps_.push_back(
        {"layer0_block_out", readQuantParams(qnn_params_, "model.layers.1.input_layernorm_input_qdq.fake_quant.scale",
                                             "model.layers.1.input_layernorm_input_qdq.fake_quant.zero_point")});
  }

  void dumpPromptSummary() const {
    const auto seq_len = static_cast<int32_t>(prompt_token_ids_.size());
    fmt::print("[Qwen2VL AOT Dump] seq_len={} token_ids_first=", seq_len);
    const int32_t first = std::min<int32_t>(seq_len, 16);
    for (int32_t i = 0; i < first; ++i) { fmt::print("{}{}", i == 0 ? "" : ",", prompt_token_ids_[i]); }
    fmt::print(" token_ids_last=");
    const int32_t start = std::max<int32_t>(0, seq_len - 16);
    for (int32_t i = start; i < seq_len; ++i) { fmt::print("{}{}", i == start ? "" : ",", prompt_token_ids_[i]); }
    fmt::print("\n");
  }

  void dumpTokenIdLine(std::ofstream& out, const char* label, int32_t begin, int32_t end) const {
    out << label << "=";
    for (int32_t i = begin; i < end; ++i) { out << (i == begin ? "" : ",") << prompt_token_ids_[i]; }
    out << "\n";
  }

  void dumpVectorLine(std::ofstream& out, const std::string& label, int32_t layer, int32_t token, const std::vector<float>& vec) {
    double dot = 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    for (float v : vec) {
      sum += v;
      sum_sq += static_cast<double>(v) * static_cast<double>(v);
      max_abs = std::max(max_abs, std::abs(static_cast<double>(v)));
    }
    const double mean = vec.empty() ? 0.0 : sum / static_cast<double>(vec.size());
    for (float v : vec) {
      const double d = static_cast<double>(v) - mean;
      dot += d * d;
    }
    const double stddev = vec.empty() ? 0.0 : std::sqrt(dot / static_cast<double>(vec.size()));
    const double rms = vec.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(vec.size()));
    out << "[LAYER_DUMP][NPU][label=" << label << "][layer=" << layer << "][token=" << token << "] mean=" << mean
        << " std=" << stddev << " rms=" << rms << " maxabs=" << max_abs;
    for (int32_t i = 0; i < 4; ++i) { out << " d" << i << "=" << (i < static_cast<int32_t>(vec.size()) ? vec[i] : 0.0f); }
    out << "\n";

    out << "[LAYER_VEC][NPU][label=" << label << "][layer=" << layer << "][token=" << token << "][dims="
        << vec.size() << "] v=";
    for (size_t i = 0; i < vec.size(); ++i) { out << (i == 0 ? "" : ",") << vec[i]; }
    out << "\n";
  }

  std::vector<int32_t> selectDumpTokens(int64_t num_tokens) const {
    std::vector<int32_t> tokens;
    const auto normalize = [&](int32_t token) -> int32_t {
      if (token < 0) { token = static_cast<int32_t>(num_tokens) + token; }
      return token;
    };

    for (auto token : dump_token_indices_) {
      token = normalize(token);
      if (token >= 0 && token < num_tokens) { tokens.push_back(token); }
    }
    if (dump_visual_tokens_ && vision_token_start_ >= 0 && vision_token_count_ > 0) {
      tokens.push_back(vision_token_start_);
      tokens.push_back(vision_token_start_ + vision_token_count_ / 2);
      tokens.push_back(vision_token_start_ + vision_token_count_ - 1);
      tokens.push_back(static_cast<int32_t>(num_tokens - 1));
    }
    if (tokens.empty()) { tokens.push_back(static_cast<int32_t>(num_tokens - 1)); }

    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
  }

  void writeDumpHeader(int64_t num_tokens, const std::vector<int32_t>& dump_tokens) const {
    std::ofstream out(dump_path_, std::ios::out | std::ios::trunc);
    if (!out) {
      fmt::print("[Qwen2VL AOT Dump] failed to open dump path: {}\n", dump_path_);
      return;
    }
    out << "# Qwen2-VL QNN AOT debug dump\n";
    out << "seq_len=" << num_tokens << "\n";
    out << "global_token=" << (num_tokens - 1) << "\n";
    out << "vision_token_start=" << vision_token_start_ << "\n";
    out << "vision_token_count=" << vision_token_count_ << "\n";
    out << "dump_tokens=";
    for (size_t i = 0; i < dump_tokens.size(); ++i) { out << (i == 0 ? "" : ",") << dump_tokens[i]; }
    out << "\n";
    if (!prompt_token_ids_.empty()) {
      const int32_t first_end = std::min<int32_t>(prompt_token_ids_.size(), 16);
      const int32_t last_begin = std::max<int32_t>(0, static_cast<int32_t>(prompt_token_ids_.size()) - 16);
      dumpTokenIdLine(out, "token_ids_first", 0, first_end);
      dumpTokenIdLine(out, "token_ids_last", last_begin, static_cast<int32_t>(prompt_token_ids_.size()));
    }
  }

  std::vector<float> dequantizeRow(Tensor& tensor, int32_t token_in_chunk, const QuantParams& qp) const {
    auto tensor_cpu = tensor.to(mllm::kCPU);
    const auto& shape = tensor_cpu.shape();
    const auto dtype = tensor_cpu.dtype();
    const bool is_uint8 = dtype == mllm::kUInt8 || dtype == mllm::kUInt8PerTensorSym || dtype == mllm::kUInt8PerTensorAsy;
    if (shape.size() == 4) {
      MLLM_RT_ASSERT_EQ(shape[0], 1);
      MLLM_RT_ASSERT(token_in_chunk >= 0 && token_in_chunk < shape[2]);
      const int32_t heads = shape[1];
      const int32_t seq = shape[2];
      const int32_t context = shape[3];
      std::vector<float> vec(heads * context);
      if (is_uint8) {
        const auto* data = tensor_cpu.ptr<uint8_t>();
        for (int32_t h = 0; h < heads; ++h) {
          const uint8_t* row = data + (h * seq + token_in_chunk) * context;
          for (int32_t d = 0; d < context; ++d) { vec[h * context + d] = dequantizeUInt8(row[d], qp); }
        }
      } else {
        const auto* data = tensor_cpu.ptr<uint16_t>();
        for (int32_t h = 0; h < heads; ++h) {
          const uint16_t* row = data + (h * seq + token_in_chunk) * context;
          for (int32_t d = 0; d < context; ++d) { vec[h * context + d] = dequantizeUInt16(row[d], qp); }
        }
      }
      return vec;
    }
    const int32_t dims = shape.back();
    MLLM_RT_ASSERT(token_in_chunk >= 0);
    MLLM_RT_ASSERT(tensor_cpu.numel() >= static_cast<int64_t>(token_in_chunk + 1) * dims);
    std::vector<float> vec(dims);
    if (is_uint8) {
      const auto* data = tensor_cpu.ptr<uint8_t>();
      const uint8_t* row = data + token_in_chunk * dims;
      for (int32_t d = 0; d < dims; ++d) { vec[d] = dequantizeUInt8(row[d], qp); }
    } else {
      const auto* data = tensor_cpu.ptr<uint16_t>();
      const uint16_t* row = data + token_in_chunk * dims;
      for (int32_t d = 0; d < dims; ++d) { vec[d] = dequantizeUInt16(row[d], qp); }
    }
    return vec;
  }

  static std::string attentionHeadGroupBase(const std::string& label) {
    static const std::vector<std::string> bases = {
        "layer0_qk_matmul_out_flat",
        "layer0_qk_scaled_out_flat",
        "layer0_qk_masked_out_flat",
        "layer0_softmax_out_flat",
    };
    for (const auto& base : bases) {
      if (label == base + "_h0") { return base; }
    }
    return "";
  }

  void dumpInputEmbedding(std::ofstream& out, Tensor& input_embeddings, int32_t token_in_chunk, int32_t global_token) {
    auto vec = dequantizeRow(input_embeddings, token_in_chunk, input_embedding_qp_);
    dumpVectorLine(out, "input_embeddings", 0, global_token, vec);
  }

  void dumpLogitsTopK(Tensor& logits, int32_t token_in_chunk, int32_t global_token, const std::string& stage) {
    if (dump_path_.empty() || dump_logits_topk_ <= 0) { return; }
    std::ofstream out(dump_path_, std::ios::out | std::ios::app);
    if (!out) {
      fmt::print("[Qwen2VL AOT Dump] failed to open dump path: {}\n", dump_path_);
      return;
    }
    auto rows = topKFromLogitsAt(logits, token_in_chunk, dump_logits_topk_, logits_qp_);
    for (int32_t rank = 0; rank < static_cast<int32_t>(rows.size()); ++rank) {
      const auto& row = rows[rank];
      out << "[LOGITS_TOPK][NPU][stage=" << stage << "][token=" << global_token << "][rank=" << rank
          << "] token_id=" << row.token_id << " raw=" << row.raw << " logit=" << row.value << "\n";
    }
    fmt::print("[Qwen2VL AOT Dump] wrote logits top-k to {}\n", dump_path_);
  }

  void dumpPrefillDebugOutputs(std::vector<Tensor>& inputs,
                               std::vector<Tensor>& outputs,
                               int32_t chunk,
                               int64_t processed_tokens,
                               const std::vector<int32_t>& dump_tokens) {
    if (dump_path_.empty()) { return; }
    std::ofstream out(dump_path_, std::ios::out | std::ios::app);
    if (!out) {
      fmt::print("[Qwen2VL AOT Dump] failed to open dump path: {}\n", dump_path_);
      return;
    }
    std::vector<int32_t> tokens_in_chunk;
    for (auto token : dump_tokens) {
      if (token >= processed_tokens && token < processed_tokens + chunk) { tokens_in_chunk.push_back(token); }
    }
    if (tokens_in_chunk.empty()) { return; }
    out << "chunk_start=" << processed_tokens << "\n";
    out << "chunk_len=" << chunk << "\n";
    for (auto global_token : tokens_in_chunk) {
      const int32_t token_in_chunk = global_token - static_cast<int32_t>(processed_tokens);
      dumpInputEmbedding(out, inputs[0], token_in_chunk, global_token);
    }

    const int32_t block_offset = 1 + 2 * config_.num_layers;
    if (dump_block_outputs_) {
      if (outputs.size() < static_cast<size_t>(block_offset + config_.num_layers)) {
        fmt::print("[Qwen2VL AOT Dump] block outputs requested but output count={} is too small\n", outputs.size());
        return;
      }
      for (auto global_token : tokens_in_chunk) {
        const int32_t token_in_chunk = global_token - static_cast<int32_t>(processed_tokens);
        for (int32_t l = 0; l < config_.num_layers; ++l) {
          auto vec = dequantizeRow(outputs[block_offset + l], token_in_chunk, block_out_qps_[l]);
          dumpVectorLine(out, "block_out", l, global_token, vec);
        }
      }
    }
    if (dump_layer0_outputs_) {
      const int32_t layer0_offset = block_offset + (dump_block_outputs_ ? config_.num_layers : 0);
      if (outputs.size() < static_cast<size_t>(layer0_offset + layer0_output_qps_.size())) {
        fmt::print("[Qwen2VL AOT Dump] layer0 outputs requested but output count={} is too small\n", outputs.size());
        return;
      }
      for (auto global_token : tokens_in_chunk) {
        const int32_t token_in_chunk = global_token - static_cast<int32_t>(processed_tokens);
        for (size_t i = 0; i < layer0_output_qps_.size();) {
          const auto& [label, qp] = layer0_output_qps_[i];
          const auto head_group_base = attentionHeadGroupBase(label);
          if (!head_group_base.empty()) {
            std::vector<float> vec;
            for (int32_t h = 0; h < qnn_cfg_.num_attention_heads; ++h) {
              MLLM_RT_ASSERT(i + h < layer0_output_qps_.size());
              auto head_vec =
                  dequantizeRow(outputs[layer0_offset + static_cast<int32_t>(i) + h], token_in_chunk,
                                layer0_output_qps_[i + h].second);
              vec.insert(vec.end(), head_vec.begin(), head_vec.end());
            }
            dumpVectorLine(out, head_group_base, 0, global_token, vec);
            i += qnn_cfg_.num_attention_heads;
            continue;
          }
          auto vec = dequantizeRow(outputs[layer0_offset + static_cast<int32_t>(i)], token_in_chunk, qp);
          dumpVectorLine(out, label, 0, global_token, vec);
          if (label == "layer0_o_proj_out") { dumpVectorLine(out, "layer0_attn_out", 0, global_token, vec); }
          if (label == "layer0_mlp_down_proj_out") { dumpVectorLine(out, "layer0_mlp_out", 0, global_token, vec); }
          ++i;
        }
      }
    }
    fmt::print("[Qwen2VL AOT Dump] wrote debug vectors to {}\n", dump_path_);
  }
};

std::vector<int32_t> parseDumpTokenIndices(const std::string& text) {
  std::vector<int32_t> tokens;
  if (text.empty()) { return tokens; }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) { continue; }
    tokens.push_back(std::stoi(item));
  }
  return tokens;
}

bool isInteractiveExitCommand(const std::string& text) {
  return text == "/exit" || text == "/quit" || text == "exit" || text == "quit";
}

mllm::models::ARGenerationOutputPast makeRequestInputs(Qwen2VLTokenizer& tokenizer,
                                                       const std::string& image_path,
                                                       const std::string& prompt,
                                                       bool visual_qnn,
                                                       const std::vector<VisualBucketGrid>& visual_bucket_grids) {
  const auto resize_override =
      visual_qnn ? chooseVisualResizeOverrideForOversizedImage(image_path, visual_bucket_grids) : VisualResizeOverride{};
  if (resize_override.enabled) {
    fmt::print("[Qwen2VL AOT] visual oversized fallback: native_grid={}x{} patches={} -> resize_grid={}x{} patches={} "
               "within bucket_grid={}x{} patches={} graph_suffix={}\n",
               resize_override.native_grid_h,
               resize_override.native_grid_w,
               resize_override.native_grid_h * resize_override.native_grid_w,
               resize_override.resize_grid_h,
               resize_override.resize_grid_w,
               resize_override.resize_grid_h * resize_override.resize_grid_w,
               resize_override.bucket.grid_h,
               resize_override.bucket.grid_w,
               resize_override.bucket.patchTokens(),
               visualGraphSuffixForPatchTokens(resize_override.bucket.patchTokens()));
  }
  return resize_override.enabled
             ? tokenizer.convertMessage({.prompt = prompt, .img_file_path = image_path},
                                        resize_override.resize_grid_h,
                                        resize_override.resize_grid_w)
             : tokenizer.convertMessage({.prompt = prompt, .img_file_path = image_path});
}

void runOneRequest(Qwen2VLAOTRunner& runner,
                   Qwen2VLTokenizer& tokenizer,
                   mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel* visual,
                   const std::string& image_path,
                   const std::string& prompt,
                   bool visual_qnn,
                   const std::vector<VisualBucketGrid>& visual_bucket_grids,
                   int32_t gen_len) {
  auto inputs = makeRequestInputs(tokenizer, image_path, prompt, visual_qnn, visual_bucket_grids);
  runner.resetStateForRequest();
  auto features = runner.preparePrompt(inputs, visual);
  fmt::print("\nResponse: ");
  runner.generate(features, tokenizer, gen_len, [](const std::string& token) { std::cout << token << std::flush; });
  fmt::print("\n");
  runner.perfSummary();
}

void runInteractiveLoop(Qwen2VLAOTRunner& runner,
                        Qwen2VLTokenizer& tokenizer,
                        mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel* visual,
                        bool visual_qnn,
                        const std::vector<VisualBucketGrid>& visual_bucket_grids,
                        const std::string& default_prompt,
                        int32_t gen_len) {
  fmt::print("\n[Qwen2VL AOT] interactive mode is ready. Enter an image path per request; /exit quits.\n");
  fmt::print("[Qwen2VL AOT] default prompt: {}\n", default_prompt);

  while (true) {
    std::string image_path;
    std::string prompt;

    fmt::print("\nImage path> ");
    std::cout.flush();
    if (!std::getline(std::cin, image_path)) { break; }
    image_path = trimInteractiveLine(image_path);
    if (image_path.empty() || isInteractiveExitCommand(image_path)) { break; }

    fmt::print("Prompt> ");
    std::cout.flush();
    if (!std::getline(std::cin, prompt)) { break; }
    prompt = trimInteractiveLine(prompt);
    if (isInteractiveExitCommand(prompt)) { break; }
    if (prompt.empty()) { prompt = default_prompt; }

    fmt::print("[Qwen2VL AOT] running request: image={}, gen_len={}\n", image_path, gen_len);
    runOneRequest(runner, tokenizer, visual, image_path, prompt, visual_qnn, visual_bucket_grids, gen_len);
  }

  fmt::print("\n[Qwen2VL AOT] interactive mode stopped.\n");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& context_path = Argparse::add<std::string>("-m|--model").help("QNN AOT context .bin path").required(true);
  auto& qnn_params_path =
      Argparse::add<std::string>("--qnn_params").help("LPBQ .mllm file used for embedding and QDQ params").required(true);
  auto& visual_model_path =
      Argparse::add<std::string>("--visual_model").help("FP32/W4A32 .mllm file used for CPU ViT when --visual_qnn is not set.");
  auto& visual_model_version =
      Argparse::add<std::string>("--visual_model_version").help("visual model file version: v1/v2").def("v2");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path").required(true);
  auto& qnn_config_path = Argparse::add<std::string>("-c|--config").help("QNN LPBQ config path").required(true);
  auto& visual_config_path =
      Argparse::add<std::string>("--visual_config").help("visual model config path").required(true);
  auto& image_path = Argparse::add<std::string>("-i|--image").help("input image path for single-shot mode").def("");
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("prompt text").def("Describe this picture.");
  auto& interactive =
      Argparse::add<bool>("--interactive").help("Run repeated image/prompt requests in one process after model initialization.");
  auto& ar_len = Argparse::add<int>("--ar_len").help("prefill graph chunk length").def(32);
  auto& context_len = Argparse::add<int>("--context_len").help("QNN context length").def(1024);
  auto& gen_len = Argparse::add<int>("--gen_len").help("max generated tokens").def(96);
  auto& input_embedding_scale =
      Argparse::add<float>("--input_embedding_scale")
          .help("input_embeddings UInt16 scale. Defaults to the Qwen2-VL visual-safe wide QP; set both input embedding "
                "QP arguments to -1 to use model.embed_tokens QP.")
          .def(kDefaultInputEmbeddingScale);
  auto& input_embedding_zero_point =
      Argparse::add<int>("--input_embedding_zero_point")
          .help("input_embeddings UInt16 zero point. Defaults to the Qwen2-VL visual-safe wide QP; set both input "
                "embedding QP arguments to -1 to use model.embed_tokens QP.")
          .def(kDefaultInputEmbeddingZeroPoint);
  auto& dump_block_outputs =
      Argparse::add<bool>("--dump_block_outputs").help("Dump per-layer block_out vectors from debug QNN context.");
  auto& dump_layer0_outputs =
      Argparse::add<bool>("--dump_layer0_outputs").help("Dump layer0 fine-grained vectors from debug QNN context.");
  auto& dump_visual_tokens =
      Argparse::add<bool>("--dump_visual_tokens").help("Dump first/middle/last visual token and final prefill token.");
  auto& dump_token_indices = Argparse::add<std::string>("--dump_token_indices")
                                 .help("Comma-separated global token indices to dump. Negative values count from end.")
                                 .def("");
  auto& dump_logits_topk =
      Argparse::add<int>("--dump_logits_topk").help("Dump top-k logits at the final prefill token.").def(0);
  auto& dump_path = Argparse::add<std::string>("--dump_path")
                        .help("Path on device for debug vector dump.")
                        .def("/data/local/tmp/qwen2vl_qnn_aot_debug_dump.log");
  auto& key_cache_dtype =
      Argparse::add<std::string>("--key_cache_dtype")
          .help("Key cache dtype for experimental contexts: uint8 or uint16. Value cache remains uint8.")
          .def("uint8");
  auto& visual_qnn =
      Argparse::add<bool>("--visual_qnn").help("Use visual graphs from the loaded combined QNN context instead of CPU/KAI visual.");
  auto& visual_bundle_layout =
      Argparse::add<std::string>("--visual_bundle_layout").help("visual QNN bundle layout: single, 6x8, early2 or tail4.").def("6x8");
  auto& visual_bucket_grids = Argparse::add<std::string>("--visual_bucket_grids")
                                  .help("Comma-separated visual QNN patch-grid buckets HxW. Example: 10x16,12x16,26x36.")
                                  .def("");

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  const bool override_input_embedding_qp = input_embedding_scale.get() > 0.0f || input_embedding_zero_point.get() >= 0;
  if (override_input_embedding_qp && (input_embedding_scale.get() <= 0.0f || input_embedding_zero_point.get() < 0)) {
    std::cerr << "input embedding override requires both --input_embedding_scale and --input_embedding_zero_point; "
                 "set both to -1 to use model.embed_tokens QP\n";
    return 1;
  }
  const bool key_cache_uint16 = key_cache_dtype.get() == "uint16";
  if (!key_cache_uint16 && key_cache_dtype.get() != "uint8") {
    std::cerr << "--key_cache_dtype must be uint8 or uint16\n";
    return 1;
  }
  if (!interactive.isSet() && !image_path.isSet()) {
    std::cerr << "-i/--image is required unless --interactive is set\n";
    return 1;
  }

  mllm::initQnnBackend(context_path.get());

  auto qnn_cfg = Qwen2VLConfig(qnn_config_path.get());
  auto visual_cfg = Qwen2VLConfig(visual_config_path.get());
  auto tokenizer = Qwen2VLTokenizer(tokenizer_path.get());

  auto qnn_params = mllm::load(qnn_params_path.get(), mllm::ModelFileVersion::kV2);
  std::unique_ptr<mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel> visual;
  if (!visual_qnn.isSet()) {
    if (!visual_model_path.isSet()) {
      std::cerr << "--visual_model is required when --visual_qnn is not set\n";
      return 1;
    }
    mllm::ModelFileVersion visual_file_version = mllm::ModelFileVersion::kV2;
    if (visual_model_version.get() == "v1") { visual_file_version = mllm::ModelFileVersion::kV1; }
    auto visual_params = mllm::load(visual_model_path.get(), visual_file_version);
    visual = std::make_unique<mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel>("visual", visual_cfg);
    visual->load(visual_params);
  }

  auto parsed_visual_bucket_grids = parseVisualBucketGrids(visual_bucket_grids.get());

  Qwen2VLAOTRunner runner(qnn_cfg, visual_cfg, qnn_params, ar_len.get(), context_len.get(), input_embedding_scale.get(),
                          input_embedding_zero_point.get(), dump_block_outputs.isSet(), dump_layer0_outputs.isSet(),
                          dump_visual_tokens.isSet(), key_cache_uint16, visual_qnn.isSet(), visual_bundle_layout.get(),
                          parsed_visual_bucket_grids,
                          dump_logits_topk.get(),
                          parseDumpTokenIndices(dump_token_indices.get()), dump_path.get());
  if (!runner.load()) {
    std::cerr << "Failed to load Qwen2-VL QNN AOT runner\n";
    return 1;
  }

  if (interactive.isSet()) {
    runInteractiveLoop(runner,
                       tokenizer,
                       visual.get(),
                       visual_qnn.isSet(),
                       parsed_visual_bucket_grids,
                       prompt.get(),
                       gen_len.get());
  } else {
    runOneRequest(runner,
                  tokenizer,
                  visual.get(),
                  image_path.get(),
                  prompt.get(),
                  visual_qnn.isSet(),
                  parsed_visual_bucket_grids,
                  gen_len.get());
  }
  mllm::memoryReport();
  return 0;
});
