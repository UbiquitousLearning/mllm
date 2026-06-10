// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/compile/ir/Node.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/compile/ir/builtin/Op.hpp>
#include <mllm/compile/ir/graph/Op.hpp>
#include <mllm/compile/ir/linalg/Op.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl_traceable.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>

using mllm::Argparse;

namespace {

std::string defaultQnnAOTEnvPath() {
  const char* qnn_root = std::getenv("QNN_SDK_ROOT");
  if (qnn_root != nullptr && qnn_root[0] != '\0') {
    return std::string(qnn_root) + "/lib/x86_64-linux-clang/";
  }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

class PatchEmbedLinear final : public mllm::nn::Module {
  int32_t patch_dim_ = 0;
  int32_t embed_dim_ = 0;

  mllm::nn::Linear proj_;

 public:
  PatchEmbedLinear() = default;

  PatchEmbedLinear(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg) : mllm::nn::Module(name) {
    patch_dim_ = cfg.visual_in_chans * cfg.visual_temporal_patch_size * cfg.visual_patch_size * cfg.visual_patch_size;
    embed_dim_ = cfg.visual_embed_dim;
    proj_ = reg<mllm::nn::Linear>("proj", patch_dim_, embed_dim_, false, mllm::aops::LinearImplTypes::kDefault);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    hidden_states = hidden_states.view({-1, patch_dim_}, true);
    hidden_states = proj_(hidden_states).view({-1, embed_dim_}, true);
    return {hidden_states};
  }
};

class Qwen2VisionTransformerPretrainedModelLinearPatch final : public mllm::nn::Module {
  PatchEmbedLinear patch_embed_;
  mllm::models::qwen2vl::PatchMerger patch_merger_;
  mllm::nn::ModuleList<mllm::models::qwen2vl::Qwen2VLVisionBlock> blocks_;

 public:
  Qwen2VisionTransformerPretrainedModelLinearPatch() = default;

  Qwen2VisionTransformerPretrainedModelLinearPatch(const std::string& name,
                                                   const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    patch_embed_ = reg<PatchEmbedLinear>("patch_embed", cfg);
    patch_merger_ = reg<mllm::models::qwen2vl::PatchMerger>("merger", cfg);
    blocks_ = reg<mllm::nn::ModuleList<mllm::models::qwen2vl::Qwen2VLVisionBlock>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];

    hidden_states = patch_embed_(hidden_states)[0];
    for (auto& b : blocks_.list()) { hidden_states = b(hidden_states, embedding_sin, embedding_cos)[0]; }
    hidden_states = patch_merger_(hidden_states)[0];

    return {hidden_states};
  }
};

class VisionMlpPrimitiveQuickGELU final : public mllm::nn::Module {
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;

  mllm::nn::Linear fc_1_;
  mllm::nn::Linear fc_2_;

 public:
  VisionMlpPrimitiveQuickGELU() = default;

  VisionMlpPrimitiveQuickGELU(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;
    fc_1_ = reg<mllm::nn::Linear>("fc1", dim_, hidden_dim_, true, cfg.linear_impl_type);
    fc_2_ = reg<mllm::nn::Linear>("fc2", hidden_dim_, dim_, true, cfg.linear_impl_type);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = fc_1_(inputs[0]);
    x = x * mllm::nn::functional::sigmoid(x * 1.702f);
    return {fc_2_(x)};
  }
};

class Qwen2VLVisionBlockAOTRewrite final : public mllm::nn::Module {
  mllm::nn::LayerNorm norm1_;
  mllm::nn::LayerNorm norm2_;

  mllm::models::qwen2vl::VisionAttention attn_;
  VisionMlpPrimitiveQuickGELU mlp_;

 public:
  Qwen2VLVisionBlockAOTRewrite() = default;

  Qwen2VLVisionBlockAOTRewrite(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    norm1_ = reg<mllm::nn::LayerNorm>("norm1", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    norm2_ = reg<mllm::nn::LayerNorm>("norm2", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    attn_ = reg<mllm::models::qwen2vl::VisionAttention>("attn", cfg);
    mlp_ = reg<VisionMlpPrimitiveQuickGELU>("mlp", cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];

    hidden_states = hidden_states + attn_(norm1_(hidden_states), visual_embedding_sin, visual_embedding_cos)[0];
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }
};

class Qwen2VisionTransformerPretrainedModelAOTRewrite final : public mllm::nn::Module {
  PatchEmbedLinear patch_embed_;
  mllm::models::qwen2vl::PatchMerger patch_merger_;
  mllm::nn::ModuleList<Qwen2VLVisionBlockAOTRewrite> blocks_;
  int32_t start_block_ = 0;
  int32_t active_blocks_ = -1;
  bool skip_merger_ = false;
  bool skip_patch_embed_ = false;

 public:
  Qwen2VisionTransformerPretrainedModelAOTRewrite() = default;

  Qwen2VisionTransformerPretrainedModelAOTRewrite(const std::string& name,
                                                  const mllm::models::qwen2vl::Qwen2VLConfig& cfg,
                                                  int32_t start_block = 0,
                                                  int32_t active_blocks = -1,
                                                  bool skip_merger = false,
                                                  bool skip_patch_embed = false)
      : mllm::nn::Module(name) {
    start_block_ = start_block;
    active_blocks_ = active_blocks;
    skip_merger_ = skip_merger;
    skip_patch_embed_ = skip_patch_embed;
    patch_embed_ = reg<PatchEmbedLinear>("patch_embed", cfg);
    patch_merger_ = reg<mllm::models::qwen2vl::PatchMerger>("merger", cfg);
    blocks_ = reg<mllm::nn::ModuleList<Qwen2VLVisionBlockAOTRewrite>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];

    if (!skip_patch_embed_) { hidden_states = patch_embed_(hidden_states)[0]; }
    auto num_blocks = active_blocks_ < 0 ? static_cast<int32_t>(blocks_.list().size()) : active_blocks_;
    MLLM_RT_ASSERT(start_block_ >= 0);
    MLLM_RT_ASSERT(num_blocks >= 0);
    MLLM_RT_ASSERT(start_block_ + num_blocks <= static_cast<int32_t>(blocks_.list().size()));
    for (int32_t i = 0; i < num_blocks; ++i) {
      hidden_states = blocks_.list()[start_block_ + i](hidden_states, embedding_sin, embedding_cos)[0];
    }
    if (!skip_merger_) { hidden_states = patch_merger_(hidden_states)[0]; }

    return {hidden_states};
  }
};

std::set<mllm::OpTypes> currentQnnAOTVisitorOps() {
  return {
      mllm::OpTypes::kEmbedding,
      mllm::OpTypes::kCastType,
      mllm::OpTypes::kAdd,
      mllm::OpTypes::kMul,
      mllm::OpTypes::kNeg,
      mllm::OpTypes::kReshape,
      mllm::OpTypes::kView,
      mllm::OpTypes::kIndex,
      mllm::OpTypes::kGather,
      mllm::OpTypes::kRMSNorm,
      mllm::OpTypes::kLinear,
      mllm::OpTypes::kTranspose,
      mllm::OpTypes::kSlice,
      mllm::OpTypes::kConcat,
      mllm::OpTypes::kRepeat,
      mllm::OpTypes::kMatMul,
      mllm::OpTypes::kReduceMax,
      mllm::OpTypes::kReduceMin,
      mllm::OpTypes::kMean,
      mllm::OpTypes::kReduceSum,
      mllm::OpTypes::kEqual,
      mllm::OpTypes::kWhere,
      mllm::OpTypes::kSoftmax,
      mllm::OpTypes::kSigmoid,
      mllm::OpTypes::kConv2D,
      mllm::OpTypes::kGELU,
      mllm::OpTypes::kLayerNorm,
  };
}

struct OpSummary {
  std::map<mllm::OpTypes, int> counts;
  std::map<mllm::OpTypes, std::vector<std::string>> names;
};

OpSummary summarizeLinalgOps(const mllm::ir::IRContext::ptr_t& ir) {
  OpSummary summary;
  auto module = ir->topLevelOp()->cast_<mllm::ir::ModuleOp>();
  auto module_writer = mllm::ir::IRWriter(ir, module->getTopRegion());

  module_writer.walk<mllm::ir::graph::SubGraphOp>(
      [&](mllm::ir::IRWriter& /*writer*/,
          const mllm::ir::graph::SubGraphOp::ptr_t& subgraph) -> mllm::ir::IRWriter::WalkResult {
        auto graph_writer = mllm::ir::IRWriter(ir, subgraph->getTopRegion());
        graph_writer.walk<mllm::ir::linalg::LinalgIROp>(
            [&](mllm::ir::IRWriter& /*inner_writer*/,
                const mllm::ir::linalg::LinalgIROp::ptr_t& op) -> mllm::ir::IRWriter::WalkResult {
              const auto type = op->getAOp()->getOpType();
              summary.counts[type] += 1;
              auto name = op->getAOp()->getName();
              if (name.empty()) { name = "<unnamed>"; }
              auto& names = summary.names[type];
              if (names.size() < 8) { names.push_back(name); }
              return mllm::ir::IRWriter::WALK_CONTINUE;
            });
        return mllm::ir::IRWriter::WALK_CONTINUE;
      });

  return summary;
}

void printSummary(const OpSummary& summary) {
  const auto supported = currentQnnAOTVisitorOps();

  fmt::print("\n{:=^72}\n", " Visual IR op summary ");
  for (const auto& [type, count] : summary.counts) {
    const bool ok = supported.count(type) > 0;
    fmt::print("{:<16} {:>5}   {}\n", mllm::optype2Str(type), count, ok ? "visitor exists" : "missing visitor");
    const auto names_it = summary.names.find(type);
    if (names_it != summary.names.end()) {
      for (const auto& name : names_it->second) { fmt::print("  - {}\n", name); }
    }
  }

  fmt::print("\n{:=^72}\n", " Missing current QNN AOT visitors ");
  bool has_missing = false;
  for (const auto& [type, count] : summary.counts) {
    if (supported.count(type) > 0) { continue; }
    has_missing = true;
    fmt::print("- {}: {}\n", mllm::optype2Str(type), count);
  }
  if (!has_missing) { fmt::print("No missing visitor found in this raw visual trace.\n"); }
  fmt::print("{:=^72}\n\n", "");
}

void reshapePatchEmbedConv3DWeightForLinear(const mllm::ParameterFile::ptr_t& params,
                                            const mllm::models::qwen2vl::Qwen2VLConfig& cfg) {
  const std::string weight_name = "visual.patch_embed.proj.weight";
  if (!params->has(weight_name)) { MLLM_ERROR_EXIT(mllm::ExitCode::kIOError, "Missing {}", weight_name); }

  const int32_t patch_dim = cfg.visual_in_chans * cfg.visual_temporal_patch_size * cfg.visual_patch_size * cfg.visual_patch_size;
  auto weight = params->pull(weight_name);
  MLLM_RT_ASSERT_EQ(weight.numel(), cfg.visual_embed_dim * patch_dim);

  params->remove(weight_name);
  params->push(weight_name, weight.view({cfg.visual_embed_dim, patch_dim}));
}

template<typename VisualModelT>
std::pair<mllm::ir::IRContext::ptr_t, mllm::Tensor> traceAndReportVisual(VisualModelT& visual,
                                                                         const std::string& output_ir,
                                                                         mllm::Tensor img,
                                                                         mllm::Tensor visual_embedding_sin,
                                                                         mllm::Tensor visual_embedding_cos) {
  mllm::ir::lowlevel::traceStart();
  mllm::ir::lowlevel::traceComment("visual inputs: img, visual_embedding_sin, visual_embedding_cos");
  auto visual_embeddings = mllm::ir::lowlevel::traceModule(visual, img, visual_embedding_sin, visual_embedding_cos)[0];
  auto visual_ir = mllm::ir::lowlevel::traceStop();

  fmt::print("visual output shape     : [{}, {}]\n", visual_embeddings.shape()[0], visual_embeddings.shape()[1]);

  mllm::redirect(output_ir, [&]() { mllm::print(visual_ir); });
  fmt::print("Raw visual IR dumped to : {}\n", output_ir);

  auto summary = summarizeLinalgOps(visual_ir);
  printSummary(summary);

  return {visual_ir, visual_embeddings};
}

void compileVisualSegment(mllm::qnn::aot::QnnAOTEnv& qnn_aot_env,
                          const std::string& qnn_aot_cfg_path,
                          const mllm::ParameterFile::ptr_t& params,
                          const mllm::ir::IRContext::ptr_t& visual_ir,
                          const std::string& qnn_graph_name) {
  mllm::ir::PassManager pm(visual_ir);
  pm.reg(mllm::qnn::aot::createQnnAOTSimpleLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_path, params, qnn_graph_name));
  if (!pm.run()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Visual QNN AOT lowering failed for graph {}.", qnn_graph_name);
  }
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Visual-capable .mllm model path.").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model file version: v1/v2.").def("v2");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path.").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Qwen2-VL config path.").required(true);
  auto& image_path = Argparse::add<std::string>("-i|--image").help("Image path used to derive real visual shapes.").required(true);
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("Prompt text.").def("describe this picture");
  auto& output_ir = Argparse::add<std::string>("-o|--output_ir").help("Output visual IR dump path.").def("qwen2vl_visual_raw.mir");
  auto& compile_context =
      Argparse::add<bool>("--compile_context").help("Lower the traced visual tower to QNN. Save context only if --output_context is set.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT config JSON path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def(defaultQnnAOTEnvPath())
                           .help("QNN AOT environment path.");
  auto& output_context_path =
      Argparse::add<std::string>("--output_context").help("Output QNN visual context path.");
  auto& qnn_graph_name =
      Argparse::add<std::string>("--qnn_graph_name").def("visual").help("QNN graph name stored in the visual context.");
  auto& linear_patch_embed = Argparse::add<bool>("--linear_patch_embed")
                                 .help("Trace an AOT-oriented visual model that rewrites Conv3D patch embedding to Linear.");
  auto& aot_rewrite = Argparse::add<bool>("--aot_rewrite")
                          .help("Trace the current AOT rewrite: Linear PatchEmbed plus primitive QuickGELU.");
  auto& start_block =
      Argparse::add<int32_t>("--start_block").def(0).help("First visual block index to trace when --aot_rewrite is set.");
  auto& visual_blocks =
      Argparse::add<int32_t>("--visual_blocks").def(-1).help("Only trace the first N visual blocks when --aot_rewrite is set.");
  auto& skip_patch_embed = Argparse::add<bool>("--skip_patch_embed").help("Skip patch embedding when --aot_rewrite is set.");
  auto& skip_merger = Argparse::add<bool>("--skip_merger").help("Skip the visual merger when --aot_rewrite is set.");
  auto& visual_bundle_context =
      Argparse::add<bool>("--visual_bundle_context")
          .help("Compile the standard 6 visual graphs into one context: patch, four 8-block segments, merger.");
  auto& visual_bundle_layout =
      Argparse::add<std::string>("--visual_bundle_layout")
          .def("6x8")
          .help("Visual bundle layout: 6x8, tail4, tail2, early2 or block1.");

  for (int32_t i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      Argparse::printHelp();
      return 0;
    }
  }

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV2;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() != "v2") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--model_version must be v1 or v2.");
  }

  auto cfg = mllm::models::qwen2vl::Qwen2VLConfig(config_path.get());
  auto tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get());
  auto params = mllm::load(model_path.get(), file_version);

  auto inputs = tokenizer.convertMessage({.prompt = prompt.get(), .img_file_path = image_path.get()});
  auto img = inputs.at("img");
  auto grid_thw = inputs.at("grid_thw");
  const auto visual_patch_tokens = img.shape()[0];
  const auto patch_flat_dim = img.shape()[1];
  if (aot_rewrite.isSet() && skip_patch_embed.isSet()) {
    img = mllm::Tensor::empty({visual_patch_tokens, cfg.visual_embed_dim}, mllm::kFloat32, mllm::kCPU).alloc();
  }

  auto inv_freq = mllm::models::qwen2vl::makeVisualRoPEInvFreq(cfg.visual_embed_dim / cfg.visual_num_heads, 10000.0);
  auto pos_ids = mllm::models::qwen2vl::makeVisualRotaryPosEmbIds(grid_thw, cfg.visual_spatial_merge_size);
  auto rotary_pos_emb_full = mllm::models::qwen2vl::makeVisualRotaryPosEmbFull(inv_freq, visual_patch_tokens);
  auto pos_emb = mllm::models::qwen2vl::makeVisualRotaryPosEmb(rotary_pos_emb_full, pos_ids, grid_thw);
  auto [visual_embedding_sin, visual_embedding_cos] = mllm::models::qwen2vl::makeVisualRotarySinCos(pos_emb);
  if (aot_rewrite.isSet()) {
    const int32_t half_dim = cfg.visual_embed_dim / cfg.visual_num_heads / 2;
    visual_embedding_sin = visual_embedding_sin.view({1, -1, 1, half_dim}, true);
    visual_embedding_cos = visual_embedding_cos.view({1, -1, 1, half_dim}, true);
  }

  fmt::print("Tracing Qwen2-VL visual tower.\n");
  fmt::print("linear patch embedding  : {}\n", (linear_patch_embed.isSet() || aot_rewrite.isSet()) ? "enabled" : "disabled");
  fmt::print("primitive QuickGELU     : {}\n", aot_rewrite.isSet() ? "enabled" : "disabled");
  fmt::print("start block             : {}\n", start_block.get());
  fmt::print("visual blocks           : {}\n", visual_blocks.get());
  fmt::print("skip patch embedding    : {}\n", skip_patch_embed.isSet() ? "enabled" : "disabled");
  fmt::print("skip merger             : {}\n", skip_merger.isSet() ? "enabled" : "disabled");
  fmt::print("QNN graph name          : {}\n", qnn_graph_name.get());
  fmt::print("img shape              : [{}, {}]\n", visual_patch_tokens, patch_flat_dim);
  fmt::print("grid_thw shape          : [{}, {}]\n", grid_thw.shape()[0], grid_thw.shape()[1]);
  fmt::print("visual sin/cos shape    :");
  for (auto dim : visual_embedding_sin.shape()) { fmt::print(" {}", dim); }
  fmt::print("\n");

  mllm::ir::IRContext::ptr_t visual_ir;
  const bool should_trace_single_graph = !visual_bundle_context.isSet();

  if (visual_bundle_context.isSet() && !aot_rewrite.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--visual_bundle_context requires --aot_rewrite.");
  }

  if (should_trace_single_graph && aot_rewrite.isSet()) {
    reshapePatchEmbedConv3DWeightForLinear(params, cfg);
    auto visual = Qwen2VisionTransformerPretrainedModelAOTRewrite("visual", cfg, start_block.get(), visual_blocks.get(),
                                                                  skip_merger.isSet(), skip_patch_embed.isSet());
    visual.load(params);
    visual_ir = traceAndReportVisual(visual, output_ir.get(), img, visual_embedding_sin, visual_embedding_cos).first;
  } else if (should_trace_single_graph && linear_patch_embed.isSet()) {
    reshapePatchEmbedConv3DWeightForLinear(params, cfg);
    auto visual = Qwen2VisionTransformerPretrainedModelLinearPatch("visual", cfg);
    visual.load(params);
    visual_ir = traceAndReportVisual(visual, output_ir.get(), img, visual_embedding_sin, visual_embedding_cos).first;
  } else if (should_trace_single_graph) {
    auto visual = mllm::models::qwen2vl::Qwen2VisionTransformerPretrainedModel("visual", cfg);
    visual.load(params);
    visual_ir = traceAndReportVisual(visual, output_ir.get(), img, visual_embedding_sin, visual_embedding_cos).first;
  }

  if (compile_context.isSet()) {
    if (!aot_rewrite.isSet()) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--compile_context currently requires --aot_rewrite.");
    }
    if (!qnn_aot_cfg_files.isSet()) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--compile_context requires --aot_config.");
    }

    auto qnn_aot_env =
        mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));
    if (visual_bundle_context.isSet()) {
      reshapePatchEmbedConv3DWeightForLinear(params, cfg);
      struct Segment {
        std::string graph_name;
        int32_t start_block;
        int32_t visual_blocks;
        bool skip_patch_embed;
        bool skip_merger;
        std::vector<int32_t> input_shape;
        std::string ir_name;
      };
      std::vector<Segment> segments;
      const std::string bundle_layout = visual_bundle_layout.get();
      if (bundle_layout == "6x8") {
        segments = {
            {"visual_patch_embed", 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed"},
            {"visual_blocks_0_8", 0, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_8"},
            {"visual_blocks_8_16", 8, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_8_16"},
            {"visual_blocks_16_24", 16, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_16_24"},
            {"visual_blocks_24_32", 24, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_32"},
            {"visual_merger", 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger"},
        };
      } else if (bundle_layout == "tail4") {
        segments = {
            {"visual_patch_embed", 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed"},
            {"visual_blocks_0_8", 0, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_8"},
            {"visual_blocks_8_16", 8, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_8_16"},
            {"visual_blocks_16_20", 16, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_16_20"},
            {"visual_blocks_20_24", 20, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_20_24"},
            {"visual_blocks_24_28", 24, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_28"},
            {"visual_blocks_28_32", 28, 4, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_28_32"},
            {"visual_merger", 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger"},
        };
      } else if (bundle_layout == "tail2") {
        segments = {
            {"visual_patch_embed", 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed"},
            {"visual_blocks_0_24", 0, 24, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_24"},
            {"visual_blocks_24_26", 24, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_26"},
            {"visual_blocks_26_28", 26, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_26_28"},
            {"visual_blocks_28_30", 28, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_28_30"},
            {"visual_blocks_30_32", 30, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_30_32"},
            {"visual_merger", 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger"},
        };
      } else if (bundle_layout == "early2") {
        segments = {
            {"visual_patch_embed", 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed"},
            {"visual_blocks_0_2", 0, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_0_2"},
            {"visual_blocks_2_4", 2, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_2_4"},
            {"visual_blocks_4_6", 4, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_4_6"},
            {"visual_blocks_6_8", 6, 2, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_6_8"},
            {"visual_blocks_8_16", 8, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_8_16"},
            {"visual_blocks_16_24", 16, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_16_24"},
            {"visual_blocks_24_32", 24, 8, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, "blocks_24_32"},
            {"visual_merger", 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger"},
        };
      } else if (bundle_layout == "block1") {
        segments.push_back({"visual_patch_embed", 0, 0, false, true, {visual_patch_tokens, patch_flat_dim}, "patch_embed"});
        for (int32_t i = 0; i < cfg.visual_depth; ++i) {
          const auto graph_name = "visual_blocks_" + std::to_string(i) + "_" + std::to_string(i + 1);
          const auto ir_name = "blocks_" + std::to_string(i) + "_" + std::to_string(i + 1);
          segments.push_back({graph_name, i, 1, true, true, {visual_patch_tokens, cfg.visual_embed_dim}, ir_name});
        }
        segments.push_back({"visual_merger", 0, 0, true, false, {visual_patch_tokens, cfg.visual_embed_dim}, "merger"});
      } else {
        MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--visual_bundle_layout must be 6x8, tail4, tail2, early2 or block1.");
      }

      for (const auto& segment : segments) {
        fmt::print("\n{:=^72}\n", fmt::format(" Compile {} ", segment.graph_name));
        auto segment_img = mllm::Tensor::empty(segment.input_shape, mllm::kFloat32, mllm::kCPU).alloc();
        auto visual = Qwen2VisionTransformerPretrainedModelAOTRewrite("visual", cfg, segment.start_block, segment.visual_blocks,
                                                                      segment.skip_merger, segment.skip_patch_embed);
        visual.load(params);
        auto segment_ir_path = output_ir.get() + "." + segment.ir_name + ".mir";
        auto segment_ir =
            traceAndReportVisual(visual, segment_ir_path, segment_img, visual_embedding_sin, visual_embedding_cos).first;
        compileVisualSegment(qnn_aot_env, qnn_aot_cfg_files.get(), params, segment_ir, segment.graph_name);
      }
    } else {
      compileVisualSegment(qnn_aot_env, qnn_aot_cfg_files.get(), params, visual_ir, qnn_graph_name.get());
    }
    if (output_context_path.isSet()) {
      qnn_aot_env.saveContext("context.0", output_context_path.get());
      mllm::print("Visual QNN AOT context saved to " + output_context_path.get());
    } else {
      mllm::print("Visual QNN AOT lowering/finalize completed; context was not saved.");
    }
  }

  mllm::print("Visual trace diagnostic completed.");
});
