// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>

#include "modeling_qwen2vl_qnn_aot.hpp"
#include "visual_aot_helpers.hpp"

using mllm::Argparse;

namespace {

constexpr float kDefaultInputEmbeddingScale = 0.002563515f;
constexpr int32_t kDefaultInputEmbeddingZeroPoint = 15604;

template <typename ParamsT>
void addCausalMaskParams(const ParamsT& params) {
  params->push("causal_mask.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("causal_mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
}

template <typename ParamsT>
mllm::Tensor makeUInt16AsymTensor(const std::vector<int32_t>& shape,
                                  const std::string& scale_name,
                                  const std::string& zp_name,
                                  const ParamsT& params) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt16);
  tensor = tensor.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  tensor.attach("scale", params->pull(scale_name).impl(), true);
  tensor.attach("zero_point", params->pull(zp_name).impl(), true);
  return tensor;
}

template <typename ParamsT>
mllm::Tensor makeUInt16SymTensor(const std::vector<int32_t>& shape,
                                 const std::string& scale_name,
                                 const ParamsT& params) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt16);
  tensor = tensor.__unsafeSetDType(mllm::kUInt16PerTensorSym);
  tensor.attach("scale", params->pull(scale_name).impl(), true);
  return tensor;
}

inline mllm::Tensor makeUInt16AsymTensor(const std::vector<int32_t>& shape, float scale, int32_t zero_point) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt16);
  tensor = tensor.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  tensor.attach("scale", mllm::Tensor::constant(scale, mllm::kFloat32).impl(), true);
  tensor.attach("zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32).impl(), true);
  return tensor;
}

template <typename ParamsT>
std::unordered_map<std::string, mllm::Tensor> makeTraceInputs(int seq_len,
                                                             int context_len,
                                                             const mllm::models::qwen2vl::Qwen2VLConfig& model_cfg,
                                                             const ParamsT& params,
                                                             bool override_input_embedding_qp,
                                                             float input_embedding_scale,
                                                             int32_t input_embedding_zero_point,
                                                             bool key_cache_uint16) {
  const int head_dim = model_cfg.hidden_size / model_cfg.num_attention_heads;

  std::unordered_map<std::string, mllm::Tensor> trace_inputs;

  if (override_input_embedding_qp) {
    trace_inputs["input_embeddings"] =
        makeUInt16AsymTensor({1, seq_len, model_cfg.hidden_size}, input_embedding_scale, input_embedding_zero_point);
  } else {
    trace_inputs["input_embeddings"] = makeUInt16AsymTensor(
        {1, seq_len, model_cfg.hidden_size}, "model.embed_tokens.scale", "model.embed_tokens.zero_point", params);
  }
  trace_inputs["llm_embedding_sin"] =
      makeUInt16AsymTensor({1, seq_len, head_dim}, "model.sin_embedding_input_qdq.fake_quant.scale",
                           "model.sin_embedding_input_qdq.fake_quant.zero_point", params);
  trace_inputs["llm_embedding_cos"] =
      makeUInt16AsymTensor({1, seq_len, head_dim}, "model.cos_embedding_input_qdq.fake_quant.scale",
                           "model.cos_embedding_input_qdq.fake_quant.zero_point", params);
  trace_inputs["causal_mask"] =
      makeUInt16AsymTensor({1, 1, seq_len, context_len}, "causal_mask.scale", "causal_mask.zero_point", params);

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    auto past_key_name = "past_key_" + std::to_string(i);
    auto past_value_name = "past_value_" + std::to_string(i);

    if (key_cache_uint16) {
      trace_inputs[past_key_name] = makeUInt16SymTensor(
          {
              1,
              model_cfg.num_key_value_heads,
              head_dim,
              context_len - seq_len,
          },
          "model.layers." + std::to_string(i) + ".self_attn.k_rope_add_0_output_qdq.fake_quant.scale",
          params);
    } else {
      trace_inputs[past_key_name] = mllm::Tensor::empty({
          1,
          model_cfg.num_key_value_heads,
          head_dim,
          context_len - seq_len,
      }, mllm::kUInt8PerTensorSym);
      trace_inputs[past_key_name].attach("scale",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale")
                                             .impl(),
                                         true);
      trace_inputs[past_key_name].attach("zero_point",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point")
                                             .impl(),
                                         true);
    }
    trace_inputs[past_value_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        context_len - seq_len,
        head_dim,
    }, mllm::kUInt8PerTensorSym);

    trace_inputs[past_value_name].attach("scale",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale")
                                             .impl(),
                                         true);
    trace_inputs[past_value_name].attach("zero_point",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point")
                                             .impl(),
                                         true);
  }

  return trace_inputs;
}

void compileVisualBundleGraphs(mllm::qnn::aot::QnnAOTEnv& qnn_aot_env,
                               const std::string& visual_aot_cfg_path,
                               const mllm::ParameterFile::ptr_t& visual_params,
                               const mllm::models::qwen2vl::Qwen2VLConfig& visual_cfg,
                               int32_t visual_patch_tokens,
                               int32_t patch_flat_dim,
                               const std::string& bundle_layout,
                               const std::string& visual_ir_prefix,
                               const std::string& graph_suffix) {
  const int32_t half_dim = visual_cfg.visual_embed_dim / visual_cfg.visual_num_heads / 2;
  auto visual_embedding_sin = mllm::Tensor::empty({1, visual_patch_tokens, 1, half_dim}, mllm::kFloat32, mllm::kCPU).alloc();
  auto visual_embedding_cos = mllm::Tensor::empty({1, visual_patch_tokens, 1, half_dim}, mllm::kFloat32, mllm::kCPU).alloc();
  auto visual_attention_mask =
      mllm::Tensor::empty({1, 1, 1, visual_patch_tokens}, mllm::kFloat32, mllm::kCPU).alloc();

  const auto segments = qwen2vl_qnn_aot::makeVisualBundleSegments(bundle_layout,
                                                                  visual_cfg.visual_depth,
                                                                  visual_patch_tokens,
                                                                  patch_flat_dim,
                                                                  visual_cfg,
                                                                  graph_suffix);

  for (const auto& segment : segments) {
    fmt::print("\n{:=^72}\n", fmt::format(" Compile {} ", segment.graph_name));
    auto segment_img = mllm::Tensor::empty(segment.input_shape, mllm::kFloat32, mllm::kCPU).alloc();
    auto visual = qwen2vl_qnn_aot::Qwen2VisionTransformerPretrainedModelAOTRewrite("visual",
                                                                                  visual_cfg,
                                                                                  segment.start_block,
                                                                                  segment.visual_blocks,
                                                                                  segment.skip_merger,
                                                                                  segment.skip_patch_embed);
    visual.load(visual_params);

    mllm::ir::lowlevel::traceStart();
    auto visual_output = segment.visual_blocks > 0
                             ? mllm::ir::lowlevel::traceModule(visual,
                                                               segment_img,
                                                               visual_embedding_sin,
                                                               visual_embedding_cos,
                                                               visual_attention_mask)[0]
                             : mllm::ir::lowlevel::traceModule(visual, segment_img, visual_embedding_sin, visual_embedding_cos)[0];
    auto visual_ir = mllm::ir::lowlevel::traceStop();

    fmt::print("visual segment output shape: [{}, {}]\n", visual_output.shape()[0], visual_output.shape()[1]);
    const auto segment_ir_path = visual_ir_prefix + "." + segment.ir_name + ".mir";
    mllm::redirect(segment_ir_path, [&]() { mllm::print(visual_ir); });
    fmt::print("Visual IR dumped to: {}\n", segment_ir_path);

    mllm::ir::PassManager pm(visual_ir);
    pm.reg(mllm::qnn::aot::createQnnAOTSimpleLoweringPipeline(&qnn_aot_env,
                                                              visual_aot_cfg_path,
                                                              visual_params,
                                                              segment.graph_name));
    if (!pm.run()) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Visual QNN AOT lowering failed for graph {}.", segment.graph_name);
    }
  }
}

void compileVisualBundleGraphsFromImage(mllm::qnn::aot::QnnAOTEnv& qnn_aot_env,
                                        const std::string& visual_aot_cfg_path,
                                        const mllm::ParameterFile::ptr_t& visual_params,
                                        const mllm::models::qwen2vl::Qwen2VLConfig& visual_cfg,
                                        const std::string& tokenizer_path,
                                        const std::string& image_path,
                                        const std::string& prompt,
                                        const std::string& bundle_layout,
                                        const std::string& visual_ir_prefix) {
  if (tokenizer_path.empty() || image_path.empty()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--include_visual_bundle requires --tokenizer and --image.");
  }

  auto tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path);
  auto inputs = tokenizer.convertMessage({.prompt = prompt, .img_file_path = image_path});
  auto img = inputs.at("img");
  compileVisualBundleGraphs(qnn_aot_env,
                            visual_aot_cfg_path,
                            visual_params,
                            visual_cfg,
                            img.shape()[0],
                            img.shape()[1],
                            bundle_layout,
                            visual_ir_prefix,
                            "");
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/")
                           .help("QNN AOT Environment path.");
  auto& output_context_path = Argparse::add<std::string>("-o|--output_context_name").help("Output QNN context path.");
  auto& context_len = Argparse::add<int>("--context_len").help("QNN context length.").def(1024);
  auto& prefill_len = Argparse::add<int>("--prefill_len").help("Prefill graph sequence length.").def(32);
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
      Argparse::add<bool>("--dump_block_outputs").help("Expose per-layer block_out tensors as graph outputs for debugging.");
  auto& dump_layer0_outputs =
      Argparse::add<bool>("--dump_layer0_outputs").help("Expose layer0 fine-grained tensors as graph outputs for debugging.");
  auto& key_cache_dtype =
      Argparse::add<std::string>("--key_cache_dtype").help("Key cache dtype for experimental contexts: uint8 or uint16.").def("uint8");
  auto& include_visual_bundle =
      Argparse::add<bool>("--include_visual_bundle")
          .help("Also compile the Qwen2-VL visual tower bundle graphs into the same QNN context.");
  auto& skip_llm_graphs =
      Argparse::add<bool>("--skip_llm_graphs")
          .help("Only compile requested auxiliary graphs, such as the visual bundle, and skip LLM prefill/decode graphs.");
  auto& visual_model_path = Argparse::add<std::string>("--visual_model")
                                .help("Optional FP32/W32A32 visual-capable .mllm for visual bundle graphs. Defaults to --model_path.");
  auto& visual_config_path = Argparse::add<std::string>("--visual_config")
                                 .help("Optional visual config. Defaults to --config.");
  auto& visual_aot_config_path = Argparse::add<std::string>("--visual_aot_config")
                                     .help("AOT config for visual bundle graphs. Defaults to --aot_config.");
  auto& visual_bundle_layout =
      Argparse::add<std::string>("--visual_bundle_layout")
          .def("6x8")
          .help("Visual bundle layout: single, 6x8, tail4, early2 or block1.");
  auto& visual_bucket_grids = Argparse::add<std::string>("--visual_bucket_grids")
                                  .def("")
                                  .help("Comma-separated visual patch grid buckets HxW. Example: 10x16,12x16,26x36.");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path for visual input shape.");
  auto& image_path = Argparse::add<std::string>("-i|--image").help("image path for visual input shape.");
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("prompt text used with --image.").def("describe this picture");
  auto& visual_ir_prefix =
      Argparse::add<std::string>("--visual_ir_prefix").def("qwen2vl_visual_combined").help("Prefix for visual IR dumps.");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !model_cfg_path.isSet() || !qnn_aot_cfg_files.isSet() || !output_context_path.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Missing required argument.");
    Argparse::printHelp();
    return -1;
  }
  if (prefill_len.get() <= 0 || context_len.get() <= prefill_len.get()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid context_len/prefill_len: {} / {}", context_len.get(),
                    prefill_len.get());
    return -1;
  }
  const bool override_input_embedding_qp = input_embedding_scale.get() > 0.0f || input_embedding_zero_point.get() >= 0;
  if (override_input_embedding_qp && (input_embedding_scale.get() <= 0.0f || input_embedding_zero_point.get() < 0)) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "input embedding override requires both --input_embedding_scale and --input_embedding_zero_point; "
                    "set both to -1 to use model.embed_tokens QP.");
  }
  const bool key_cache_uint16 = key_cache_dtype.get() == "uint16";
  if (!key_cache_uint16 && key_cache_dtype.get() != "uint8") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--key_cache_dtype must be uint8 or uint16.");
  }

  auto model_cfg = mllm::models::qwen2vl::Qwen2VLConfig(model_cfg_path.get());
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  addCausalMaskParams(params);

  mllm::models::qwen2vl::qnn_aot::DebugOutputConfig debug_outputs{
      .dump_block_outputs = dump_block_outputs.isSet(),
      .dump_layer0_outputs = dump_layer0_outputs.isSet(),
      .key_cache_uint16 = key_cache_uint16,
  };
  auto model = mllm::models::qwen2vl::qnn_aot::Qwen2VLForCausalLM(model_cfg, debug_outputs);
  model.load(params);

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  auto trace_and_dump = [&](int seq_len, const std::string& mir_path) {
    mllm::print("Tracing Qwen2-VL LLM QNN AOT graph, seq=" + std::to_string(seq_len));
    auto trace_inputs =
        makeTraceInputs(seq_len, context_len.get(), model_cfg, params, override_input_embedding_qp,
                        input_embedding_scale.get(), input_embedding_zero_point.get(), key_cache_uint16);
    auto ir = model.trace(trace_inputs, {});
    mllm::print("Trace completed, lowering to QNN AOT.");

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect(mir_path, [&]() { mllm::print(ir["model"]); });
    mllm::print("IR dumped to " + mir_path);
  };

  if (skip_llm_graphs.isSet() && !include_visual_bundle.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--skip_llm_graphs requires --include_visual_bundle or another graph family.");
  }

  if (!skip_llm_graphs.isSet()) {
    trace_and_dump(prefill_len.get(), "qwen2vl_qnn_aot_" + std::to_string(prefill_len.get()) + ".mir");
    trace_and_dump(1, "qwen2vl_qnn_aot_1.mir");
  } else {
    mllm::print("Skipping LLM prefill/decode graph compilation.");
  }

  if (include_visual_bundle.isSet()) {
    const auto visual_model = visual_model_path.isSet() ? visual_model_path.get() : model_path.get();
    const auto visual_config = visual_config_path.isSet() ? visual_config_path.get() : model_cfg_path.get();
    const auto visual_aot_config = visual_aot_config_path.isSet() ? visual_aot_config_path.get() : qnn_aot_cfg_files.get();
    auto visual_cfg = mllm::models::qwen2vl::Qwen2VLConfig(visual_config);
    auto visual_params = mllm::load(visual_model, mllm::ModelFileVersion::kV2);
    qwen2vl_qnn_aot::reshapePatchEmbedConv3DWeightForLinear(visual_params, visual_cfg);

    const int32_t patch_flat_dim =
        visual_cfg.visual_in_chans * visual_cfg.visual_temporal_patch_size * visual_cfg.visual_patch_size * visual_cfg.visual_patch_size;
    const auto buckets = qwen2vl_qnn_aot::parseVisualBucketGrids(visual_bucket_grids.get());
    if (!buckets.empty()) {
      const auto bucket_tokens = qwen2vl_qnn_aot::uniqueVisualBucketPatchTokens(buckets);
      fmt::print("Compiling {} visual bucket shape(s): ", bucket_tokens.size());
      for (size_t i = 0; i < bucket_tokens.size(); ++i) { fmt::print("{}{}", i == 0 ? "" : ",", bucket_tokens[i]); }
      fmt::print("\n");
      for (const int32_t patch_tokens : bucket_tokens) {
        compileVisualBundleGraphs(qnn_aot_env,
                                  visual_aot_config,
                                  visual_params,
                                  visual_cfg,
                                  patch_tokens,
                                  patch_flat_dim,
                                  visual_bundle_layout.get(),
                                  visual_ir_prefix.get(),
                                  qwen2vl_qnn_aot::visualGraphSuffixForPatchTokens(patch_tokens));
      }
    } else {
      compileVisualBundleGraphsFromImage(qnn_aot_env,
                                         visual_aot_config,
                                         visual_params,
                                         visual_cfg,
                                         tokenizer_path.get(),
                                         image_path.get(),
                                         prompt.get(),
                                         visual_bundle_layout.get(),
                                         visual_ir_prefix.get());
    }
  }

  qnn_aot_env.saveContext("context.0", output_context_path.get());
  mllm::print("Qwen2-VL QNN AOT compilation completed.");
  mllm::print("Context: " + output_context_path.get());
});
