// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <nlohmann/json.hpp>
#include "mllm/core/aops/Conv1DOp.hpp"
#include "mllm/core/aops/Conv3DOp.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/aops/MatMulOp.hpp"
#include "mllm/core/aops/FillOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/aops/EmbeddingOp.hpp"
#include "mllm/core/aops/MultimodalRoPEOp.hpp"
#include "mllm/core/aops/KVCacheOp.hpp"
#include "mllm/core/aops/CausalMaskOp.hpp"
#include "mllm/core/aops/SoftmaxOp.hpp"
#include "mllm/core/aops/TransposeOp.hpp"
#include "mllm/core/aops/RMSNormOp.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"
#include "mllm/core/aops/X2XOp.hpp"
#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/aops/SplitOp.hpp"
#include "mllm/core/aops/STFTOp.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/core/aops/RepeatOp.hpp"
#include "mllm/core/aops/PermuteOp.hpp"
#include "mllm/core/aops/GELUOp.hpp"
#include "mllm/core/aops/LayerNormOp.hpp"
#include "mllm/core/aops/VisionRoPEOp.hpp"
#include "mllm/core/aops/QuickGELUOp.hpp"
#include "mllm/core/aops/CloneOp.hpp"
#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/aops/ReduceOps.hpp"
#include "mllm/core/aops/ReLUOp.hpp"
#include "mllm/core/aops/ContiguousOp.hpp"
#include "mllm/core/aops/ReshapeOp.hpp"
#include "mllm/core/aops/SliceOp.hpp"
#include "mllm/core/aops/ParamOp.hpp"
#include "mllm/core/aops/IndexOp.hpp"
#include "mllm/core/aops/TopKOp.hpp"
#include "mllm/core/aops/CopyOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/compile/jit/interpreter/AopsFromJson.hpp"

namespace mllm::jit::interpreter {

BaseOp::ptr_t aopsFromJson(const nlohmann::json& json) {
  if (json.contains("op_type")) {
    std::string op_type = json["op_type"];

    // Route to specific fromJson functions based on op_type
    if (op_type == "Conv1D") {
      return __conv1dFromJson(json);
    } else if (op_type == "Conv3D") {
      return __conv3dFromJson(json);
    } else if (op_type == "Linear") {
      return __linearFromJson(json);
    } else if (op_type == "MatMul") {
      return __matmulFromJson(json);
    } else if (op_type == "Fill") {
      return __fillFromJson(json);
    } else if (op_type == "Add") {
      return __addFromJson(json);
    } else if (op_type == "Sub") {
      return __subFromJson(json);
    } else if (op_type == "Mul") {
      return __mulFromJson(json);
    } else if (op_type == "Div") {
      return __divFromJson(json);
    } else if (op_type == "Abs") {
      return __absFromJson(json);
    } else if (op_type == "Log") {
      return __logFromJson(json);
    } else if (op_type == "Embedding") {
      return __embeddingFromJson(json);
    } else if (op_type == "MultimodalRoPE") {
      return __multimodalRopeFromJson(json);
    } else if (op_type == "KVCache") {
      return __kvCacheFromJson(json);
    } else if (op_type == "CausalMask") {
      return __causalMaskFromJson(json);
    } else if (op_type == "Softmax") {
      return __softmaxFromJson(json);
    } else if (op_type == "Transpose") {
      return __transposeFromJson(json);
    } else if (op_type == "RMSNorm") {
      return __rmsNormFromJson(json);
    } else if (op_type == "SiLU") {
      return __siluFromJson(json);
    } else if (op_type == "CastType") {
      return __castTypeFromJson(json);
    } else if (op_type == "X2X") {
      return __x2xFromJson(json);
    } else if (op_type == "View") {
      return __viewFromJson(json);
    } else if (op_type == "Split") {
      return __splitFromJson(json);
    } else if (op_type == "STFT") {
      return __stftFromJson(json);
    } else if (op_type == "FlashAttention2") {
      return __flashAttention2FromJson(json);
    } else if (op_type == "Repeat") {
      return __repeatFromJson(json);
    } else if (op_type == "Permute") {
      return __permuteFromJson(json);
    } else if (op_type == "GELU") {
      return __geluFromJson(json);
    } else if (op_type == "LayerNorm") {
      return __layerNormFromJson(json);
    } else if (op_type == "VisionRoPE") {
      return __visionRopeFromJson(json);
    } else if (op_type == "QuickGELU") {
      return __quickGeluFromJson(json);
    } else if (op_type == "Copy") {
      return __copyFromJson(json);
    } else if (op_type == "Clone") {
      return __cloneFromJson(json);
    } else if (op_type == "Neg") {
      return __negFromJson(json);
    } else if (op_type == "Concat") {
      return __concatFromJson(json);
    } else if (op_type == "ReduceMax") {
      return __reduceMaxFromJson(json);
    } else if (op_type == "ReduceMin") {
      return __reduceMinFromJson(json);
    } else if (op_type == "ReduceSum") {
      return __reduceSumFromJson(json);
    } else if (op_type == "ReLU") {
      return __reluFromJson(json);
    } else if (op_type == "Contiguous") {
      return __contiguousFromJson(json);
    } else if (op_type == "Reshape") {
      return __reshapeFromJson(json);
    } else if (op_type == "Slice") {
      return __sliceFromJson(json);
    } else if (op_type == "Param") {
      return __paramFromJson(json);
    } else if (op_type == "Index") {
      return __indexFromJson(json);
    } else if (op_type == "TopK") {
      return __topkFromJson(json);
    } else if (op_type == "Mean") {
      return __meanFromJson(json);
    } else if (op_type == "Clip") {
      return __clipFromJson(json);
    } else if (op_type == "Exp") {
      return __expFromJson(json);
    } else if (op_type == "Sin") {
      return __sinFromJson(json);
    } else if (op_type == "Cos") {
      return __cosFromJson(json);
    }
  }

  MLLM_WARN("Unsupported op in aopsFromJson: {}", json.dump());
  return nullptr;
}

BaseOp::ptr_t __conv1dFromJson(const nlohmann::json& json) {
  aops::Conv1DOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("in_channels")) options.in_channels = opts["in_channels"];
    if (opts.contains("out_channels")) options.out_channels = opts["out_channels"];
    if (opts.contains("kernel_size")) options.kernel_size = opts["kernel_size"];
    if (opts.contains("stride")) options.stride = opts["stride"];
    if (opts.contains("bias")) options.bias = opts["bias"];
    if (opts.contains("padding")) options.padding = opts["padding"];
    if (opts.contains("groups")) options.groups = opts["groups"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kConv1D, options);
  return op;
}

BaseOp::ptr_t __conv3dFromJson(const nlohmann::json& json) {
  aops::Conv3DOpOptions options;
  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("in_channels")) options.in_channels = opts["in_channels"];
    if (opts.contains("out_channels")) options.out_channels = opts["out_channels"];
    if (opts.contains("kernel_size")) {
      // Handle array of kernel sizes
      if (opts["kernel_size"].is_array()) {
        for (const auto& item : opts["kernel_size"]) { options.kernel_size.push_back(item); }
      }
    }
    if (opts.contains("stride")) {
      // Handle array of strides
      if (opts["stride"].is_array()) {
        for (const auto& item : opts["stride"]) { options.stride.push_back(item); }
      }
    }
    if (opts.contains("bias")) options.bias = opts["bias"];
    if (opts.contains("impl_type")) {
      // Convert string to Conv3DOpImplType enum
      std::string impl_type_str = opts["impl_type"];
      options.impl_type = aops::str2Conv3DOpImplType(impl_type_str);
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kConv3D, options);
  return op;
}

BaseOp::ptr_t __linearFromJson(const nlohmann::json& json) {
  aops::LinearOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("in_channels")) options.in_channels = opts["in_channels"];
    if (opts.contains("out_channels")) options.out_channels = opts["out_channels"];
    if (opts.contains("bias")) options.bias = opts["bias"];
    if (opts.contains("impl_type")) {
      // Convert string to LinearImplTypes enum
      std::string impl_type_str = opts["impl_type"];
      options.impl_type = aops::str2LinearImplTypes(impl_type_str);
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kLinear, options);
  return op;
}

BaseOp::ptr_t __matmulFromJson(const nlohmann::json& json) {
  aops::MatMulOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("transpose_a")) options.transpose_a = opts["transpose_a"];
    if (opts.contains("transpose_b")) options.transpose_b = opts["transpose_b"];
    if (opts.contains("matmul_type")) {
      // Convert string to MatMulOpType enum
      std::string matmul_type_str = opts["matmul_type"];
      options.matmul_type = aops::str2MatMulOpType(matmul_type_str);
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kMatMul, options);
  return op;
}

BaseOp::ptr_t __fillFromJson(const nlohmann::json& json) {
  aops::FillOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kFill, options);
  return op;
}

BaseOp::ptr_t __addFromJson(const nlohmann::json& json) {
  aops::AddOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kAdd, options);
  return op;
}

BaseOp::ptr_t __subFromJson(const nlohmann::json& json) {
  aops::SubOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSub, options);
  return op;
}

BaseOp::ptr_t __mulFromJson(const nlohmann::json& json) {
  aops::MulOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kMul, options);
  return op;
}

BaseOp::ptr_t __divFromJson(const nlohmann::json& json) {
  aops::DivOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kDiv, options);
  return op;
}

BaseOp::ptr_t __absFromJson(const nlohmann::json& json) {
  aops::AbsOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kAbs, options);
  return op;
}

BaseOp::ptr_t __logFromJson(const nlohmann::json& json) {
  aops::LogOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kLog, options);
  return op;
}

BaseOp::ptr_t __embeddingFromJson(const nlohmann::json& json) {
  aops::EmbeddingOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("vocab_size")) options.vocab_size = opts["vocab_size"];
    if (opts.contains("hidden_size")) options.hidden_size = opts["hidden_size"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kEmbedding, options);
  return op;
}

BaseOp::ptr_t __multimodalRopeFromJson(const nlohmann::json& json) {
  aops::MultimodalRoPEOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("type")) options.type = static_cast<aops::MultimodalRoPEOpOptionsType>(static_cast<int>(opts["type"]));

    if (opts.contains("qwen2vl_options")) {
      const auto& qwen_opts = opts["qwen2vl_options"];
      if (qwen_opts.contains("rope_theta")) options.qwen2vl_options.rope_theta = qwen_opts["rope_theta"];
      if (qwen_opts.contains("max_position_embeddings")) {
        options.qwen2vl_options.max_position_embeddings = qwen_opts["max_position_embeddings"];
      }
      if (qwen_opts.contains("mrope_section")) {
        for (const auto& item : qwen_opts["mrope_section"]) { options.qwen2vl_options.mrope_section.push_back(item); }
      }
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kMultimodalRoPE, options);
  return op;
}

BaseOp::ptr_t __kvCacheFromJson(const nlohmann::json& json) {
  aops::KVCacheOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("layer_idx")) options.layer_idx = opts["layer_idx"];
    if (opts.contains("q_head")) options.q_head = opts["q_head"];
    if (opts.contains("kv_head")) options.kv_head = opts["kv_head"];
    if (opts.contains("head_dim")) options.head_dim = opts["head_dim"];
    if (opts.contains("use_fa2")) options.use_fa2 = opts["use_fa2"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kKVCache, options);
  return op;
}

BaseOp::ptr_t __causalMaskFromJson(const nlohmann::json& json) {
  aops::CausalMaskOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("sliding_window")) options.sliding_window = opts["sliding_window"];
    if (opts.contains("window_size")) options.window_size = opts["window_size"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kCausalMask, options);
  return op;
}

BaseOp::ptr_t __softmaxFromJson(const nlohmann::json& json) {
  aops::SoftmaxOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("axis")) options.axis = opts["axis"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSoftmax, options);
  return op;
}

BaseOp::ptr_t __transposeFromJson(const nlohmann::json& json) {
  aops::TransposeOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim0")) options.dim0 = opts["dim0"];
    if (opts.contains("dim1")) options.dim1 = opts["dim1"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kTranspose, options);
  return op;
}

BaseOp::ptr_t __rmsNormFromJson(const nlohmann::json& json) {
  aops::RMSNormOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("epsilon")) options.epsilon = opts["epsilon"];
    if (opts.contains("add_unit_offset")) options.add_unit_offset = opts["add_unit_offset"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kRMSNorm, options);
  return op;
}

BaseOp::ptr_t __siluFromJson(const nlohmann::json& json) {
  aops::SiLUOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSiLU, options);
  return op;
}

BaseOp::ptr_t __castTypeFromJson(const nlohmann::json& json) {
  aops::CastTypeOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dtype")) {
      // TODO: Implement str2Type function
      // For now, using default
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kCastType, options);
  return op;
}

BaseOp::ptr_t __x2xFromJson(const nlohmann::json& json) {
  aops::X2XOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("device")) { options.device = str2DeviceType(opts["device"]); }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kX2X, options);
  return op;
}

BaseOp::ptr_t __viewFromJson(const nlohmann::json& json) {
  aops::ViewOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("to_shape")) {
      for (const auto& item : opts["to_shape"]) { options.to_shape.push_back(item); }
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kView, options);
  return op;
}

BaseOp::ptr_t __splitFromJson(const nlohmann::json& json) {
  aops::SplitOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("split_size_or_sections")) {
      for (const auto& item : opts["split_size_or_sections"]) { options.split_size_or_sections.push_back(item); }
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSplit, options);
  return op;
}

BaseOp::ptr_t __stftFromJson(const nlohmann::json& json) {
  aops::STFTOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("n_fft")) options.n_fft = opts["n_fft"];
    if (opts.contains("hop_length")) options.hop_length = opts["hop_length"];
    if (opts.contains("win_length")) options.win_length = opts["win_length"];
    if (opts.contains("onesided")) options.onesided = opts["onesided"];
    if (opts.contains("center")) options.center = opts["center"];
    if (opts.contains("pad_mode")) options.pad_mode = opts["pad_mode"];
    if (opts.contains("return_complex")) options.return_complex = opts["return_complex"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSTFT, options);
  return op;
}

BaseOp::ptr_t __flashAttention2FromJson(const nlohmann::json& json) {
  aops::FlashAttention2OpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("B")) options.B = opts["B"];
    if (opts.contains("q_head")) options.q_head = opts["q_head"];
    if (opts.contains("kv_head")) options.kv_head = opts["kv_head"];
    if (opts.contains("D")) options.D = opts["D"];
    if (opts.contains("hp_exp")) options.hp_exp = opts["hp_exp"];
    if (opts.contains("causal_mask")) options.causal_mask = opts["causal_mask"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kFlashAttention2, options);
  return op;
}

BaseOp::ptr_t __repeatFromJson(const nlohmann::json& json) {
  aops::RepeatOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("repeat_times")) options.repeat_times = opts["repeat_times"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kRepeat, options);
  return op;
}

BaseOp::ptr_t __permuteFromJson(const nlohmann::json& json) {
  aops::PermuteOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("axis")) {
      for (const auto& item : opts["axis"]) { options.axis.push_back(item); }
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kPermute, options);
  return op;
}

BaseOp::ptr_t __geluFromJson(const nlohmann::json& json) {
  aops::GELUOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kGELU, options);
  return op;
}

BaseOp::ptr_t __layerNormFromJson(const nlohmann::json& json) {
  aops::LayerNormOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("normalized_shape")) {
      for (const auto& item : opts["normalized_shape"]) { options.normalized_shape.push_back(item); }
    }
    if (opts.contains("elementwise_affine")) options.elementwise_affine = opts["elementwise_affine"];
    if (opts.contains("bias")) options.bias = opts["bias"];
    if (opts.contains("eps")) options.eps = opts["eps"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kLayerNorm, options);
  return op;
}

BaseOp::ptr_t __visionRopeFromJson(const nlohmann::json& json) {
  aops::VisionRoPEOpOptions options{};

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("type")) options.type = static_cast<aops::VisionRoPEOpOptionsType>(static_cast<int>(opts["type"]));

    if (opts.contains("qwen2vl_rope_op_options")) {
      const auto& qwen_opts = opts["qwen2vl_rope_op_options"];
      if (qwen_opts.contains("dims")) options.qwen2vl_rope_op_options.dims = qwen_opts["dims"];
      if (qwen_opts.contains("spatial_merge_size")) {
        options.qwen2vl_rope_op_options.spatial_merge_size = qwen_opts["spatial_merge_size"];
      }
      if (qwen_opts.contains("theta")) options.qwen2vl_rope_op_options.theta = qwen_opts["theta"];
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kVisionRoPE, options);
  return op;
}

BaseOp::ptr_t __quickGeluFromJson(const nlohmann::json& json) {
  aops::QuickGELUOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kQuickGELU, options);
  return op;
}

BaseOp::ptr_t __copyFromJson(const nlohmann::json& json) {
  aops::CopyOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kCopy, options);
  return op;
}

BaseOp::ptr_t __cloneFromJson(const nlohmann::json& json) {
  aops::CloneOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kClone, options);
  return op;
}

BaseOp::ptr_t __negFromJson(const nlohmann::json& json) {
  aops::NegOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kNeg, options);
  return op;
}

BaseOp::ptr_t __concatFromJson(const nlohmann::json& json) {
  aops::ConcatOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kConcat, options);
  return op;
}

BaseOp::ptr_t __reduceMaxFromJson(const nlohmann::json& json) {
  aops::ReduceMaxOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("keep_dim")) options.keep_dim = opts["keep_dim"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kReduceMax, options);
  return op;
}

BaseOp::ptr_t __reduceMinFromJson(const nlohmann::json& json) {
  aops::ReduceMinOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("keep_dim")) options.keep_dim = opts["keep_dim"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kReduceMin, options);
  return op;
}

BaseOp::ptr_t __reduceSumFromJson(const nlohmann::json& json) {
  aops::ReduceSumOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("keep_dim")) options.keep_dim = opts["keep_dim"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kReduceSum, options);
  return op;
}

BaseOp::ptr_t __reluFromJson(const nlohmann::json& json) {
  aops::ReLUOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kReLU, options);
  return op;
}

BaseOp::ptr_t __contiguousFromJson(const nlohmann::json& json) {
  aops::ContiguousOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kContiguous, options);
  return op;
}

BaseOp::ptr_t __reshapeFromJson(const nlohmann::json& json) {
  aops::ReshapeOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kReshape, options);
  return op;
}

BaseOp::ptr_t __sliceFromJson(const nlohmann::json& json) {
  aops::SliceOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("indices")) {
      for (const auto& item : opts["indices"]) {
        mllm::SliceIndicesPair slice_index;
        if (item.contains("start")) slice_index.start_ = item["start"];
        if (item.contains("end")) slice_index.end_ = item["end"];
        if (item.contains("step")) slice_index.step_ = item["step"];
        options.indices_.push_back(slice_index);
      }
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSlice, options);
  return op;
}

BaseOp::ptr_t __paramFromJson(const nlohmann::json& json) {
  aops::ParamOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("name")) options.name = opts["name"];
    if (opts.contains("shape")) {
      for (const auto& item : opts["shape"]) { options.shape.push_back(item); }
    }
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kParam, options);
  return op;
}

BaseOp::ptr_t __indexFromJson(const nlohmann::json& json) {
  aops::IndexOpOptions options;

  // Note: ComplexIndexingList is a complex structure that would require
  // more detailed parsing based on the actual JSON structure.
  // For now, we just create an empty options object.

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kIndex, options);
  return op;
}

BaseOp::ptr_t __topkFromJson(const nlohmann::json& json) {
  aops::TopKOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("k")) options.k = opts["k"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("largest")) options.largest = opts["largest"];
    if (opts.contains("sorted")) options.sorted = opts["sorted"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kTopK, options);
  return op;
}

BaseOp::ptr_t __meanFromJson(const nlohmann::json& json) {
  aops::MeanOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("dim")) options.dim = opts["dim"];
    if (opts.contains("keep_dim")) options.keep_dim = opts["keep_dim"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kMean, options);
  return op;
}

BaseOp::ptr_t __clipFromJson(const nlohmann::json& json) {
  aops::ClipOpOptions options;

  if (json.contains("op_options")) {
    const auto& opts = json["op_options"];
    if (opts.contains("min_val")) options.min_val = opts["min_val"];
    if (opts.contains("max_val")) options.max_val = opts["max_val"];
  }

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kClip, options);
  return op;
}

BaseOp::ptr_t __expFromJson(const nlohmann::json& json) {
  aops::ExpOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kExp, options);
  return op;
}

BaseOp::ptr_t __sinFromJson(const nlohmann::json& json) {
  aops::SinOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kSin, options);
  return op;
}

BaseOp::ptr_t __cosFromJson(const nlohmann::json& json) {
  aops::CosOpOptions options;

  DeviceTypes backend = DeviceTypes::kCPU;
  if (json.contains("backend")) { backend = str2DeviceType(json["backend"]); }

  // Use Context to create op
  auto op = Context::instance().getBackend(backend)->createOp(OpTypes::kCos, options);
  return op;
}

}  // namespace mllm::jit::interpreter
