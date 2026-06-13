// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3.5 QNN AOT compilation variant.
//
// Phase 1 strategy: only full attention layers (6 of 24) + MLP are compiled
// to QNN context binaries for HTP execution. GDN layers remain on CPU.
// This allows us to leverage existing QNN visitors (Linear, RMSNorm, Softmax,
// Sigmoid, MatMul, etc.) without needing GDN-specific custom HTP ops.
//
// The model graph is split per full-attention decoder layer for AOT compilation.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"

// Reuse the base model components — the QNN AOT variant shares the same
// architecture, the difference is in how it's compiled (which layers get
// marked as QNN-offloadable).
#include "mllm/models/qwen3_5/modeling_qwen3_5.hpp"

namespace mllm::models::qwen3_5 {

// The QNN AOT variant uses the same classes as the base model.
// The AOT compilation pipeline (MarkQnnGraphPass, LLM2QnnLoweringPass, etc.)
// will traverse the IR and lower supported ops to QNN.
//
// For Phase 1, the AOT pipeline should:
// 1. Mark full attention decoder layers for QNN compilation
// 2. Leave GDN decoder layers unmarked (they run on CPU)
// 3. Compile attention + MLP subgraphs into QNN context binaries
//
// Usage:
//   auto cfg = Qwen3_5Config("config.json");
//   auto model = Qwen3_5ForCausalLM(cfg);  // same model class
//   // AOT compilation is controlled by the pipeline configuration,
//   // not by a separate model class.

using Qwen3_5ForCausalLMQnnAOT = Qwen3_5ForCausalLM;

}  // namespace mllm::models::qwen3_5
