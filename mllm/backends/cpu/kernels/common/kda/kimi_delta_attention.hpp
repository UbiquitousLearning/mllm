// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

namespace mllm::cpu::kda {

// Kimi Delta Attention recurrent update used by Ling-3.0.
//
// q/k/v/g: [B, S, H, D]
// beta:    [B, S, H] (already passed through sigmoid)
// a_log:   [H]
// dt_bias: [H, D]
// state:   [B, H, D, D], updated in place
// output:  [B, S, H, D]
//
// q and k are L2-normalized inside this function and q is additionally
// scaled by 1 / sqrt(D), matching the FLA KDA reference implementation.
void kimiDeltaAttentionF32(const float* q, const float* k, const float* v, const float* gate_logits, const float* beta,
                           const float* a_log, const float* dt_bias, float* state, float* output, int batch_size,
                           int sequence_length, int num_heads, int head_dim, bool safe_gate, float lower_bound);

void kimiDeltaAttentionF32(const float* q, const float* k, const float* v, const float* gate_logits, const float* beta,
                           const float* a_log, const float* dt_bias, float* state, float* output, int batch_size,
                           int sequence_length, int num_heads, int head_dim, bool safe_gate, float lower_bound,
                           int thread_count);

}  // namespace mllm::cpu::kda
