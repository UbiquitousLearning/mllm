// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cassert>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/primitives.hpp"

#include "mllm/backends/cpu/kernels/common/fa2/mma0.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/mma1.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/rescale.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/softmax.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/init_tmp.hpp"
#include "mllm/backends/cpu/kernels/common/fa2/scale_cast_copy.hpp"

namespace mllm::cpu::fa2 {

template<typename Config_ = DefaultFlashAttention2Config>
struct FlashAttention2WorkSpace {
  using KernelConfig = Config_::KernelConfig;
  using Numeric = typename KernelConfig::Numeric;

  using ElementAccumulator = typename Numeric::ElementAccumulator;
  using ElementCompute = typename Numeric::ElementCompute;

  ElementCompute* acc_s_cast = nullptr;
  ElementAccumulator* acc_o = nullptr;
  ElementAccumulator* acc_s = nullptr;
  ElementAccumulator* logsum = nullptr;
  ElementAccumulator* scoremax = nullptr;
  ElementAccumulator* scoremax_prev = nullptr;
  ElementAccumulator* score_scale = nullptr;
  ElementAccumulator* score_sum = nullptr;
};

template<typename Config_ = DefaultFlashAttention2Config>
struct FlashAttention2 {
  using KernelConfig = Config_::KernelConfig;

  using Arch = typename Config_::Arch;
  using Mma0Layout = typename Config_::Mma0Layout;
  using Mma1Layout = typename Config_::Mma1Layout;
  using Numeric = typename Config_::Numeric;
  using Memory = typename Config_::Memory;
  using Tile = typename Config_::Tile;

  using ElementAccumulator = typename Numeric::ElementAccumulator;
  using ElementCompute = typename Numeric::ElementCompute;

  static constexpr bool kHasCausalMask = Config_::kHasCausalMask;
  static constexpr bool kHasSlidingWindow = Config_::kHasSlidingWindow;
  static constexpr bool kHasDropout = Config_::kHasDropout;
  static constexpr bool kHighPrecision = Config_::kHighPrecision;

  static constexpr int kTileM = Tile::kTileM;
  static constexpr int kTileN = Tile::kTileN;
  static constexpr int kTileK = Tile::kTileK;

  static constexpr int8_t DO_NOT_MASK_TILE = 0;
  static constexpr int8_t MASK_ALL_TILE = 1;
  static constexpr int8_t MASK_HALF_TILE = 2;

  MLLM_FORCE_INLINE void run(const FlashAttention2WorkSpace<Config_>& workspace, const ElementCompute* __restrict__ Q,
                             const ElementCompute* __restrict__ K, const ElementCompute* __restrict__ V,
                             ElementCompute* __restrict__ O, const int32_t batch_size, const int32_t q_head_size,
                             const int32_t kv_head_size, const int32_t seq_size_q, const int32_t seq_size_k,
                             const int32_t dim_size, const int32_t thread_count) {
    static_assert(kTileM == kTileN, "M and N must be equal");
    static_assert(kTileM % 4 == 0);
    assert(q_head_size % thread_count == 0);
    assert(dim_size % 8 == 0);

    acc_s_cast_ = workspace.acc_s_cast;        // [threads][kTileM * kTileN]
    acc_o_ = workspace.acc_o;                  // [threads][kTileM * dim_size]
    acc_s_ = workspace.acc_s;                  // [threads][kTileM * kTileN]
    logsum_ = workspace.logsum;                // [threads][kTileM]
    scoremax_ = workspace.scoremax;            // [threads][kTileM]
    scoremax_prev_ = workspace.scoremax_prev;  // [threads][kTileM]
    score_scale_ = workspace.score_scale;      // [threads][kTileM]
    score_sum_ = workspace.score_sum;          // [threads][kTileM]

    // Prefill and Append mode
    if (seq_size_q != 1) {
      __fa2_prefill_append(Q, K, V, O, batch_size, q_head_size, kv_head_size, seq_size_q, seq_size_k, dim_size, thread_count);
    } else {
      __fa2_decode(Q, K, V, O, batch_size, q_head_size, kv_head_size, seq_size_q, seq_size_k, dim_size, thread_count);
    }
  }

 private:
  MLLM_FORCE_INLINE int32_t kv_head_index(int32_t q_head_index) { return q_head_index / head_repeat_times_; }

  MLLM_FORCE_INLINE int8_t mask_behavior(const int32_t t_r_idx, const int32_t t_c_idx, const int32_t seq_size_q,
                                         const int32_t seq_size_k) {
    const int32_t global_r_start = t_r_idx * kTileM;
    const int32_t global_r_end = global_r_start + kTileM < seq_size_q ? global_r_start + kTileM : seq_size_q;
    const int32_t global_c_start = t_c_idx * kTileN;
    const int32_t global_c_end = global_c_start + kTileN < seq_size_k ? global_c_start + kTileN : seq_size_k;
    int delta_pos = seq_size_k - seq_size_q;

    if (global_c_start - delta_pos > (global_r_end - 1)) return MASK_ALL_TILE;
    if (global_r_end == global_c_end - delta_pos) return MASK_HALF_TILE;
    return DO_NOT_MASK_TILE;
  }

  MLLM_FORCE_INLINE void init_temp(ElementAccumulator* logsum, ElementAccumulator* scoremax, ElementAccumulator* acc_o,
                                   const int32_t dim_size) {
    InitTemporary<Arch, Tile, Numeric, Memory>::run(logsum, scoremax, acc_o, dim_size);
  }

  MLLM_FORCE_INLINE void mma0(const ElementCompute* __restrict__ q_block, const ElementCompute* __restrict__ k_block,
                              ElementAccumulator* __restrict__ acc_s, const int32_t dim_size, const int32_t q_stride_size,
                              const int32_t kv_stride_size, int8_t causal_behavior) {
    MMA0<Arch, Tile, Numeric, Mma0Layout, Memory>::run(q_block, k_block, acc_s, dim_size, q_stride_size, kv_stride_size);

    if (causal_behavior == MASK_HALF_TILE) {
      for (int i = 0; i < kTileM; ++i) {
        for (int j = 0; j < kTileN; ++j) {
          if (j > i) { acc_s[i * kTileN + j] = FA2_FLOAT_NEG_INF; }
        }
      }
    }
  }

  // === Prefill and Append mode. Tile is Br_n_fixed and Bc_n_fixed(Original kTileM x kTileN).
  // q_block  :Br_n_fixed x dim_size
  // k_block  :Bc_n_fixed x dim_size
  // acc_s    :Br_n_fixed x Bc_n_fixed
  MLLM_FORCE_INLINE void mma0_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                         const ElementCompute* __restrict__ q_block, const ElementCompute* __restrict__ k_block,
                                         ElementAccumulator* __restrict__ acc_s, const int32_t dim_size,
                                         const int32_t q_stride_size, const int32_t kv_stride_size, int8_t causal_behavior) {
    MMA0Tail<Arch, Tile, Numeric, Mma0Layout, Memory>::run(q_block, k_block, acc_s, dim_size, q_stride_size, kv_stride_size,
                                                           Br_n_fixed, Bc_n_fixed);

    if (causal_behavior == MASK_HALF_TILE) {
      for (int i = 0; i < Br_n_fixed; ++i) {
        for (int j = 0; j < Bc_n_fixed; ++j) {
          if (j > i) { acc_s[i * Bc_n_fixed + j] = FA2_FLOAT_NEG_INF; }
        }
      }
    }
  }

  MLLM_FORCE_INLINE void mma1(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                              ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size,
                              int8_t causal_behavior) {
    MMA1<Arch, Tile, Numeric, Mma1Layout, Memory>::run(w_block, v_block, acc_o, head_size, dim_size);
  }

  // w_block is Br_n_fixed x Bc_n_fixed
  // v_block is Bc_n_fixed x dim_size
  MLLM_FORCE_INLINE void mma1_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                         const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                         ElementAccumulator* __restrict__ acc_o, const int32_t head_size,
                                         const int32_t dim_size) {
    MMA1Tail<Arch, Tile, Numeric, Mma0Layout, Memory>::run(w_block, v_block, acc_o, head_size, dim_size, Br_n_fixed,
                                                           Bc_n_fixed);
  }

  MLLM_FORCE_INLINE void softmax(const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast,
                                 ElementAccumulator* scoremax, ElementAccumulator* scoremax_prev,
                                 ElementAccumulator* score_scale, ElementAccumulator* score_sum, ElementAccumulator* logsum,
                                 int8_t causal_behavior) {
    Softmax<Arch, Tile, Numeric, Memory, kHighPrecision>::run(acc_s, acc_s_cast, scoremax, scoremax_prev, score_scale,
                                                              score_sum, logsum, scale_);
  }

  MLLM_FORCE_INLINE void softmax_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                            const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast,
                                            ElementAccumulator* scoremax, ElementAccumulator* scoremax_prev,
                                            ElementAccumulator* score_scale, ElementAccumulator* score_sum,
                                            ElementAccumulator* logsum) {
    Softmax<Arch, Tile, Numeric, Memory, kHighPrecision>::run_tail(Br_n_fixed, Bc_n_fixed, acc_s, acc_s_cast, scoremax,
                                                                   scoremax_prev, score_scale, score_sum, logsum, scale_);
  }

  MLLM_FORCE_INLINE void rescale(ElementAccumulator* __restrict__ acc_o, ElementAccumulator* __restrict__ score_scale,
                                 const int32_t dim_size, int8_t causal_behavior) {
    Rescale<Arch, Tile, Numeric, Memory>::run(acc_o, score_scale, dim_size);
  }

  MLLM_FORCE_INLINE void rescale_pa_n_fixed(const int32_t Br_n_fixed, const int32_t Bc_n_fixed,
                                            ElementAccumulator* __restrict__ acc_o,
                                            ElementAccumulator* __restrict__ score_scale, const int32_t dim_size) {
    Rescale<Arch, Tile, Numeric, Memory>::run_tail(Br_n_fixed, Bc_n_fixed, acc_o, score_scale, dim_size);
  }

  MLLM_FORCE_INLINE void scale_and_cast_copy(const ElementAccumulator* __restrict__ acc_o,
                                             const ElementAccumulator* __restrict__ logsum,
                                             ElementCompute* __restrict__ o_block, const int32_t head_size,
                                             const int32_t dim_size) {
    ScaleCastCopy<Arch, Tile, Numeric, Memory>::run(acc_o, logsum, o_block, head_size, dim_size);
  }

  MLLM_FORCE_INLINE void scale_and_cast_copy_pa_n_fixed(const int32_t Br_n_fixed, const ElementAccumulator* __restrict__ acc_o,
                                                        const ElementAccumulator* __restrict__ logsum,
                                                        ElementCompute* __restrict__ o_block, const int32_t t_r_idx,
                                                        const int32_t head_size, const int32_t dim_size) {
    ScaleCastCopy<Arch, Tile, Numeric, Memory>::run_tail(Br_n_fixed, -1, acc_o, logsum, o_block, head_size, dim_size);
  }

  MLLM_FORCE_INLINE
  void __fa2_prefill_append(const ElementCompute* __restrict__ Q, const ElementCompute* __restrict__ K,
                            const ElementCompute* __restrict__ V, ElementCompute* __restrict__ O, const int32_t batch_size,
                            const int32_t q_head_size, const int32_t kv_head_size, const int32_t seq_size_q,
                            const int32_t seq_size_k, const int32_t dim_size, const int32_t thread_count) {
    const int32_t Tr = seq_size_q / kTileM;
    const int32_t Tr_left = seq_size_q % kTileM;
    const int32_t Tc = seq_size_k / kTileN;
    const int32_t Tc_left = seq_size_k % kTileN;

    head_repeat_times_ = q_head_size / kv_head_size;
    scale_ = sqrt(1.0 / dim_size);

    // Loops
    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      // FIXME Parallel Here
      for (int32_t h_idx = 0; h_idx < q_head_size; ++h_idx) {
        // FIXME: Parallel Here
        const int32_t thread_id = 0;
        const int32_t this_thread_head = h_idx;
        const int32_t this_thread_head_q = this_thread_head;
        const int32_t this_thread_head_kv = kv_head_index(this_thread_head_q);

        // Loop S_Q
        for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
          // Init all temps
          init_temp(logsum_ + thread_id * kTileM, scoremax_ + thread_id * kTileM, acc_o_ + thread_id * kTileM * dim_size,
                    dim_size);

          // Loop S_KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const ElementCompute* tile_q = Q + b_idx * seq_size_q * q_head_size * dim_size
                                           + t_r_idx * kTileM * q_head_size * dim_size + this_thread_head_q * dim_size;
            const ElementCompute* tile_k = K + b_idx * seq_size_k * kv_head_size * dim_size
                                           + t_c_idx * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            const ElementCompute* tile_v = V + b_idx * seq_size_k * kv_head_size * dim_size
                                           + t_c_idx * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            ElementAccumulator* tile_acc_s = acc_s_ + thread_id * kTileM * kTileN;
            ElementAccumulator* acc_o = acc_o_ + thread_id * kTileM * dim_size;

            // Get the mask behavior
            // 0. do not mask
            // 1. mask half
            // 2. mask all
            int8_t causal_behavior = DO_NOT_MASK_TILE;
            if constexpr (kHasCausalMask) { causal_behavior = mask_behavior(t_r_idx, t_c_idx, seq_size_q, seq_size_k); }

            if (causal_behavior != MASK_ALL_TILE) {
              // Q @ K^T
              mma0(tile_q, tile_k, tile_acc_s, dim_size, q_head_size * dim_size, kv_head_size * dim_size, causal_behavior);

              // Softmax
              softmax(acc_s_ + thread_id * kTileM * kTileN, acc_s_cast_ + thread_id * kTileM * kTileN,
                      scoremax_ + thread_id * kTileM, scoremax_prev_ + thread_id * kTileM, score_scale_ + thread_id * kTileM,
                      score_sum_ + thread_id * kTileM, logsum_ + thread_id * kTileM, causal_behavior);

              // Rescale
              rescale(acc_o, score_scale_ + thread_id * kTileM, dim_size, causal_behavior);

              // W @ V
              mma1(acc_s_cast_ + thread_id * kTileM * kTileN, tile_v, acc_o, kv_head_size, dim_size, causal_behavior);
            }
          }

          // Process the last block of KV
          if (Tc_left) {
            const ElementCompute* tile_q = Q + b_idx * seq_size_q * q_head_size * dim_size
                                           + t_r_idx * kTileM * q_head_size * dim_size + this_thread_head_q * dim_size;
            const ElementCompute* tile_k = K + b_idx * seq_size_k * kv_head_size * dim_size
                                           + Tc * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            const ElementCompute* tile_v = V + b_idx * seq_size_k * kv_head_size * dim_size
                                           + Tc * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            ElementAccumulator* tile_acc_s = acc_s_ + thread_id * kTileM * kTileN;
            ElementAccumulator* acc_o = acc_o_ + thread_id * kTileM * dim_size;

            // Get the mask behavior
            // 0. do not mask
            // 1. mask half
            // 2. mask all
            int8_t causal_behavior = DO_NOT_MASK_TILE;
            if constexpr (kHasCausalMask) { causal_behavior = mask_behavior(t_r_idx, Tc, seq_size_q, seq_size_k); }

            if (causal_behavior != MASK_ALL_TILE) {
              // Q @ K^T
              mma0_pa_n_fixed(kTileM, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, q_head_size * dim_size,
                              kv_head_size * dim_size, causal_behavior);

              // Softmax
              softmax_pa_n_fixed(kTileM, Tc_left, acc_s_ + thread_id * kTileM * kTileN,
                                 acc_s_cast_ + thread_id * kTileM * kTileN, scoremax_ + thread_id * kTileM,
                                 scoremax_prev_ + thread_id * kTileM, score_scale_ + thread_id * kTileM,
                                 score_sum_ + thread_id * kTileM, logsum_ + thread_id * kTileM);

              // Rescale
              rescale_pa_n_fixed(kTileM, Tc_left, acc_o, score_scale_ + thread_id * kTileM, dim_size);

              // W @ V
              mma1_pa_n_fixed(kTileM, Tc_left, acc_s_cast_ + thread_id * kTileM * kTileN, tile_v, acc_o, kv_head_size,
                              dim_size);
            }
          }

          // Scale acc_o and cast store
          scale_and_cast_copy(acc_o_ + thread_id * kTileM * dim_size, logsum_ + thread_id * kTileM,
                              O + b_idx * seq_size_q * q_head_size * dim_size + t_r_idx * kTileM * q_head_size * dim_size
                                  + this_thread_head * dim_size,
                              q_head_size, dim_size);
        }

        // Process left Q
        if (Tr_left) {
          // Init all temps
          init_temp(logsum_ + thread_id * kTileM, scoremax_ + thread_id * kTileM, acc_o_ + thread_id * kTileM * dim_size,
                    dim_size);

          // Loop S_KV
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const ElementCompute* tile_q = Q + b_idx * seq_size_q * q_head_size * dim_size
                                           + Tr * kTileM * q_head_size * dim_size + this_thread_head_q * dim_size;
            const ElementCompute* tile_k = K + b_idx * seq_size_k * kv_head_size * dim_size
                                           + t_c_idx * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            const ElementCompute* tile_v = V + b_idx * seq_size_k * kv_head_size * dim_size
                                           + t_c_idx * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            ElementAccumulator* tile_acc_s = acc_s_ + thread_id * kTileM * kTileN;
            ElementAccumulator* acc_o = acc_o_ + thread_id * kTileM * dim_size;

            // Get the mask behavior
            // 0. do not mask
            // 1. mask half
            // 2. mask all
            int8_t causal_behavior = DO_NOT_MASK_TILE;
            if constexpr (kHasCausalMask) { causal_behavior = mask_behavior(Tr, t_c_idx, seq_size_q, seq_size_k); }

            if (causal_behavior != MASK_ALL_TILE) {
              // Q @ K^T
              mma0_pa_n_fixed(Tr_left, kTileN, tile_q, tile_k, tile_acc_s, dim_size, q_head_size * dim_size,
                              kv_head_size * dim_size, causal_behavior);

              // Softmax
              softmax_pa_n_fixed(Tr_left, kTileN, acc_s_ + thread_id * kTileM * kTileN,
                                 acc_s_cast_ + thread_id * kTileM * kTileN, scoremax_ + thread_id * kTileM,
                                 scoremax_prev_ + thread_id * kTileM, score_scale_ + thread_id * kTileM,
                                 score_sum_ + thread_id * kTileM, logsum_ + thread_id * kTileM);

              // Rescale
              rescale_pa_n_fixed(Tr_left, kTileN, acc_o, score_scale_ + thread_id * kTileM, dim_size);

              // W @ V
              mma1_pa_n_fixed(Tr_left, kTileN, acc_s_cast_ + thread_id * kTileM * kTileN, tile_v, acc_o, kv_head_size,
                              dim_size);
            }
          }

          // Process the last block of KV
          if (Tc_left) {
            const ElementCompute* tile_q = Q + b_idx * seq_size_q * q_head_size * dim_size
                                           + Tr * kTileM * q_head_size * dim_size + this_thread_head_q * dim_size;
            const ElementCompute* tile_k = K + b_idx * seq_size_k * kv_head_size * dim_size
                                           + Tc * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            const ElementCompute* tile_v = V + b_idx * seq_size_k * kv_head_size * dim_size
                                           + Tc * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            ElementAccumulator* tile_acc_s = acc_s_ + thread_id * kTileM * kTileN;
            ElementAccumulator* acc_o = acc_o_ + thread_id * kTileM * dim_size;

            // Get the mask behavior
            // 0. do not mask
            // 1. mask half
            // 2. mask all
            int8_t causal_behavior = DO_NOT_MASK_TILE;
            if constexpr (kHasCausalMask) { causal_behavior = mask_behavior(Tr, Tc, seq_size_q, seq_size_k); }

            if (causal_behavior != MASK_ALL_TILE) {
              // Q @ K^T
              mma0_pa_n_fixed(Tr_left, Tc_left, tile_q, tile_k, tile_acc_s, dim_size, q_head_size * dim_size,
                              kv_head_size * dim_size, causal_behavior);

              // Softmax
              softmax_pa_n_fixed(Tr_left, Tc_left, acc_s_ + thread_id * kTileM * kTileN,
                                 acc_s_cast_ + thread_id * kTileM * kTileN, scoremax_ + thread_id * kTileM,
                                 scoremax_prev_ + thread_id * kTileM, score_scale_ + thread_id * kTileM,
                                 score_sum_ + thread_id * kTileM, logsum_ + thread_id * kTileM);

              // Rescale
              rescale_pa_n_fixed(Tr_left, Tc_left, acc_o, score_scale_ + thread_id * kTileM, dim_size);

              // W @ V
              mma1_pa_n_fixed(Tr_left, Tc_left, acc_s_cast_ + thread_id * kTileM * kTileN, tile_v, acc_o, kv_head_size,
                              dim_size);
            }
          }

          // Scale acc_o and cast store
          scale_and_cast_copy_pa_n_fixed(Tr_left, acc_o_ + thread_id * kTileM * dim_size, logsum_ + thread_id * kTileM,
                                         O + b_idx * seq_size_q * q_head_size * dim_size + Tr * kTileM * q_head_size * dim_size
                                             + this_thread_head * dim_size,
                                         Tr, q_head_size, dim_size);
        }
      }
    }
  }

  // q_block  :1 x dim_size
  // k_block  :kTileN x dim_size
  // acc_s    :1 x kTileN(Still use kTileM x kTileN memory space, but others keeps empty)
  MLLM_FORCE_INLINE void mma0_d(const ElementCompute* __restrict__ q_block, const ElementCompute* __restrict__ k_block,
                                ElementAccumulator* __restrict__ acc_s, const int32_t dim_size, const int32_t q_stride_size,
                                const int32_t kv_stride_size) {
    MMA0Tail<Arch, Tile, Numeric, Mma0Layout, Memory>::run(q_block, k_block, acc_s, dim_size, q_stride_size, kv_stride_size, 1,
                                                           kTileN);
  }

  MLLM_FORCE_INLINE void mma0_d_n_fixed(const int32_t Bc_n_fixed, const ElementCompute* __restrict__ q_block,
                                        const ElementCompute* __restrict__ k_block, ElementAccumulator* __restrict__ acc_s,
                                        const int32_t dim_size, const int32_t q_stride_size, const int32_t kv_stride_size) {
    MMA0Tail<Arch, Tile, Numeric, Mma0Layout, Memory>::run(q_block, k_block, acc_s, dim_size, q_stride_size, kv_stride_size, 1,
                                                           Bc_n_fixed);
  }

  MLLM_FORCE_INLINE void mma1_d(const ElementCompute* __restrict__ w_block, const ElementCompute* __restrict__ v_block,
                                ElementAccumulator* __restrict__ acc_o, const int32_t head_size, const int32_t dim_size) {
    MMA1Tail<Arch, Tile, Numeric, Mma0Layout, Memory>::run(w_block, v_block, acc_o, head_size, dim_size, 1, kTileN);
  }

  MLLM_FORCE_INLINE void mma1_d_n_fixed(const int32_t Bc_n_fixed, const ElementCompute* __restrict__ w_block,
                                        const ElementCompute* __restrict__ v_block, ElementAccumulator* __restrict__ acc_o,
                                        const int32_t head_size, const int32_t dim_size) {
    MMA1Tail<Arch, Tile, Numeric, Mma0Layout, Memory>::run(w_block, v_block, acc_o, head_size, dim_size, 1, Bc_n_fixed);
  }

  MLLM_FORCE_INLINE
  void softmax_d(const ElementAccumulator* __restrict__ acc_s, ElementCompute* acc_s_cast, ElementAccumulator* scoremax,
                 ElementAccumulator* scoremax_prev, ElementAccumulator* score_scale, ElementAccumulator* score_sum,
                 ElementAccumulator* logsum) {
    Softmax<Arch, Tile, Numeric, Memory, kHighPrecision>::run_tail(1, kTileN, acc_s, acc_s_cast, scoremax, scoremax_prev,
                                                                   score_scale, score_sum, logsum, scale_);
  }

  MLLM_FORCE_INLINE void softmax_d_n_fixed(const int32_t Bc_n_fixed, const ElementAccumulator* __restrict__ acc_s,
                                           ElementCompute* acc_s_cast, ElementAccumulator* scoremax,
                                           ElementAccumulator* scoremax_prev, ElementAccumulator* score_scale,
                                           ElementAccumulator* score_sum, ElementAccumulator* logsum) {
    Softmax<Arch, Tile, Numeric, Memory, kHighPrecision>::run_tail(1, Bc_n_fixed, acc_s, acc_s_cast, scoremax, scoremax_prev,
                                                                   score_scale, score_sum, logsum, scale_);
  }

  MLLM_FORCE_INLINE void rescale_d(ElementAccumulator* __restrict__ acc_o, ElementAccumulator* __restrict__ score_scale,
                                   const int32_t dim_size) {
    Rescale<Arch, Tile, Numeric, Memory>::run_tail(1, kTileN, acc_o, score_scale, dim_size);
  }

  MLLM_FORCE_INLINE void rescale_d_n_fixed(const int32_t Bc_n_fixed, ElementAccumulator* __restrict__ acc_o,
                                           ElementAccumulator* __restrict__ score_scale, const int32_t dim_size) {
    Rescale<Arch, Tile, Numeric, Memory>::run_tail(1, Bc_n_fixed, acc_o, score_scale, dim_size);
  }

  MLLM_FORCE_INLINE void scale_and_cast_copy_d(const ElementAccumulator* __restrict__ acc_o,
                                               const ElementAccumulator* __restrict__ logsum,
                                               ElementCompute* __restrict__ o_block, const int32_t t_r_idx,
                                               const int32_t head_size, const int32_t dim_size) {
    ScaleCastCopy<Arch, Tile, Numeric, Memory>::run_tail(1, -1, acc_o, logsum, o_block, head_size, dim_size);
  }

  // === Decode mode. Tile is always 1 and kTileN.
  MLLM_FORCE_INLINE void init_temp_d(ElementAccumulator* logsum, ElementAccumulator* scoremax, ElementAccumulator* acc_o,
                                     const int32_t dim_size) {
    InitTemporary<Arch, Tile, Numeric, Memory>::run_decode(logsum, scoremax, acc_o, dim_size);
  }

  MLLM_FORCE_INLINE
  void __fa2_decode(const ElementCompute* __restrict__ Q, const ElementCompute* __restrict__ K,
                    const ElementCompute* __restrict__ V, ElementCompute* __restrict__ O, const int32_t batch_size,
                    const int32_t q_head_size, const int32_t kv_head_size, const int32_t seq_size_q, const int32_t seq_size_k,
                    const int32_t dim_size, const int32_t thread_count) {
    const int32_t Tr = 1;
    const int32_t Tc = seq_size_k / kTileN;
    const int32_t Tc_left = seq_size_k % kTileN;
    scale_ = sqrt(1.0 / dim_size);
    head_repeat_times_ = q_head_size / kv_head_size;

    for (int32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      // FIXME: Parallel here
      for (int32_t h_idx = 0; h_idx < q_head_size; ++h_idx) {
        // FIXME: Parallel here
        const int32_t thread_id = 0;
        const int32_t this_thread_head = h_idx;
        const int32_t this_thread_head_q = this_thread_head;
        const int32_t this_thread_head_kv = kv_head_index(this_thread_head_q);

        // Loop Q
        for (int t_r_idx = 0; t_r_idx < Tr; ++t_r_idx) {
          // Init all temps
          init_temp_d(logsum_ + thread_id * kTileM, scoremax_ + thread_id * kTileM, acc_o_ + thread_id * kTileM * dim_size,
                      dim_size);
          for (int t_c_idx = 0; t_c_idx < Tc; ++t_c_idx) {
            const ElementCompute* tile_q = Q + b_idx * seq_size_q * q_head_size * dim_size
                                           + t_r_idx * 1 * q_head_size * dim_size + this_thread_head_q * dim_size;
            const ElementCompute* tile_k = K + b_idx * seq_size_k * kv_head_size * dim_size
                                           + t_c_idx * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            const ElementCompute* tile_v = V + b_idx * seq_size_k * kv_head_size * dim_size
                                           + t_c_idx * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            ElementAccumulator* tile_acc_s = acc_s_ + thread_id * 1 * kTileN;
            ElementAccumulator* acc_o = acc_o_ + thread_id * kTileM * dim_size;

            // Q @ K^T
            mma0_d(tile_q, tile_k, tile_acc_s, dim_size, q_head_size * dim_size, kv_head_size * dim_size);

            // Softmax
            softmax_d(acc_s_ + thread_id * kTileM * kTileN, acc_s_cast_ + thread_id * kTileM * kTileN,
                      scoremax_ + thread_id * kTileM, scoremax_prev_ + thread_id * kTileM, score_scale_ + thread_id * kTileM,
                      score_sum_ + thread_id * kTileM, logsum_ + thread_id * kTileM);

            // Rescale
            rescale_d(acc_o, score_scale_ + thread_id * kTileM, dim_size);

            // W @ V
            mma1_d(acc_s_cast_ + thread_id * kTileM * kTileN, tile_v, acc_o, kv_head_size, dim_size);
          }

          if (Tc_left) {
            const ElementCompute* tile_q = Q + b_idx * seq_size_q * q_head_size * dim_size
                                           + t_r_idx * kTileM * q_head_size * dim_size + this_thread_head_q * dim_size;
            const ElementCompute* tile_k = K + b_idx * seq_size_k * kv_head_size * dim_size
                                           + Tc * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            const ElementCompute* tile_v = V + b_idx * seq_size_k * kv_head_size * dim_size
                                           + Tc * kTileN * kv_head_size * dim_size + this_thread_head_kv * dim_size;
            ElementAccumulator* tile_acc_s = acc_s_ + thread_id * kTileM * kTileN;
            ElementAccumulator* acc_o = acc_o_ + thread_id * kTileM * dim_size;

            // Q @ K^T
            mma0_d_n_fixed(Tc_left, tile_q, tile_k, tile_acc_s, dim_size, q_head_size * dim_size, kv_head_size * dim_size);

            // Softmax
            softmax_d_n_fixed(Tc_left, acc_s_ + thread_id * kTileM * kTileN, acc_s_cast_ + thread_id * kTileM * kTileN,
                              scoremax_ + thread_id * kTileM, scoremax_prev_ + thread_id * kTileM,
                              score_scale_ + thread_id * kTileM, score_sum_ + thread_id * kTileM, logsum_ + thread_id * kTileM);

            // Rescale
            rescale_d_n_fixed(Tc_left, acc_o, score_scale_ + thread_id * kTileM, dim_size);

            // W @ V
            mma1_d_n_fixed(Tc_left, acc_s_cast_ + thread_id * kTileM * kTileN, tile_v, acc_o, kv_head_size, dim_size);
          }

          // Scale acc_o and cast store
          scale_and_cast_copy_d(acc_o_ + thread_id * kTileM * dim_size, logsum_ + thread_id * kTileM,
                                O + b_idx * seq_size_q * q_head_size * dim_size + t_r_idx * 1 * q_head_size * dim_size
                                    + this_thread_head * dim_size,
                                t_r_idx, q_head_size, dim_size);
        }
      }
    }
  }

  int32_t head_repeat_times_;
  float scale_;
  ElementCompute* acc_s_cast_;         // [threads][kTileM * kTileN]
  ElementAccumulator* acc_o_;          // [threads][kTileM * dim_size]
  ElementAccumulator* acc_s_;          // [threads][kTileM * kTileN]
  ElementAccumulator* logsum_;         // [threads][kTileM]
  ElementAccumulator* scoremax_;       // [threads][kTileM]
  ElementAccumulator* scoremax_prev_;  // [threads][kTileM]
  ElementAccumulator* score_scale_;    // [threads][kTileM]
  ElementAccumulator* score_sum_;      // [threads][kTileM]
};

}  // namespace mllm::cpu::fa2
