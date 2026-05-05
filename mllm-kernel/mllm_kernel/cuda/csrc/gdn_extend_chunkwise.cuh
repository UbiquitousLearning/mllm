// Chunkwise-parallel fused GDN extend using WY representation.
//
// Reference: Gated Delta Networks (Yang et al., 2024) — "Chunkwise Parallel
// DeltaNet" approach.  For each chunk of CHUNK_C tokens the sequential
// recurrence is decomposed into:
//
//   WY form:  S_T = γ_C · (S_0 (I − W K^T) + Ũ K^T)
//
// where W (K-space) and Ũ (V-space) are built by a sequential O(C) scan
// inside the kernel using shared memory, and the output / state-update
// become parallel matmul-like operations that avoid per-token global state
// traffic.
//
// ──────────────────────────────────────────────────────────────────────────
// Optimisation log (relative to original single-block-reduce implementation)
// ──────────────────────────────────────────────────────────────────────────
// [OPT-1] Batched WY inner loop (Phase 1)
//   Original: 2×t block_reduce_sum calls per token t, each requiring 2
//   __syncthreads → O(C²) syncs per chunk (~1984 for C=32, BLOCK_K=128).
//   New: one batched warp-then-cross-warp reduction per token → 2 syncs
//   regardless of t, giving O(C) syncs (~64 for C=32).
//   Mechanism: all warps write partial dot-products for ALL i<t into
//   batch_wbuf/batch_kbuf (smem), then warp-0 threads fan-in in parallel
//   across the CHUNK_C indices after a single __syncthreads.
//
// [OPT-2] Batched KQ inner loop (Phase 2)
//   Original: (r+1) block_reduce_sum calls per output position r → O(C²)
//   syncs (~1248 for C=32).
//   New: same batched scheme (reuses batch_wbuf / pw_smem from Phase 1) →
//   2 syncs per output position, O(C) total (~64 for C=32).
//
// [OPT-3] Numerical guard for γ underflow
//   When gam_t underflows to 0 (aggressive decay + long chunks), β/γ would
//   be ±inf.  Since γ_r ≤ γ_t when r≥t, the output contribution
//   γ_r · Ũ[t] is also zero in that regime; we set U_smem[t]=0 explicitly.
//
// [OPT-4 REVERTED — analysis]
//   CHUNK_C=64 on SM80+ was intended to halve chunk iterations, but
//   Phase 1 and Phase 2 both contain O(C²) inner loops (warp_reduce_sum
//   over i=0..t_loc-1 / 0..r).  Doubling C quadruples per-chunk O(C²)
//   work while only halving chunk count → net 2× MORE arithmetic.
//   Worse: 83.9 KB smem (BLOCK_K=128, CHUNK_C=64) limits occupancy to
//   1 block/SM on SM87 (164 KB/SM) versus 3 blocks/SM for 42.5 KB
//   (CHUNK_C=32).  Combined effect: ~3-4× slower in practice.
//   CHUNK_C=32 fits comfortably in the default 48 KB smem limit for
//   all supported BLOCK_K values (max 42.5 KB at BLOCK_K=128), so no
//   cudaFuncSetAttribute or runtime SM detection is required.
//
// Sync budget per chunk (BLOCK_K=128, CHUNK_C=32):
//   Phase 1 per token: k_norm(2) + syncA(1) + batchedWY syncB+C(2)
//                    + syncD(1) + block_reduce_bv(2) + syncE(1) = 9
//   Phase 2 per output: q_norm(2) + syncA(1) + block_reduce_bv(2)
//                     + batchedKQ syncB+C(2) + syncD(1) = 8
//   Total per chunk C=32: 9×32 + 8×32 = 544  (was ~3400 before OPT-1/2)
//
// Works on all SM versions (SM70+).
//
// Grid : (NV, batch_size * HV)   NV = ceil(V / BV)
// Block: BLOCK_K threads         BLOCK_K = round_up_32(K), max 256
//
// Shared memory (BLOCK_K=64,  CHUNK_C=32, BV=32): ≈ 21.5 KB   (well within 48 KB)
// Shared memory (BLOCK_K=128, CHUNK_C=32, BV=32): ≈ 42.5 KB   (well within 48 KB)
// Shared memory (BLOCK_K=256, CHUNK_C=16, BV=32): ≈ 39.4 KB   (well within 48 KB)
//
// Algorithm per chunk [cs, cs+C):
//
//   Phase 1  WY construction  — sequential O(C) steps:
//     t=0..C-1:
//       α_t = exp(-exp(A_log)·softplus(a_t+dt_bias)),  β_t = sigmoid(b_t)
//       γ_t = ∏_{j≤t} α_j
//       k̂_t  = L2norm(k_t)        → K_smem[t, :]
//       bk_t = β_t · k̂_t
//       pw[i] = <W_smem[i,:], bk_t>  for i < t   (batched warp-reduce)
//       pk[i] = <K_smem[i,:], bk_t>  for i < t   (batched warp-reduce)
//       W_smem[t,k] = bk_t[k] − Σ_i K_smem[i,k]·pw[i]
//       SW_smem[t,v] = <S_0[v,:], W_smem[t,:]>         (block_reduce_bv)
//       U_smem[t,v] = (β_t/γ_t)·v_t[v] − Σ_i U_smem[i,v]·pk[i]
//                     (guarded: 0 when γ_t underflows)
//
//   Phase 2  Output  — for each r = 0..C-1:
//     q̂_r  = L2norm(q_r)/√K
//     SQ[v] = <S_0[v,:], q̂_r>                          (block_reduce_bv)
//     KQ[i] = <K_smem[i,:], q̂_r>   for i ≤ r           (batched warp-reduce)
//     Δ_i[v] = U_smem[i,v] − SW_smem[i,v]
//     O[v,r] = γ_r · (SQ[v] + Σ_{i≤r} Δ_i[v]·KQ[i])
//
//   Phase 3  State update  — parallel over K (no block reduce):
//     S_new[v,k] = γ_C · (S_0[v,k] + Σ_i Δ_i[v]·K_smem[i,k])

#pragma once

#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace GDNExtendChunkwiseKernel {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr int BV = 32;  // V-tile size — matches gdn_extend / gdn_decode

// ---------------------------------------------------------------------------
// Warp-level scalar reduction
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }

// ---------------------------------------------------------------------------
// Block-level scalar reduction → broadcast to all threads via smem[0]
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int wid  = threadIdx.x / 32;
    const int lid  = threadIdx.x % 32;
    const int nw   = blockDim.x / 32;
    val = warp_reduce_sum(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < nw) ? smem[lid] : 0.f;
        v = warp_reduce_sum(v);
        if (lid == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// Block-level BV-vector reduction → broadcast via smem broadcast_buf[BV]
// ---------------------------------------------------------------------------

__device__ __forceinline__ void block_reduce_bv(
    float        partial[BV],
    float* __restrict__ reduce_buf,    // [num_warps * BV]
    float* __restrict__ broadcast_buf, // [BV]
    float        out[BV]
) {
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x % 32;
    const int nw  = blockDim.x / 32;

#pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        float v = warp_reduce_sum(partial[bv]);
        if (lid == 0) reduce_buf[wid * BV + bv] = v;
    }
    __syncthreads();

    if (wid == 0) {
#pragma unroll
        for (int bv = 0; bv < BV; bv++) {
            float v = (lid < nw) ? reduce_buf[lid * BV + bv] : 0.f;
            v = warp_reduce_sum(v);
            if (lid == 0) broadcast_buf[bv] = v;
        }
    }
    __syncthreads();

#pragma unroll
    for (int bv = 0; bv < BV; bv++) out[bv] = broadcast_buf[bv];
}

// ---------------------------------------------------------------------------
// Main chunkwise kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_K, int CHUNK_C>
__global__ void gdn_extend_chunkwise_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    const T* __restrict__ a_ptr,
    const T* __restrict__ b_ptr,
    const float* __restrict__ A_log_ptr,
    const float* __restrict__ dt_bias_ptr,
    float*       __restrict__ state_pool,
    const int64_t* __restrict__ cache_indices,
    const int64_t* __restrict__ cu_seqlens,
    T*           __restrict__ output_ptr,
    const int batch_size,
    const int H,
    const int HV,
    const int K,
    const int V
) {
    const int bv_blk     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int i_n        = batch_head / HV;
    const int i_hv       = batch_head % HV;
    const int i_h        = i_hv * H / HV;
    const int k_idx      = threadIdx.x;
    const int v_start    = bv_blk * BV;
    const int num_warps  = BLOCK_K / 32;
    const int wid        = k_idx / 32;
    const int lid        = k_idx % 32;

    if (i_n >= batch_size) return;

    // ── Shared memory layout ──────────────────────────────────────────────
    //
    //  sq          [BLOCK_K]              — scratch: current q/bk vector
    //  warp_buf    [num_warps]            — scalar block-reduce workspace
    //                                       (used by block_reduce_sum for norms)
    //  pw_smem     [CHUNK_C]              — [Phase1] reduced pw[i] = <W[i],bk_t>
    //                                       [Phase2] reused as kq[i] = <K[i],q̂_r>
    //  pk_smem     [CHUNK_C]              — [Phase1] reduced pk[i] = <K[i],bk_t>
    //  batch_wbuf  [num_warps × CHUNK_C]  — cross-warp scratch for pw / kq
    //  batch_kbuf  [num_warps × CHUNK_C]  — cross-warp scratch for pk
    //  K_smem      [CHUNK_C × BLOCK_K]    — normalised key per chunk-token
    //  W_smem      [CHUNK_C × BLOCK_K]    — WY W vectors
    //  U_smem      [CHUNK_C × BV]         — WY Ũ vectors (V-space)
    //  SW_smem     [CHUNK_C × BV]         — S₀ @ W_smem (precomputed)
    //  gamma_sm    [CHUNK_C]              — cumulative decay in chunk
    //  rbv         [num_warps × BV]       — block_reduce_bv workspace
    //  bcast       [BV]                   — block_reduce_bv broadcast
    extern __shared__ float smem[];

    int off = 0;
    float* sq         = smem + off; off += BLOCK_K;
    float* warp_buf   = smem + off; off += num_warps;
    float* pw_smem    = smem + off; off += CHUNK_C;
    float* pk_smem    = smem + off; off += CHUNK_C;
    float* batch_wbuf = smem + off; off += num_warps * CHUNK_C;
    float* batch_kbuf = smem + off; off += num_warps * CHUNK_C;
    float* K_smem     = smem + off; off += CHUNK_C * BLOCK_K;
    float* W_smem     = smem + off; off += CHUNK_C * BLOCK_K;
    float* U_smem     = smem + off; off += CHUNK_C * BV;
    float* SW_smem    = smem + off; off += CHUNK_C * BV;
    float* gamma_sm   = smem + off; off += CHUNK_C;
    float* rbv        = smem + off; off += num_warps * BV;
    float* bcast      = smem + off; // [BV]

    const float A_log_val   = A_log_ptr[i_hv];
    const float dt_bias_val = dt_bias_ptr[i_hv];

    const int64_t seq_start = cu_seqlens[i_n];
    const int64_t seq_end   = cu_seqlens[i_n + 1];

    // ── Load initial state into registers ─────────────────────────────────
    const int64_t pool_idx   = cache_indices[i_n];
    const int64_t state_base =
        pool_idx * (int64_t)HV * V * K + i_hv * (int64_t)V * K;

    float state[BV];
#pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = v_start + bv;
        state[bv] = (v_idx < V && k_idx < K)
            ? state_pool[state_base + (int64_t)v_idx * K + k_idx]
            : 0.f;
    }

    // ── Chunk loop ─────────────────────────────────────────────────────────
    for (int64_t cs = seq_start; cs < seq_end; cs += CHUNK_C) {
        const int C      = (int)min((int64_t)CHUNK_C, seq_end - cs);
        float prev_gamma = 1.f;

        // ============================================================
        // Phase 1 — WY construction
        //   Sequential over t_loc = 0..C-1.
        //   Inner dot products pw[i]/pk[i] are computed via one batched
        //   warp→cross-warp reduction (2 __syncthreads total per token)
        //   instead of the original 2×t individual block_reduce_sum calls.
        // ============================================================
        for (int t_loc = 0; t_loc < C; t_loc++) {
            const int64_t t_g = cs + t_loc;

            // ── Gating scalars ────────────────────────────────────────
            const float a_val  = to_float(a_ptr[t_g * HV + i_hv]);
            const float b_val  = to_float(b_ptr[t_g * HV + i_hv]);
            const float x      = a_val + dt_bias_val;
            const float sp_x   = (x <= 20.f) ? logf(1.f + expf(x)) : x;
            const float alpha  = expf(-expf(A_log_val) * sp_x);
            const float beta   = 1.f / (1.f + expf(-b_val));
            const float gam_t  = prev_gamma * alpha;
            prev_gamma = gam_t;
            if (k_idx == 0) gamma_sm[t_loc] = gam_t;

            // ── L2-normalise k → K_smem[t_loc,:];  bk_t → sq ────────
            const float k_raw = (k_idx < K)
                ? to_float(k_ptr[t_g * H * K + i_h * K + k_idx]) : 0.f;
            const float k_sq  = block_reduce_sum(k_raw * k_raw, warp_buf);
            // block_reduce_sum includes 2 __syncthreads internally
            const float k_n   = k_raw * rsqrtf(k_sq + 1e-6f);
            K_smem[t_loc * BLOCK_K + k_idx] = k_n;
            sq[k_idx] = beta * k_n;   // bk_t
            __syncthreads();          // sync A: K_smem[t_loc], sq, gamma_sm ready

            // ── [OPT-1] Batched WY inner loop ────────────────────────
            // Compute pw[i]=<W[i,:],bk_t> and pk[i]=<K[i,:],bk_t> for
            // all i < t_loc with only 2 __syncthreads.
            //
            // Step 1: every thread accumulates warp-level partial sums
            //   for all i simultaneously (no sync needed for warp ops).
            for (int i = 0; i < t_loc; i++) {
                float pw_p = warp_reduce_sum(W_smem[i * BLOCK_K + k_idx] * sq[k_idx]);
                float pk_p = warp_reduce_sum(K_smem[i * BLOCK_K + k_idx] * sq[k_idx]);
                // Lane 0 of each warp records its partial sum.
                if (lid == 0) {
                    batch_wbuf[wid * CHUNK_C + i] = pw_p;
                    batch_kbuf[wid * CHUNK_C + i] = pk_p;
                }
            }
            __syncthreads();   // sync B: warp partials visible to warp 0

            // Step 2: warp-0 threads fan-in across warps, distributing
            //   the CHUNK_C reduction indices across the 32 lanes.
            if (wid == 0) {
                for (int i = lid; i < t_loc; i += 32) {
                    float pw_tot = 0.f, pk_tot = 0.f;
                    for (int w = 0; w < num_warps; w++) {
                        pw_tot += batch_wbuf[w * CHUNK_C + i];
                        pk_tot += batch_kbuf[w * CHUNK_C + i];
                    }
                    pw_smem[i] = pw_tot;
                    pk_smem[i] = pk_tot;
                }
            }
            __syncthreads();   // sync C: pw_smem / pk_smem broadcast

            // Step 3: apply corrections — pure register arithmetic,
            //   no sync needed.
            float w_corr = 0.f;
            float u_corr = 0.f;
            for (int i = 0; i < t_loc; i++) {
                w_corr += K_smem[i * BLOCK_K + k_idx] * pw_smem[i];
                if (k_idx < BV) u_corr += U_smem[i * BV + k_idx] * pk_smem[i];
            }

            // W_smem[t_loc, :] = bk_t − w_corr
            W_smem[t_loc * BLOCK_K + k_idx] = sq[k_idx] - w_corr;

            // Load v for V-owning threads
            float v_val = 0.f;
            if (k_idx < BV) {
                const int v_idx = v_start + k_idx;
                v_val = (v_idx < V)
                    ? to_float(v_ptr[t_g * HV * V + i_hv * V + v_idx]) : 0.f;
            }
            __syncthreads();   // sync D: W_smem[t_loc] visible for block_reduce_bv

            // SW_smem[t_loc, v] = <S₀[v,:], W_smem[t_loc,:]>
            float psw[BV];
#pragma unroll
            for (int bv = 0; bv < BV; bv++)
                psw[bv] = state[bv] * W_smem[t_loc * BLOCK_K + k_idx];
            float sw_r[BV];
            block_reduce_bv(psw, rbv, bcast, sw_r);
            // block_reduce_bv includes 2 __syncthreads internally

            if (k_idx < BV) {
                SW_smem[t_loc * BV + k_idx] = sw_r[k_idx];
                // [OPT-3] Numerical guard: when gam_t underflows to 0.f the
                // contribution β/γ·v would be ±inf, but the corresponding
                // output term γ_r·Ũ[t] = 0 for any r≥t (since γ_r≤γ_t=0).
                // Setting U_smem to 0 is therefore mathematically exact.
                U_smem[t_loc * BV + k_idx] = (gam_t > 0.f)
                    ? (beta / gam_t) * v_val - u_corr
                    : 0.f;
            }
            __syncthreads();   // sync E: SW/U/gamma_sm visible for next t_loc
        }

        // ============================================================
        // Phase 2 — Output for each token r in chunk.
        //   Inner KQ dot products are batched the same way as Phase 1,
        //   reusing batch_wbuf (Phase 1 data no longer needed) and
        //   pw_smem (also safe to overwrite at each r).
        // ============================================================
        for (int r = 0; r < C; r++) {
            const int64_t t_g = cs + r;

            // ── L2-normalise q̂_r/√K → sq ────────────────────────────
            const float q_raw = (k_idx < K)
                ? to_float(q_ptr[t_g * H * K + i_h * K + k_idx]) : 0.f;
            const float q_sq  = block_reduce_sum(q_raw * q_raw, warp_buf);
            // block_reduce_sum: 2 __syncthreads internally
            const float q_n   = q_raw * rsqrtf(q_sq + 1e-6f) * rsqrtf((float)K);
            sq[k_idx] = q_n;
            __syncthreads();   // sync A: sq ready

            // SQ[v] = <S₀[v,:], q̂_r>
            float psq[BV];
#pragma unroll
            for (int bv = 0; bv < BV; bv++) psq[bv] = state[bv] * sq[k_idx];
            float sqs[BV];
            block_reduce_bv(psq, rbv, bcast, sqs);
            // block_reduce_bv: 2 __syncthreads internally

            // ── [OPT-2] Batched KQ inner loop ────────────────────────
            // KQ[i] = <K_smem[i,:], q̂_r> for i = 0..r
            // Reuses batch_wbuf (Phase 1 done) and pw_smem (kq output).
            //
            // Step 1: warp partial sums for all i ≤ r simultaneously.
            for (int i = 0; i <= r; i++) {
                float kq_p = warp_reduce_sum(K_smem[i * BLOCK_K + k_idx] * sq[k_idx]);
                if (lid == 0) batch_wbuf[wid * CHUNK_C + i] = kq_p;
            }
            __syncthreads();   // sync B: warp partials visible

            // Step 2: warp-0 cross-warp fan-in.
            if (wid == 0) {
                for (int i = lid; i <= r; i += 32) {
                    float tot = 0.f;
                    for (int w = 0; w < num_warps; w++)
                        tot += batch_wbuf[w * CHUNK_C + i];
                    pw_smem[i] = tot;   // pw_smem repurposed as kq_smem
                }
            }
            __syncthreads();   // sync C: kq values (pw_smem[0..r]) visible

            // Accumulate Δ_i · KQ_i for i = 0..r  (pure register arithmetic)
            float accum_k = 0.f;
            if (k_idx < BV) {
                for (int i = 0; i <= r; i++)
                    accum_k += (U_smem[i * BV + k_idx] - SW_smem[i * BV + k_idx])
                               * pw_smem[i];
            }

            // Write output
            const float gam_r = gamma_sm[r];
            if (k_idx < BV) {
                const int v_idx = v_start + k_idx;
                if (v_idx < V)
                    output_ptr[t_g * HV * V + i_hv * V + v_idx] =
                        from_float<T>(gam_r * (sqs[k_idx] + accum_k));
            }
            __syncthreads();   // sync D: protect sq/batch_wbuf/pw_smem
        }

        // ============================================================
        // Phase 3 — State update (fully parallel over K, no block reduce)
        //   S_new[v,k] = γ_C · (S₀[v,k] + Σ_i Δ_i[v]·K_smem[i,k])
        // ============================================================
        const float gam_C = gamma_sm[C - 1];
#pragma unroll
        for (int bv = 0; bv < BV; bv++) {
            float ds = 0.f;
            for (int i = 0; i < C; i++) {
                const float delta = U_smem[i * BV + bv] - SW_smem[i * BV + bv];
                ds += delta * K_smem[i * BLOCK_K + k_idx];
            }
            state[bv] = gam_C * (state[bv] + ds);
        }
        __syncthreads();   // smem safe to overwrite in next chunk
    }

    // ── Write final state back to pool ────────────────────────────────────
#pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = v_start + bv;
        if (v_idx < V && k_idx < K)
            state_pool[state_base + (int64_t)v_idx * K + k_idx] = state[bv];
    }
}

// ---------------------------------------------------------------------------
// Shared-memory size helper  (constexpr, updated for new smem layout)
// ---------------------------------------------------------------------------

template <int BLOCK_K, int CHUNK_C>
constexpr size_t chunkwise_smem_bytes() {
    constexpr int nw = BLOCK_K / 32;
    // sq + warp_buf + pw_smem + pk_smem + batch_wbuf + batch_kbuf
    //   + K_smem + W_smem + U_smem + SW_smem + gamma + rbv + bcast
    return static_cast<size_t>(
        BLOCK_K              // sq
      + nw                   // warp_buf
      + CHUNK_C              // pw_smem  [OPT-1/2]
      + CHUNK_C              // pk_smem  [OPT-1]
      + nw * CHUNK_C         // batch_wbuf [OPT-1/2]
      + nw * CHUNK_C         // batch_kbuf [OPT-1]
      + CHUNK_C * BLOCK_K    // K_smem
      + CHUNK_C * BLOCK_K    // W_smem
      + CHUNK_C * BV         // U_smem
      + CHUNK_C * BV         // SW_smem
      + CHUNK_C              // gamma_sm
      + nw * BV              // rbv
      + BV                   // bcast
    ) * sizeof(float);
}

// ---------------------------------------------------------------------------
// Smem size reference (bytes) — all within 48 KB default limit:
//   BLOCK_K=64,  CHUNK_C=32, nw=2: ~21.5 KB  (K≤64)
//   BLOCK_K=128, CHUNK_C=32, nw=4: ~42.5 KB  (K=128, most common)
//   BLOCK_K=256, CHUNK_C=16, nw=8: ~39.4 KB  (K=256)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Launch wrapper (called via TVM FFI)
// ---------------------------------------------------------------------------

void run(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView state_pool,
    tvm::ffi::TensorView cache_indices,
    tvm::ffi::TensorView cu_seqlens,
    tvm::ffi::TensorView output
) {
    using namespace mllm_kernel::host;

    auto TT  = SymbolicSize{"total_tokens"};
    auto BS  = SymbolicSize{"batch_size"};
    auto BS1 = SymbolicSize{"batch_size_plus_1"};
    auto H_  = SymbolicSize{"H"};
    auto HV_ = SymbolicSize{"HV"};
    auto K_  = SymbolicSize{"K"};
    auto V_  = SymbolicSize{"V"};
    auto PS  = SymbolicSize{"pool_size"};
    auto dtype  = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    dtype.set_options<fp16_t, bf16_t>();

    (void)TensorMatcher({TT, H_, K_}).with_dtype(dtype).with_device(device).verify(q);
    (void)TensorMatcher({TT, H_, K_}).with_dtype(dtype).with_device(device).verify(k);
    (void)TensorMatcher({TT, HV_, V_}).with_dtype(dtype).with_device(device).verify(v);
    (void)TensorMatcher({TT, HV_}).with_dtype(dtype).with_device(device).verify(a);
    (void)TensorMatcher({TT, HV_}).with_dtype(dtype).with_device(device).verify(b);
    (void)TensorMatcher({HV_}).with_dtype<float>().with_device(device).verify(A_log);
    (void)TensorMatcher({HV_}).with_dtype<float>().with_device(device).verify(dt_bias);
    (void)TensorMatcher({PS, HV_, V_, K_}).with_dtype<float>().with_device(device).verify(state_pool);
    (void)TensorMatcher({BS}).with_device(device).verify(cache_indices);
    (void)TensorMatcher({BS1}).with_device(device).verify(cu_seqlens);
    (void)TensorMatcher({TT, HV_, V_}).with_dtype(dtype).with_device(device).verify(output);

    RuntimeCheck(BS1.unwrap() == BS.unwrap() + 1,
                 "cu_seqlens must have batch_size+1 elements");

    if (TT.unwrap() == 0) return;

    const int batch_size = static_cast<int>(BS.unwrap());
    const int H  = static_cast<int>(H_.unwrap());
    const int HV = static_cast<int>(HV_.unwrap());
    const int K  = static_cast<int>(K_.unwrap());
    const int V  = static_cast<int>(V_.unwrap());

    int block_k = ((K + 31) / 32) * 32;
    if (block_k > 256) block_k = 256;
    const int NV = (V + BV - 1) / BV;
    dim3 grid(NV, batch_size * HV);
    dim3 block(block_k);

    const DLDevice dl_device = device.unwrap();

    // ── Launch macro ──────────────────────────────────────────────────────
    // All configurations stay within the 48 KB default dynamic-smem limit:
    //   BLOCK_K ≤ 64,  CHUNK_C=32 → ~21.5 KB
    //   BLOCK_K = 128, CHUNK_C=32 → ~42.5 KB
    //   BLOCK_K = 256, CHUNK_C=16 → ~39.4 KB
    // No cudaFuncSetAttribute / runtime SM detection needed.
#define LAUNCH_CW(CType, BKVAL, CCVAL)                                        \
    LaunchKernel(grid, block, dl_device,                                       \
                 chunkwise_smem_bytes<BKVAL, CCVAL>())(                        \
        gdn_extend_chunkwise_kernel<CType, BKVAL, CCVAL>,                     \
        static_cast<const CType*>(q.data_ptr()),                              \
        static_cast<const CType*>(k.data_ptr()),                              \
        static_cast<const CType*>(v.data_ptr()),                              \
        static_cast<const CType*>(a.data_ptr()),                              \
        static_cast<const CType*>(b.data_ptr()),                              \
        static_cast<const float*>(A_log.data_ptr()),                          \
        static_cast<const float*>(dt_bias.data_ptr()),                        \
        static_cast<float*>(state_pool.data_ptr()),                           \
        static_cast<const int64_t*>(cache_indices.data_ptr()),                \
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),                   \
        static_cast<CType*>(output.data_ptr()),                               \
        batch_size, H, HV, K, V)

    // ── Dispatch ──────────────────────────────────────────────────────────
    if (dtype.is_type<bf16_t>()) {
        if      (block_k <= 64)  { LAUNCH_CW(__nv_bfloat16, 64,  32); }
        else if (block_k == 128) { LAUNCH_CW(__nv_bfloat16, 128, 32); }
        else                     { LAUNCH_CW(__nv_bfloat16, 256, 16); }
    } else {
        if      (block_k <= 64)  { LAUNCH_CW(__half, 64,  32); }
        else if (block_k == 128) { LAUNCH_CW(__half, 128, 32); }
        else                     { LAUNCH_CW(__half, 256, 16); }
    }

#undef LAUNCH_CW
}

}  // namespace GDNExtendChunkwiseKernel
