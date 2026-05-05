"""Chunkwise parallel GDN extend (prefill) using WY representation.

Implements the WY decomposition from the Gated DeltaNet paper to
parallelize the GDN recurrent scan over token chunks.  Within each
chunk the key operations (output and state computation) are expressed
as batch matrix multiplications, enabling efficient GPU utilisation.

The inter-chunk state propagation remains sequential, but reduces the
number of iterations from T to ceil(T / C) where C is the chunk size.

Algorithm (per chunk of C tokens with initial state S_0)
---------------------------------------------------------
1. **WY construction** (sequential, O(C²K) per head):

       w_i = β_i k_i - K_{<i}(W_{<i}^T (β_i k_i))
       ũ_i = (β_i / γ_i) v_i - Ũ_{<i}(K_{<i}^T (β_i k_i))

   where γ_i = Π_{j=0}^{i} α_j is the cumulative decay.

2. **Output computation** (parallel, batch matmul):

       O = γ · (S_0 Q − (S_0 W) · triu(K^T Q) + Ũ · triu(K^T Q))

   ``triu`` is the causal mask (upper-triangular: j ≤ r).

3. **State update** (parallel, batch matmul):

       S_new = γ_C · (S_0 − (S_0 W) K^T + Ũ^T K)

Usage::

    from pymllm.layers.attention.gdn_chunkwise import gdn_extend_chunkwise

    output = gdn_extend_chunkwise(
        q, k, v, a, b, A_log, dt_bias,
        state_pool, cache_indices, cu_seqlens,
        chunk_size=64,
    )
"""

from __future__ import annotations

import torch


def gdn_extend_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Chunkwise parallel GDN extend with WY representation.

    Parameters
    ----------
    q : Tensor [total_tokens, H, K]
        Query tensor (bf16/fp16).  NOT pre-normalised.
    k : Tensor [total_tokens, H, K]
        Key tensor (bf16/fp16).
    v : Tensor [total_tokens, HV, V]
        Value tensor (bf16/fp16).
    a : Tensor [total_tokens, HV]
        Raw decay-gate input (before softplus/exp).
    b : Tensor [total_tokens, HV]
        Raw update-gate input (before sigmoid).
    A_log : Tensor [HV]
        Log-space decay parameter, float32.
    dt_bias : Tensor [HV]
        Bias for decay gate, float32.
    state_pool : Tensor [pool_size, HV, V, K]
        Pooled recurrent state, float32.  Modified **in-place**.
    cache_indices : Tensor [batch_size]
        Pool index per request, int64.
    cu_seqlens : Tensor [batch_size + 1]
        Cumulative sequence lengths, int64.
    chunk_size : int
        Number of tokens per chunk (default 64).

    Returns
    -------
    Tensor [total_tokens, HV, V]
        Output tensor, same dtype as *v*.
    """
    total_tokens, H, K = q.shape
    HV = v.shape[1]
    V = v.shape[2]
    batch_size = cache_indices.shape[0]
    device = q.device

    orig_dtype = v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()

    # --- gating (all tokens) ---
    x = a + dt_bias.unsqueeze(0)
    softplus_x = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log.unsqueeze(0)) * softplus_x
    alpha = torch.exp(g)
    beta = torch.sigmoid(b)

    # --- L2 normalise q, k (per key-head, before GQA expansion) ---
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    q = q * (K ** -0.5)

    # --- GQA expansion ---
    if H != HV:
        repeats = HV // H
        q = q.repeat_interleave(repeats, dim=1)
        k = k.repeat_interleave(repeats, dim=1)

    output = torch.zeros(total_tokens, HV, V, device=device, dtype=torch.float32)

    for i in range(batch_size):
        seq_start = int(cu_seqlens[i].item())
        seq_end = int(cu_seqlens[i + 1].item())
        seq_len = seq_end - seq_start
        if seq_len == 0:
            continue

        pool_idx = int(cache_indices[i].item())
        state = state_pool[pool_idx].clone()  # [HV, V, K]

        q_seq = q[seq_start:seq_end]
        k_seq = k[seq_start:seq_end]
        v_seq = v[seq_start:seq_end]
        alpha_seq = alpha[seq_start:seq_end]
        beta_seq = beta[seq_start:seq_end]

        for cs in range(0, seq_len, chunk_size):
            ce = min(cs + chunk_size, seq_len)
            C = ce - cs

            q_c = q_seq[cs:ce]          # [C, HV, K]
            k_c = k_seq[cs:ce]          # [C, HV, K]
            v_c = v_seq[cs:ce]          # [C, HV, V]
            alpha_c = alpha_seq[cs:ce]  # [C, HV]
            beta_c = beta_seq[cs:ce]    # [C, HV]

            state = _process_chunk(
                q_c, k_c, v_c, alpha_c, beta_c,
                state, output,
                seq_start + cs, C, HV, K, V, device,
            )

        state_pool[pool_idx] = state

    return output.to(orig_dtype)


def _process_chunk(
    q_c: torch.Tensor,      # [C, HV, K]
    k_c: torch.Tensor,      # [C, HV, K]
    v_c: torch.Tensor,      # [C, HV, V]
    alpha_c: torch.Tensor,  # [C, HV]
    beta_c: torch.Tensor,   # [C, HV]
    state: torch.Tensor,    # [HV, V, K]
    output: torch.Tensor,   # [total_tokens, HV, V]  (written in-place)
    write_offset: int,
    C: int,
    HV: int,
    K: int,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Process one chunk: WY construction → matmul output → state update."""

    # ---- cumulative decay within chunk ----
    log_alpha_c = torch.log(alpha_c.clamp(min=1e-10))  # [C, HV]
    cum_log = torch.cumsum(log_alpha_c, dim=0)          # [C, HV]
    gamma = torch.exp(cum_log)                           # [C, HV]
    gamma_C = gamma[-1]                                  # [HV]

    # ---- precompute β·k and (β/γ)·v ----
    bk = beta_c.unsqueeze(-1) * k_c                      # [C, HV, K]
    bv_scaled = (beta_c / gamma).unsqueeze(-1) * v_c     # [C, HV, V]

    # Contiguous buffers for WY vectors (permuted: [HV, dim, C])
    bk_p = bk.permute(1, 2, 0).contiguous()              # [HV, K, C]
    bv_p = bv_scaled.permute(1, 2, 0).contiguous()       # [HV, V, C]
    K_p = k_c.permute(1, 2, 0).contiguous()              # [HV, K, C]

    W_buf = torch.zeros(HV, K, C, device=device, dtype=torch.float32)
    U_buf = torch.zeros(HV, V, C, device=device, dtype=torch.float32)

    # ================================================================
    # Phase 1 — WY construction (sequential over C, parallel over HV)
    # ================================================================
    W_buf[:, :, 0] = bk_p[:, :, 0]
    U_buf[:, :, 0] = bv_p[:, :, 0]

    for t in range(1, C):
        bk_t = bk_p[:, :, t : t + 1]            # [HV, K, 1]
        K_prev = K_p[:, :, :t]                   # [HV, K, t]
        W_prev = W_buf[:, :, :t]                 # [HV, K, t]
        U_prev = U_buf[:, :, :t]                 # [HV, V, t]

        # proj_w = W_prev^T @ (β_t k_t)
        proj_w = torch.bmm(W_prev.transpose(1, 2), bk_t)   # [HV, t, 1]
        # w correction = K_prev @ proj_w
        w_corr = torch.bmm(K_prev, proj_w)                  # [HV, K, 1]
        W_buf[:, :, t : t + 1] = bk_t - w_corr

        # proj_k = K_prev^T @ (β_t k_t)  (shared for U update)
        proj_k = torch.bmm(K_prev.transpose(1, 2), bk_t)   # [HV, t, 1]
        u_corr = torch.bmm(U_prev, proj_k)                  # [HV, V, 1]
        U_buf[:, :, t : t + 1] = bv_p[:, :, t : t + 1] - u_corr

    # ================================================================
    # Phase 2 — output computation (batch matmul)
    # ================================================================
    Q_p = q_c.permute(1, 2, 0).contiguous()      # [HV, K, C]

    SQ = torch.bmm(state, Q_p)                    # [HV, V, C]
    SW = torch.bmm(state, W_buf)                   # [HV, V, C]

    # K^T Q with causal mask (j ≤ r → upper triangular)
    KQ = torch.bmm(K_p.transpose(1, 2), Q_p)      # [HV, C, C]
    causal = torch.triu(torch.ones(C, C, device=device, dtype=torch.float32))
    KQ_masked = KQ * causal.unsqueeze(0)

    SW_KQ = torch.bmm(SW, KQ_masked)              # [HV, V, C]
    U_KQ = torch.bmm(U_buf, KQ_masked)            # [HV, V, C]

    # O = γ · (S_0 Q − (S_0 W) triu(K^T Q) + Ũ triu(K^T Q))
    gamma_bc = gamma.permute(1, 0).unsqueeze(1)    # [HV, 1, C]
    O_chunk = gamma_bc * (SQ - SW_KQ + U_KQ)      # [HV, V, C]

    output[write_offset : write_offset + C] = O_chunk.permute(2, 0, 1)

    # ================================================================
    # Phase 3 — state update
    # ================================================================
    # S_new = γ_C · (S_0 − S_0 W K^T + Ũ^T K)
    K_pT = K_p.transpose(1, 2)                    # [HV, C, K]
    SW_K = torch.bmm(SW, K_pT)                    # [HV, V, K]
    UK = torch.bmm(U_buf, K_pT)                   # [HV, V, K]

    gamma_C_bc = gamma_C.unsqueeze(-1).unsqueeze(-1)
    return gamma_C_bc * (state - SW_K + UK)
