"""Kernel-level benchmark: W4A16 (GPTQ-Marlin) vs W8A8 (Triton quant + CUTLASS GEMM).

Isolates kernel performance from serving framework overhead.
Shapes are from Qwen3-VL-2B linear layers.

Usage:
    cd /workspace/.worktrees/pymllm-qwen3-vl-w8a8
    python3 mllm-kernel/benchmarks/bench_w4a16_vs_w8a8.py
"""
from __future__ import annotations

import time
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def bench(fn: Callable, warmup: int = 5, repeat: int = 20) -> float:
    """Returns median latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# W8A8 kernel loaders
# ---------------------------------------------------------------------------

def load_cutlass_mm():
    from mllm_kernel.cuda.jit.int8_scaled_mm_cutlass import int8_scaled_mm
    return int8_scaled_mm


def load_triton_quant():
    from pymllm.quantization.kernels.int8_activation_triton import per_token_quant_int8
    return per_token_quant_int8


# ---------------------------------------------------------------------------
# W4A16 kernel loader
# ---------------------------------------------------------------------------

def load_marlin():
    from mllm_kernel.cuda.jit import gptq_marlin_gemm, gptq_marlin_repack
    from pymllm.quantization.methods.compressed_tensors import (
        marlin_make_workspace,
        marlin_make_empty_g_idx,
        marlin_permute_scales,
        SCALAR_TYPE_UINT4B8,
    )
    return gptq_marlin_gemm, gptq_marlin_repack, marlin_make_workspace, \
           marlin_make_empty_g_idx, marlin_permute_scales, SCALAR_TYPE_UINT4B8


def prepare_marlin_weights(K: int, N: int, group_size: int, device: str):
    """Create fake W4A16 weights in Marlin format for benchmarking."""
    gptq_marlin_gemm, gptq_marlin_repack, marlin_make_workspace, \
        marlin_make_empty_g_idx, marlin_permute_scales, SCALAR_TYPE_UINT4B8 = load_marlin()

    pack_factor = 8  # 32 / 4 bits
    w_packed = torch.randint(
        0, 2**31, (N, K // pack_factor), dtype=torch.int32, device=device,
    )
    w_scale = (
        torch.rand(N, K // group_size, dtype=torch.float16, device=device) + 0.01
    )

    repacked = gptq_marlin_repack(
        w_packed.t().contiguous(),
        perm=torch.empty(0, dtype=torch.int32, device=device),
        size_k=K, size_n=N, num_bits=4,
    )
    scales_perm = marlin_permute_scales(
        w_scale.t().contiguous(), size_k=K, size_n=N, group_size=group_size,
    )
    workspace = marlin_make_workspace(torch.device(device))
    g_idx = marlin_make_empty_g_idx(torch.device(device))

    return repacked, scales_perm, workspace, g_idx, SCALAR_TYPE_UINT4B8


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks():
    device = "cuda"
    group_size = 32

    shapes = [
        # (M, K, N, description)
        (1, 2048, 6144, "QKV proj"),
        (1, 2048, 2048, "O proj"),
        (1, 6144, 2048, "down proj"),
        (93, 2048, 6144, "QKV proj"),
        (93, 2048, 2048, "O proj"),
        (93, 6144, 2048, "down proj"),
        (128, 2048, 6144, "QKV proj"),
    ]

    # Load kernels
    cutlass_mm = load_cutlass_mm()
    triton_quant = load_triton_quant()
    gptq_marlin_gemm = load_marlin()[0]

    # Header
    print(f"{'Shape':<22s} {'':>6s}  {'W4A16':>8s}  {'W8A8':>8s}  {'W8A8':>8s}  {'W8A8':>8s}")
    print(f"{'(M, K, N)':<22s} {'desc':>6s}  {'Marlin':>8s}  {'quant':>8s}  {'GEMM':>8s}  {'total':>8s}")
    print("-" * 72)

    for M, K, N, desc in shapes:
        torch.manual_seed(42)

        # ----- W8A8 setup -----
        x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
        w_int8_col = torch.randint(
            -127, 128, (N, K), dtype=torch.int8, device=device,
        ).t()  # (K, N) col-major, stride(0)==1
        w_scale_f32 = torch.rand(N, dtype=torch.float32, device=device) * 0.01

        # Pre-quantize for GEMM-only bench
        x_q, x_s = triton_quant(x_fp16)

        ms_quant = bench(lambda: triton_quant(x_fp16))
        ms_gemm = bench(lambda: cutlass_mm(x_q, w_int8_col, x_s, w_scale_f32, torch.float16))
        ms_w8a8 = ms_quant + ms_gemm

        # ----- W4A16 setup -----
        repacked, scales_perm, workspace, g_idx, scalar_type = \
            prepare_marlin_weights(K, N, group_size, device)
        x_marlin = torch.randn(M, K, device=device, dtype=torch.float16)

        def run_marlin():
            return gptq_marlin_gemm(
                a=x_marlin, c=None, b_q_weight=repacked, b_scales=scales_perm,
                global_scale=None, b_zeros=g_idx, g_idx=g_idx, perm=g_idx,
                workspace=workspace, b_q_type_id=scalar_type.id,
                size_m=M, size_n=N, size_k=K, is_k_full=True,
                use_fp32_reduce=True, is_zp_float=False,
            )

        ms_marlin = bench(run_marlin)

        # ----- Print -----
        tag = "decode" if M <= 8 else "prefill"
        print(
            f"  ({M:>3},{K:>4},{N:>4}) {desc:<8s}"
            f"  {ms_marlin:>7.3f}   {ms_quant:>7.3f}   {ms_gemm:>7.3f}   {ms_w8a8:>7.3f}"
        )

    # Summary
    print()
    print("W4A16 Marlin : gptq_marlin_gemm (int4 weight * fp16 activation, 1 kernel)")
    print("W8A8 quant   : Triton per_token_quant_int8 (fp16 -> int8, 1 kernel)")
    print("W8A8 GEMM    : CUTLASS int8_scaled_mm (int8 * int8, fused scale, 1 kernel)")
    print("W8A8 total   : quant + GEMM (2 kernel launches)")
    print()
    print("Key insight: W8A8 GEMM alone is faster than W4A16 Marlin,")
    print("but activation quantization overhead makes W8A8 total slower at decode (M=1).")


if __name__ == "__main__":
    print("=" * 72)
    print("W4A16 vs W8A8 Kernel Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"SM: {torch.cuda.get_device_capability(0)}")
    print("=" * 72)
    run_benchmarks()
