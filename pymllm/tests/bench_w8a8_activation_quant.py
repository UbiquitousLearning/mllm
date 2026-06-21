"""Benchmark W8A8 activation quantization implementations.

Covers: torch path (current) and (future) Triton kernel.
This script is reusable across phases.

Usage:
    python pymllm/tests/bench_w8a8_activation_quant.py
"""
from __future__ import annotations

import time

import torch


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def torch_per_token_quant_int8(x: torch.Tensor):
    """Current torch-based activation quantization."""
    x_fp32 = x.to(torch.float32)
    absmax = torch.clamp(x_fp32.abs().amax(dim=-1, keepdim=True), min=1e-10)
    x_scale = absmax / 127.0
    x_q = torch.round(x_fp32 / x_scale).clamp(-128, 127).to(torch.int8)
    return x_q.contiguous(), x_scale.contiguous()


def _try_load_triton_kernel():
    try:
        from pymllm.quantization.kernels.int8_activation_triton import per_token_quant_int8
        return per_token_quant_int8
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_fn(fn, args, warmup=5, repeat=20) -> float:
    """Returns median latency in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def run_benchmarks():
    device = "cuda"

    shapes = [
        # (M, K) — M=tokens, K=hidden_dim
        (1, 2048),
        (8, 2048),
        (16, 2048),
        (32, 2048),
        (93, 2048),
        (128, 2048),
        (256, 2048),
    ]

    backends = {}
    backends["torch"] = torch_per_token_quant_int8

    triton_fn = _try_load_triton_kernel()
    if triton_fn is not None:
        backends["triton"] = triton_fn

    print(f"{'Shape':>16s}", end="")
    for name in backends:
        print(f"  {name:>12s}", end="")
    print()
    print("-" * (16 + 14 * len(backends)))

    for M, K in shapes:
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        row_label = f"({M},{K})"
        print(f"{row_label:>16s}", end="")

        for name, fn in backends.items():
            try:
                ms = bench_fn(fn, (x,))
                print(f"  {ms:>9.3f} ms", end="")
            except Exception as e:
                print(f"  {'ERR':>12s}", end="")

        print()


if __name__ == "__main__":
    print("=" * 50)
    print("W8A8 Activation Quantization Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"SM: {torch.cuda.get_device_capability(0)}")
    print("=" * 50)
    run_benchmarks()
