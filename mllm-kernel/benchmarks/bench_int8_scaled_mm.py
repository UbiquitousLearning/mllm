"""Benchmark int8_scaled_mm implementations.

Covers torch._int_mm and the CUTLASS W8A8 kernel.

Usage:
    python benchmarks/bench_int8_scaled_mm.py
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Reference / backend implementations
# ---------------------------------------------------------------------------

def _torch_int_mm_scaled(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """torch._int_mm + scale dequant reference backend."""
    m = mat_a.shape[0]
    if m <= 16:
        padded = torch.zeros((17, mat_a.shape[1]), device=mat_a.device, dtype=torch.int8)
        padded[:m].copy_(mat_a)
        out_i32 = torch._int_mm(padded, mat_b)[:m]
    else:
        out_i32 = torch._int_mm(mat_a, mat_b)
    out = out_i32.to(torch.float32)
    out.mul_(scales_a.view(-1, 1))
    out.mul_(scales_b.view(1, -1))
    out = out.to(out_dtype)
    if bias is not None:
        out.add_(bias)
    return out


def _try_load_cutlass_kernel():
    try:
        from mllm_kernel.cuda.jit.int8_scaled_mm_cutlass import int8_scaled_mm
        return int8_scaled_mm
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_fn(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Returns median latency in ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def run_benchmarks():
    device = "cuda"
    out_dtype = torch.float16

    # Shapes representative of Qwen3-VL-2B linear layers
    shapes = [
        # (M, K, N) — M=seq_len, K=in_features, N=out_features
        (1, 2048, 2048),      # decode, hidden->hidden
        (1, 2048, 6144),      # decode, hidden->3*hidden (QKV)
        (8, 2048, 6144),      # small batch
        (16, 2048, 6144),     # boundary (torch._int_mm M<=16 padding)
        (32, 2048, 6144),     # medium batch
        (93, 2048, 6144),     # typical prefill
        (128, 2048, 6144),    # larger prefill
        (93, 6144, 2048),     # prefill, wide->narrow (down_proj)
    ]

    backends = {}

    # Backend: torch._int_mm
    backends["torch._int_mm"] = _torch_int_mm_scaled

    # Backend: CUTLASS
    cutlass_fn = _try_load_cutlass_kernel()
    if cutlass_fn is not None:
        backends["cutlass"] = cutlass_fn

    print(f"{'Shape':>20s}", end="")
    for name in backends:
        print(f"  {name:>16s}", end="")
    print()
    print("-" * (20 + 18 * len(backends)))

    results = []
    for M, K, N in shapes:
        torch.manual_seed(42)
        mat_a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=device)
        mat_b = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=device)
        scales_a = torch.rand(M, dtype=torch.float32, device=device) + 0.01
        scales_b = torch.rand(N, dtype=torch.float32, device=device) + 0.01

        # CUTLASS needs col-major B
        mat_b_colmaj = mat_b.t().contiguous().t()

        row = {"shape": f"({M},{K},{N})"}
        print(f"{row['shape']:>20s}", end="")

        for name, fn in backends.items():
            kwargs = dict(out_dtype=out_dtype)
            b_arg = mat_b_colmaj if name == "cutlass" else mat_b
            try:
                ms = bench_fn(fn, (mat_a, b_arg, scales_a, scales_b), kwargs)
                row[name] = f"{ms:.3f}"
                print(f"  {ms:>13.3f} ms", end="")
            except Exception as e:
                row[name] = f"ERR: {e}"
                print(f"  {'ERROR':>16s}", end="")

        print()
        results.append(row)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("INT8 Scaled MM Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"SM: {torch.cuda.get_device_capability(0)}")
    print("=" * 60)
    run_benchmarks()
