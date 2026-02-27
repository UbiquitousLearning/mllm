"""Benchmark store_cache vs torch index with torch.profiler.

Example:
python benchmarks/bench_store_cache.py --warmup 20 --iters 200 --batch-size 512 --num-slots 8192
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from mllm_kernel.cuda.jit import can_use_store_cache, store_cache


def _run_store_cache_once(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
):
    store_cache(k, v, k_cache, v_cache, indices)


def _run_torch_index_once(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
):
    k_cache[indices] = k
    v_cache[indices] = v


def _profile_path(
    name: str,
    fn,
    *,
    warmup: int,
    iters: int,
    row_limit: int,
    trace_path: str | None,
):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()

    events = prof.key_averages()
    # torch profiler times are in microseconds.
    # PyTorch versions vary between *cuda* and *device* naming.
    time_attr = (
        "self_cuda_time_total"
        if events and hasattr(events[0], "self_cuda_time_total")
        else "self_device_time_total"
    )
    sort_key = (
        "self_cuda_time_total"
        if time_attr == "self_cuda_time_total"
        else "self_device_time_total"
    )
    total_self_device_us = sum(float(getattr(evt, time_attr, 0.0)) for evt in events)
    avg_self_device_us = total_self_device_us / max(iters, 1)

    print(f"\n=== {name} ===")
    print(
        prof.key_averages().table(
            sort_by=sort_key,
            row_limit=row_limit,
        )
    )
    print(f"{name} total self device time: {total_self_device_us:.2f} us")
    print(f"{name} avg self device time/iter: {avg_self_device_us:.2f} us")

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"{name} trace exported: {trace_path}")

    return avg_self_device_us


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark store_cache vs torch index using torch.profiler"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-slots", type=int, default=16384)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument("--export-trace-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    row_dim = args.head_num * args.head_dim
    row_bytes = row_dim * torch.tensor([], dtype=dtype).element_size()
    if not can_use_store_cache(row_bytes):
        raise RuntimeError(f"store_cache is unavailable for row_bytes={row_bytes}")

    k = torch.randn(args.batch_size, row_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch_size, row_dim, device=device, dtype=dtype)
    # Use unique indices to avoid write conflicts.
    indices = torch.randperm(args.num_slots, device=device)[: args.batch_size].to(
        torch.int64
    )
    k_cache = torch.zeros(args.num_slots, row_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    print("=== store_cache profiler benchmark ===")
    print(
        f"shape: batch={args.batch_size}, row_dim={row_dim}, slots={args.num_slots}, dtype={dtype}"
    )
    print(f"warmup={args.warmup}, iters={args.iters}, row_limit={args.row_limit}")

    trace_dir = args.export_trace_dir.strip()
    store_trace = f"{trace_dir}/store_cache_trace.json" if trace_dir else None
    torch_trace = f"{trace_dir}/torch_index_trace.json" if trace_dir else None

    store_avg_us = _profile_path(
        "store_cache",
        lambda: _run_store_cache_once(k, v, k_cache, v_cache, indices),
        warmup=args.warmup,
        iters=args.iters,
        row_limit=args.row_limit,
        trace_path=store_trace,
    )
    torch_avg_us = _profile_path(
        "torch_index",
        lambda: _run_torch_index_once(k, v, k_cache, v_cache, indices),
        warmup=args.warmup,
        iters=args.iters,
        row_limit=args.row_limit,
        trace_path=torch_trace,
    )
    speedup = torch_avg_us / max(store_avg_us, 1e-12)
    print(f"\nSpeedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
