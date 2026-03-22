"""Benchmark create_kv_indices vs naive torch gather using torch.profiler.

Example:
    python benchmarks/bench_create_kv_indices.py --batch-size 512 --max-reqs 2048 --max-ctx 4096
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from mllm_kernel.cuda.jit.create_kv_indices import create_kv_indices


def _make_batch(
    *,
    max_reqs: int,
    max_ctx: int,
    batch_size: int,
    use_start_offsets: bool,
    device: torch.device,
    seed: int,
):
    g_cuda = torch.Generator(device=device).manual_seed(seed)
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)

    req_to_token = torch.arange(
        max_reqs * max_ctx, dtype=torch.int32, device=device
    ).reshape(max_reqs, max_ctx)

    assert batch_size <= max_reqs
    req_pool_indices = torch.randperm(max_reqs, generator=g_cuda, device=device)[
        :batch_size
    ].to(torch.int32)

    page_kernel_lens_list = []
    kv_start_idx_list = []
    for _ in range(batch_size):
        L = int(torch.randint(1, max_ctx, (1,), generator=g_cpu).item())
        if use_start_offsets:
            start_max = max_ctx - L
            start = int(torch.randint(0, max(start_max, 1), (1,), generator=g_cpu).item())
        else:
            start = 0
        page_kernel_lens_list.append(L)
        kv_start_idx_list.append(start)

    page_kernel_lens = torch.tensor(
        page_kernel_lens_list, dtype=torch.int32, device=device
    )
    kv_start_idx = torch.tensor(kv_start_idx_list, dtype=torch.int32, device=device)

    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(page_kernel_lens, dim=0)

    kv_indices = torch.empty(
        int(kv_indptr[-1].item()), dtype=torch.int32, device=device
    )

    return (
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    )


def _profile(
    name: str, fn, *, warmup: int, iters: int, row_limit: int, trace_path: str | None
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
    total_us = sum(float(getattr(evt, time_attr, 0.0)) for evt in events)
    avg_us = total_us / max(iters, 1)

    print(f"\n=== {name} ===")
    print(
        prof.key_averages().table(
            sort_by=sort_key,
            row_limit=row_limit,
        )
    )
    print(f"{name} total self device time: {total_us:.2f} us")
    print(f"{name} avg self device time/iter: {avg_us:.2f} us")

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"{name} trace exported: {trace_path}")

    return avg_us


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark create_kv_indices vs naive torch gather",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-reqs", type=int, default=2048)
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument("--export-trace-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-start-offsets",
        action="store_true",
        help="Enable non-zero kv_start_idx to emulate sliding-window decode",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    (
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    ) = _make_batch(
        max_reqs=args.max_reqs,
        max_ctx=args.max_ctx,
        batch_size=args.batch_size,
        use_start_offsets=args.use_start_offsets,
        device=device,
        seed=args.seed,
    )

    print("=== create_kv_indices profiler benchmark ===")
    print(
        f"batch_size={args.batch_size}, max_reqs={args.max_reqs}, max_ctx={args.max_ctx}, "
        f"use_start_offsets={args.use_start_offsets}"
    )
    print(f"warmup={args.warmup}, iters={args.iters}, row_limit={args.row_limit}")

    trace_dir = args.export_trace_dir.strip()
    kernel_trace = f"{trace_dir}/create_kv_indices_trace.json" if trace_dir else None
    torch_trace = f"{trace_dir}/torch_gather_trace.json" if trace_dir else None

    def _run_kernel_once():
        create_kv_indices(
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
        )

    def _run_torch_once():
        # Torch reference implementation on device: gather per-sequence ranges
        # from req_to_token into a flat buffer.
        out = []
        for i in range(args.batch_size):
            req = req_pool_indices[i].item()
            start = kv_start_idx[i].item() if args.use_start_offsets else 0
            L = page_kernel_lens[i].item()
            row = req_to_token[req, start : start + L]
            out.append(row)
        torch.cat(out, out=kv_indices)

    kernel_avg_us = _profile(
        "create_kv_indices",
        _run_kernel_once,
        warmup=args.warmup,
        iters=args.iters,
        row_limit=args.row_limit,
        trace_path=kernel_trace,
    )

    torch_avg_us = _profile(
        "torch_reference",
        _run_torch_once,
        warmup=args.warmup,
        iters=args.iters,
        row_limit=args.row_limit,
        trace_path=torch_trace,
    )

    speedup = torch_avg_us / max(kernel_avg_us, 1e-12)
    print(f"\nSpeedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
