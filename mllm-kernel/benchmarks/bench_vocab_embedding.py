"""Benchmark vocab_embedding ops vs torch baseline with torch.profiler.

Example:
    python benchmarks/bench_vocab_embedding.py --op all --warmup 20 --iters 200
    python benchmarks/bench_vocab_embedding.py --op embedding_lookup --num-tokens 1024
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from mllm_kernel.cuda.jit.vocab_embedding import (
    assemble_deepstack_embedding,
    embedding_lookup,
    embedding_lookup_multimodal,
    embedding_lookup_with_image,
)

ALL_OPS = [
    "embedding_lookup",
    "embedding_lookup_with_image",
    "assemble_deepstack_embedding",
    "embedding_lookup_multimodal",
]


def _run_embedding_lookup_once(input_ids, embedding_table):
    embedding_lookup(input_ids, embedding_table)


def _run_torch_embedding_lookup_once(input_ids, embedding_table, output):
    output[:] = embedding_table[input_ids.long()]


def _run_embedding_lookup_with_image_once(input_ids, embedding_table, image_embeds):
    embedding_lookup_with_image(input_ids, embedding_table, image_embeds)


def _run_torch_embedding_lookup_with_image_once(
    input_ids, embedding_table, image_embeds, output
):
    vocab_size = embedding_table.shape[0]
    ids_long = input_ids.long()
    text_mask = input_ids < vocab_size
    img_mask = ~text_mask
    output[text_mask] = embedding_table[ids_long[text_mask]]
    output[img_mask] = image_embeds[ids_long[img_mask] - vocab_size]


def _run_assemble_deepstack_once(input_ids, deepstack_features, vocab_size):
    assemble_deepstack_embedding(input_ids, deepstack_features, vocab_size)


def _run_torch_assemble_deepstack_once(
    input_ids, deepstack_features, vocab_size, output
):
    output.zero_()
    img_mask = input_ids >= vocab_size
    output[img_mask] = deepstack_features[input_ids[img_mask].long() - vocab_size]


def _run_multimodal_once(
    input_ids,
    embedding_table,
    multimodal_indices,
    image_embeds,
    audio_embeds,
    image_token_id,
    audio_token_id,
):
    embedding_lookup_multimodal(
        input_ids,
        embedding_table,
        multimodal_indices,
        image_embeds,
        audio_embeds,
        image_token_id,
        audio_token_id,
    )


def _run_torch_multimodal_once(
    input_ids,
    embedding_table,
    multimodal_indices,
    image_embeds,
    audio_embeds,
    image_token_id,
    audio_token_id,
    output,
):
    vocab_size = embedding_table.shape[0]
    ids_long = input_ids.long()
    output.zero_()
    text_mask = (ids_long >= 0) & (ids_long < vocab_size)
    output[text_mask] = embedding_table[ids_long[text_mask]]
    if image_embeds is not None and image_embeds.shape[0] > 0 and image_token_id >= 0:
        img_mask = input_ids == image_token_id
        output[img_mask] = image_embeds[multimodal_indices[img_mask].long()]
    if audio_embeds is not None and audio_embeds.shape[0] > 0 and audio_token_id >= 0:
        aud_mask = input_ids == audio_token_id
        output[aud_mask] = audio_embeds[multimodal_indices[aud_mask].long()]


def _make_lookup_inputs(
    *, num_tokens, vocab_size, hidden_size, dtype, device, seed
):
    torch.manual_seed(seed)
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), device=device, dtype=torch.int32
    )
    embedding_table = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    return input_ids, embedding_table


def _make_lookup_with_image_inputs(
    *, num_tokens, vocab_size, hidden_size, image_token_len, dtype, device, seed
):
    torch.manual_seed(seed)
    n_text = num_tokens // 2
    n_image = num_tokens - n_text
    text_ids = torch.randint(
        0, vocab_size, (n_text,), device=device, dtype=torch.int32
    )
    image_ids = torch.randint(
        vocab_size,
        vocab_size + image_token_len,
        (n_image,),
        device=device,
        dtype=torch.int32,
    )
    input_ids = torch.cat([text_ids, image_ids])
    embedding_table = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    image_embeds = torch.randn(
        image_token_len, hidden_size, device=device, dtype=dtype
    )
    return input_ids, embedding_table, image_embeds


def _make_deepstack_inputs(
    *, num_tokens, vocab_size, hidden_size, num_image_tokens, dtype, device, seed
):
    torch.manual_seed(seed)
    input_ids = torch.randint(
        vocab_size,
        vocab_size + num_image_tokens,
        (num_tokens,),
        device=device,
        dtype=torch.int32,
    )
    deepstack_features = torch.randn(
        num_image_tokens, hidden_size, device=device, dtype=dtype
    )
    return input_ids, deepstack_features


def _make_multimodal_inputs(
    *,
    num_tokens,
    vocab_size,
    hidden_size,
    image_token_len,
    audio_token_len,
    image_token_id,
    audio_token_id,
    dtype,
    device,
    seed,
):
    torch.manual_seed(seed)

    n_text = num_tokens // 3
    n_image = num_tokens // 3
    n_audio = num_tokens - n_text - n_image

    text_ids = torch.randint(
        0, vocab_size, (n_text,), device=device, dtype=torch.int32
    )
    image_ids = torch.full(
        (n_image,), image_token_id, device=device, dtype=torch.int32
    )
    audio_ids = torch.full(
        (n_audio,), audio_token_id, device=device, dtype=torch.int32
    )
    input_ids = torch.cat([text_ids, image_ids, audio_ids])

    text_idx = torch.zeros(n_text, device=device, dtype=torch.int32)
    image_idx = torch.randint(
        0, image_token_len, (n_image,), device=device, dtype=torch.int32
    )
    audio_idx = torch.randint(
        0, audio_token_len, (n_audio,), device=device, dtype=torch.int32
    )
    multimodal_indices = torch.cat([text_idx, image_idx, audio_idx])

    perm = torch.randperm(num_tokens, device=device)
    input_ids = input_ids[perm]
    multimodal_indices = multimodal_indices[perm]

    embedding_table = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    image_embeds = torch.randn(
        image_token_len, hidden_size, device=device, dtype=dtype
    )
    audio_embeds = torch.randn(
        audio_token_len, hidden_size, device=device, dtype=dtype
    )
    return input_ids, embedding_table, multimodal_indices, image_embeds, audio_embeds


def _profile_path(
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
        description="Benchmark vocab_embedding ops vs torch baseline using torch.profiler"
    )
    parser.add_argument(
        "--op",
        type=str,
        default="all",
        choices=["all"] + ALL_OPS,
    )
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--image-token-len", type=int, default=576)
    parser.add_argument("--audio-token-len", type=int, default=256)
    parser.add_argument("--image-token-id", type=int, default=32000)
    parser.add_argument("--audio-token-id", type=int, default=32001)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
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
    trace_dir = args.export_trace_dir.strip()

    ops = ALL_OPS if args.op == "all" else [args.op]

    for op in ops:
        if op == "embedding_lookup":
            input_ids, embedding_table = _make_lookup_inputs(
                num_tokens=args.num_tokens,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                dtype=dtype,
                device=device,
                seed=args.seed,
            )
            output = torch.empty(
                args.num_tokens, args.hidden_size, dtype=dtype, device=device
            )

            print("=== embedding_lookup profiler benchmark ===")
            print(
                f"shape: num_tokens={args.num_tokens}, vocab_size={args.vocab_size}, "
                f"hidden_size={args.hidden_size}, dtype={dtype}"
            )
            print(
                f"warmup={args.warmup}, iters={args.iters}, row_limit={args.row_limit}"
            )

            kernel_trace = (
                f"{trace_dir}/embedding_lookup_kernel_trace.json"
                if trace_dir
                else None
            )
            torch_trace = (
                f"{trace_dir}/embedding_lookup_torch_trace.json"
                if trace_dir
                else None
            )

            kernel_avg_us = _profile_path(
                "embedding_lookup",
                lambda: _run_embedding_lookup_once(input_ids, embedding_table),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=kernel_trace,
            )

            torch_avg_us = _profile_path(
                "torch_embedding_lookup",
                lambda: _run_torch_embedding_lookup_once(
                    input_ids, embedding_table, output
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=torch_trace,
            )

            speedup = torch_avg_us / max(kernel_avg_us, 1e-12)
            print(f"\nSpeedup: {speedup:.3f}x")

        elif op == "embedding_lookup_with_image":
            input_ids, embedding_table, image_embeds = _make_lookup_with_image_inputs(
                num_tokens=args.num_tokens,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                image_token_len=args.image_token_len,
                dtype=dtype,
                device=device,
                seed=args.seed,
            )
            output = torch.empty(
                args.num_tokens, args.hidden_size, dtype=dtype, device=device
            )

            print("=== embedding_lookup_with_image profiler benchmark ===")
            print(
                f"shape: num_tokens={args.num_tokens}, vocab_size={args.vocab_size}, "
                f"hidden_size={args.hidden_size}, image_token_len={args.image_token_len}, "
                f"dtype={dtype}"
            )
            print(
                f"warmup={args.warmup}, iters={args.iters}, row_limit={args.row_limit}"
            )

            kernel_trace = (
                f"{trace_dir}/embedding_lookup_with_image_kernel_trace.json"
                if trace_dir
                else None
            )
            torch_trace = (
                f"{trace_dir}/embedding_lookup_with_image_torch_trace.json"
                if trace_dir
                else None
            )

            kernel_avg_us = _profile_path(
                "embedding_lookup_with_image",
                lambda: _run_embedding_lookup_with_image_once(
                    input_ids, embedding_table, image_embeds
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=kernel_trace,
            )

            torch_avg_us = _profile_path(
                "torch_embedding_lookup_with_image",
                lambda: _run_torch_embedding_lookup_with_image_once(
                    input_ids, embedding_table, image_embeds, output
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=torch_trace,
            )

            speedup = torch_avg_us / max(kernel_avg_us, 1e-12)
            print(f"\nSpeedup: {speedup:.3f}x")

        elif op == "assemble_deepstack_embedding":
            input_ids, deepstack_features = _make_deepstack_inputs(
                num_tokens=args.num_tokens,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                num_image_tokens=args.image_token_len,
                dtype=dtype,
                device=device,
                seed=args.seed,
            )
            output = torch.empty(
                args.num_tokens, args.hidden_size, dtype=dtype, device=device
            )

            print("=== assemble_deepstack_embedding profiler benchmark ===")
            print(
                f"shape: num_tokens={args.num_tokens}, vocab_size={args.vocab_size}, "
                f"hidden_size={args.hidden_size}, num_image_tokens={args.image_token_len}, "
                f"dtype={dtype}"
            )
            print(
                f"warmup={args.warmup}, iters={args.iters}, row_limit={args.row_limit}"
            )

            kernel_trace = (
                f"{trace_dir}/assemble_deepstack_kernel_trace.json"
                if trace_dir
                else None
            )
            torch_trace = (
                f"{trace_dir}/assemble_deepstack_torch_trace.json"
                if trace_dir
                else None
            )

            vocab_size = args.vocab_size

            kernel_avg_us = _profile_path(
                "assemble_deepstack_embedding",
                lambda: _run_assemble_deepstack_once(
                    input_ids, deepstack_features, vocab_size
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=kernel_trace,
            )

            torch_avg_us = _profile_path(
                "torch_assemble_deepstack",
                lambda: _run_torch_assemble_deepstack_once(
                    input_ids, deepstack_features, vocab_size, output
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=torch_trace,
            )

            speedup = torch_avg_us / max(kernel_avg_us, 1e-12)
            print(f"\nSpeedup: {speedup:.3f}x")

        elif op == "embedding_lookup_multimodal":
            (
                input_ids,
                embedding_table,
                multimodal_indices,
                image_embeds,
                audio_embeds,
            ) = _make_multimodal_inputs(
                num_tokens=args.num_tokens,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                image_token_len=args.image_token_len,
                audio_token_len=args.audio_token_len,
                image_token_id=args.image_token_id,
                audio_token_id=args.audio_token_id,
                dtype=dtype,
                device=device,
                seed=args.seed,
            )
            output = torch.empty(
                args.num_tokens, args.hidden_size, dtype=dtype, device=device
            )

            print("=== embedding_lookup_multimodal profiler benchmark ===")
            print(
                f"shape: num_tokens={args.num_tokens}, vocab_size={args.vocab_size}, "
                f"hidden_size={args.hidden_size}, image_token_len={args.image_token_len}, "
                f"audio_token_len={args.audio_token_len}, dtype={dtype}"
            )
            print(
                f"image_token_id={args.image_token_id}, "
                f"audio_token_id={args.audio_token_id}"
            )
            print(
                f"warmup={args.warmup}, iters={args.iters}, row_limit={args.row_limit}"
            )

            image_token_id = args.image_token_id
            audio_token_id = args.audio_token_id

            kernel_trace = (
                f"{trace_dir}/multimodal_kernel_trace.json" if trace_dir else None
            )
            torch_trace = (
                f"{trace_dir}/multimodal_torch_trace.json" if trace_dir else None
            )

            kernel_avg_us = _profile_path(
                "embedding_lookup_multimodal",
                lambda: _run_multimodal_once(
                    input_ids,
                    embedding_table,
                    multimodal_indices,
                    image_embeds,
                    audio_embeds,
                    image_token_id,
                    audio_token_id,
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=kernel_trace,
            )

            torch_avg_us = _profile_path(
                "torch_multimodal",
                lambda: _run_torch_multimodal_once(
                    input_ids,
                    embedding_table,
                    multimodal_indices,
                    image_embeds,
                    audio_embeds,
                    image_token_id,
                    audio_token_id,
                    output,
                ),
                warmup=args.warmup,
                iters=args.iters,
                row_limit=args.row_limit,
                trace_path=torch_trace,
            )

            speedup = torch_avg_us / max(kernel_avg_us, 1e-12)
            print(f"\nSpeedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
