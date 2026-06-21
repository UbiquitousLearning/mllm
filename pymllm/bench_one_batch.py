"""SGLang-style one-batch benchmark for pymllm.

This module intentionally bypasses the HTTP server, tokenizer workers,
scheduler, and detokenizer.  It drives :class:`pymllm.executor.ModelRunner`
directly to measure one static prefill followed by token-by-token decode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import torch

from pymllm.configs.global_config import GlobalConfig, make_args, read_args
from pymllm.executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchSetting:
    batch_size: int
    input_len: int
    output_len: int


@dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: list[int] = field(default_factory=lambda: [1])
    input_len: list[int] = field(default_factory=lambda: [256, 512, 1024])
    output_len: list[int] = field(default_factory=lambda: [128])
    result_filename: Path = Path("/tmp/pymllm_bench_one_batch.jsonl")
    log_decode_step: int = 0
    seed: int = 42
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: list[str] = field(default_factory=lambda: ["CPU", "GPU"])
    profile_stage: str = "all"
    profile_filename_prefix: str = "pymllm_profile"
    profile_start_step: Optional[int] = None
    profile_steps: int = 1
    skip_warmup: bool = False


@dataclass
class DecodeState:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    mrope_position_deltas: Optional[torch.Tensor] = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer, got {value!r}"
        )
    return parsed


def add_bench_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        "bench_one_batch",
        "Options for the low-level one-batch benchmark.",
    )
    group.add_argument("--run-name", default=BenchArgs.run_name)
    group.add_argument(
        "--batch-size",
        nargs="+",
        type=_positive_int,
        default=[1],
        help="Batch sizes to sweep.",
    )
    group.add_argument(
        "--input-len",
        nargs="+",
        type=_positive_int,
        default=[256, 512, 1024],
        help="Prefill/input lengths to sweep.",
    )
    group.add_argument(
        "--output-len",
        nargs="+",
        type=_positive_int,
        default=[128],
        help="Output lengths to sweep. Matches SGLang's total output token semantics.",
    )
    group.add_argument(
        "--result-filename",
        type=Path,
        default=BenchArgs.result_filename,
        help="JSONL result file. Rows are appended.",
    )
    group.add_argument(
        "--log-decode-step",
        type=_non_negative_int,
        default=0,
        help="Log every N decode steps. 0 disables per-step logging.",
    )
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--profile", action="store_true")
    group.add_argument("--profile-record-shapes", action="store_true")
    group.add_argument(
        "--profile-activities",
        nargs="+",
        choices=["CPU", "GPU"],
        default=["CPU", "GPU"],
    )
    group.add_argument(
        "--profile-stage",
        choices=["all", "prefill", "decode"],
        default="all",
    )
    group.add_argument(
        "--profile-filename-prefix",
        default=BenchArgs.profile_filename_prefix,
    )
    group.add_argument(
        "--profile-start-step",
        type=_non_negative_int,
        default=None,
        help="Decode step index where profiling starts. Defaults to the middle step.",
    )
    group.add_argument(
        "--profile-steps",
        type=_positive_int,
        default=1,
        help="Number of decode steps to profile.",
    )
    group.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the initial non-recorded warmup run.",
    )
    return parser


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m pymllm.bench_one_batch",
        description="Run a SGLang-style direct ModelRunner one-batch benchmark.",
    )
    make_args(parser)
    add_bench_args(parser)
    return parser


def _bench_args_from_namespace(namespace: argparse.Namespace) -> BenchArgs:
    return BenchArgs(
        run_name=namespace.run_name,
        batch_size=list(namespace.batch_size),
        input_len=list(namespace.input_len),
        output_len=list(namespace.output_len),
        result_filename=Path(namespace.result_filename),
        log_decode_step=namespace.log_decode_step,
        seed=namespace.seed,
        profile=namespace.profile,
        profile_record_shapes=namespace.profile_record_shapes,
        profile_activities=list(namespace.profile_activities),
        profile_stage=namespace.profile_stage,
        profile_filename_prefix=namespace.profile_filename_prefix,
        profile_start_step=namespace.profile_start_step,
        profile_steps=namespace.profile_steps,
        skip_warmup=namespace.skip_warmup,
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[GlobalConfig, BenchArgs]:
    parser = make_parser()
    cfg = read_args(argv=argv, parser=parser)
    namespace = parser.parse_args(argv)
    return cfg, _bench_args_from_namespace(namespace)


def generate_settings(args: BenchArgs) -> list[BenchSetting]:
    return [
        BenchSetting(batch_size=batch_size, input_len=input_len, output_len=output_len)
        for batch_size in args.batch_size
        for input_len in args.input_len
        for output_len in args.output_len
    ]


def make_synthetic_input_ids(
    *,
    batch_size: int,
    input_len: int,
    vocab_size: int,
    seed: int,
    device: str | torch.device,
) -> torch.Tensor:
    upper = max(1, min(int(vocab_size or 10000), 10000))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=upper,
        size=(batch_size, input_len),
        generator=generator,
        dtype=torch.int32,
        device="cpu",
    )
    return input_ids.to(device=device)


def summarize_latencies(
    *,
    setting: BenchSetting,
    prefill_latency: float,
    decode_latencies: Sequence[float],
    run_name: str,
    device: str,
    dtype: str,
    cuda_graph: bool,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    median_decode_latency = (
        float(statistics.median(decode_latencies)) if decode_latencies else 0.0
    )
    total_latency = float(prefill_latency + sum(decode_latencies))
    result: dict[str, Any] = {
        "run_name": run_name,
        "batch_size": setting.batch_size,
        "input_len": setting.input_len,
        "output_len": setting.output_len,
        "prefill_latency": float(prefill_latency),
        "prefill_throughput": _safe_div(
            setting.batch_size * setting.input_len,
            prefill_latency,
        ),
        "median_decode_latency": median_decode_latency,
        "median_decode_throughput": _safe_div(
            setting.batch_size,
            median_decode_latency,
        ),
        "total_latency": total_latency,
        "overall_throughput": _safe_div(
            setting.batch_size * (setting.input_len + setting.output_len),
            total_latency,
        ),
        "device": device,
        "dtype": dtype,
        "cuda_graph": cuda_graph,
    }
    if extra:
        result.update(extra)
    return result


def make_profile_trace_path(
    *,
    output_dir: Path,
    prefix: str,
    run_name: str,
    setting: BenchSetting,
    stage: str,
    step: Optional[int] = None,
) -> Path:
    safe_run_name = _sanitize_filename_part(run_name)
    safe_prefix = _sanitize_filename_part(prefix)
    step_part = f"_step{step}" if step is not None else ""
    filename = (
        f"{safe_prefix}_{safe_run_name}_bs{setting.batch_size}"
        f"_in{setting.input_len}_out{setting.output_len}_{stage}"
        f"{step_part}.trace.json"
    )
    return output_dir / filename


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "default"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _sync_device(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        root_logger.setLevel(level)
    logging.getLogger("pymllm").setLevel(level)


def _load_hf_config(cfg: GlobalConfig) -> None:
    if cfg.server.model_path is None:
        raise ValueError("--server.model_path is required")

    from transformers import AutoConfig

    cfg.model.hf_config = AutoConfig.from_pretrained(
        str(cfg.server.model_path),
        trust_remote_code=cfg.server.trust_remote_code,
    )
    logger.info("Loaded model config: %s", cfg.model.hf_config.__class__.__name__)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, sort_keys=True) + "\n")


def _profile_stage_enabled(args: BenchArgs, stage: str) -> bool:
    return args.profile and args.profile_stage in ("all", stage)


def _profiler_activities(args: BenchArgs) -> list[Any]:
    from torch.profiler import ProfilerActivity

    activities = []
    if "CPU" in args.profile_activities:
        activities.append(ProfilerActivity.CPU)
    if "GPU" in args.profile_activities:
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        else:
            logger.warning("GPU profiling requested but CUDA is not available.")
    return activities


@contextmanager
def _maybe_profile(
    *,
    args: BenchArgs,
    setting: BenchSetting,
    stage: str,
    step: Optional[int] = None,
) -> Iterator[None]:
    if not _profile_stage_enabled(args, stage):
        with nullcontext():
            yield
        return

    activities = _profiler_activities(args)
    if not activities:
        with nullcontext():
            yield
        return

    from torch.profiler import profile

    output_dir = Path(os.environ.get("PYMLLM_TORCH_PROFILER_DIR", "/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = make_profile_trace_path(
        output_dir=output_dir,
        prefix=args.profile_filename_prefix,
        run_name=args.run_name,
        setting=setting,
        stage=stage,
        step=step,
    )
    with profile(
        activities=activities,
        record_shapes=args.profile_record_shapes,
    ) as profiler:
        yield
        profiler.step()
    profiler.export_chrome_trace(str(trace_path))
    logger.info("Wrote torch profiler trace: %s", trace_path)


class PymllmBenchRunner:
    def __init__(self, runner: ModelRunner):
        self.runner = runner
        self.device = runner.device

    @classmethod
    def create(cls, cfg: GlobalConfig) -> "PymllmBenchRunner":
        runner = ModelRunner(
            server_config=cfg.server,
            model_config=cfg.model,
            gpu_id=cfg.server.base_gpu_id,
        )
        runner.initialize()
        return cls(runner)

    def clear(self) -> None:
        if self.runner.req_to_token_pool is None:
            raise RuntimeError("ModelRunner req_to_token_pool is not initialized")
        if self.runner.token_to_kv_pool_allocator is None:
            raise RuntimeError(
                "ModelRunner token_to_kv_pool_allocator is not initialized"
            )
        self.runner.req_to_token_pool.clear()
        self.runner.token_to_kv_pool_allocator.clear()

    def extend(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, DecodeState]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch_size, input_len]")

        self._require_initialized()
        batch_size, input_len = input_ids.shape
        req_slots = self.runner.req_to_token_pool.alloc(batch_size)
        if req_slots is None:
            raise RuntimeError(f"Failed to allocate {batch_size} request slots")

        total_tokens = batch_size * input_len
        out_cache_loc = self.runner.token_to_kv_pool_allocator.alloc(total_tokens)
        if out_cache_loc is None:
            for slot in req_slots:
                self.runner.req_to_token_pool.free(slot)
            raise RuntimeError(f"Failed to allocate {total_tokens} KV slots")

        offset = 0
        for slot in req_slots:
            self.runner.req_to_token_pool.write(
                (slot, slice(0, input_len)),
                out_cache_loc[offset : offset + input_len],
            )
            offset += input_len

        req_pool_indices = torch.tensor(
            req_slots, dtype=torch.int64, device=self.device
        )
        if self.runner.gdn_pool is not None:
            self.runner.gdn_pool.reset_states(req_pool_indices)

        seq_lens = torch.full(
            (batch_size,),
            input_len,
            dtype=torch.int32,
            device=self.device,
        )
        extend_seq_lens = torch.full_like(seq_lens, input_len)
        extend_prefix_lens = torch.zeros_like(seq_lens)

        forward_batch = self.runner.prepare_forward_batch_extend(
            input_ids=input_ids.reshape(-1).to(device=self.device, dtype=torch.int32),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            out_cache_loc=out_cache_loc.to(torch.int64),
        )
        logits_output = self.runner.forward(forward_batch)
        next_token_ids = self._sample_greedy(logits_output, forward_batch)
        state = DecodeState(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            mrope_position_deltas=getattr(
                forward_batch, "mrope_position_deltas", None
            ),
        )
        return next_token_ids, state

    def decode(
        self,
        input_ids: torch.Tensor,
        state: DecodeState,
    ) -> tuple[torch.Tensor, DecodeState]:
        self._require_initialized()
        batch_size = int(state.req_pool_indices.shape[0])
        if input_ids.shape != (batch_size,):
            raise ValueError(
                f"decode input_ids must have shape ({batch_size},), got {tuple(input_ids.shape)}"
            )

        out_cache_loc = self.runner.token_to_kv_pool_allocator.alloc(batch_size)
        if out_cache_loc is None:
            raise RuntimeError(f"Failed to allocate {batch_size} decode KV slots")

        seq_lens = state.seq_lens + 1
        for i in range(batch_size):
            slot = int(state.req_pool_indices[i].item())
            write_pos = int(seq_lens[i].item()) - 1
            self.runner.req_to_token_pool.write(
                (slot, slice(write_pos, write_pos + 1)),
                out_cache_loc[i : i + 1],
            )

        forward_batch = self.runner.prepare_forward_batch_decode(
            input_ids=input_ids.to(device=self.device, dtype=torch.int32),
            req_pool_indices=state.req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc.to(torch.int64),
            mrope_position_deltas=state.mrope_position_deltas,
        )
        logits_output = self.runner.forward(forward_batch)
        next_token_ids = self._sample_greedy(logits_output, forward_batch)
        return next_token_ids, DecodeState(
            req_pool_indices=state.req_pool_indices,
            seq_lens=seq_lens,
            mrope_position_deltas=state.mrope_position_deltas,
        )

    def shutdown(self) -> None:
        self.runner.shutdown()

    def _sample_greedy(self, logits_output: Any, forward_batch: Any) -> torch.Tensor:
        temperatures = torch.zeros(
            (forward_batch.batch_size,),
            dtype=torch.float32,
            device=self.device,
        )
        return self.runner.sample(
            logits_output,
            forward_batch,
            temperatures=temperatures,
        ).to(torch.int32)

    def _require_initialized(self) -> None:
        if self.runner.req_to_token_pool is None:
            raise RuntimeError("ModelRunner req_to_token_pool is not initialized")
        if self.runner.token_to_kv_pool_allocator is None:
            raise RuntimeError(
                "ModelRunner token_to_kv_pool_allocator is not initialized"
            )


def _timed_call(
    device: str | torch.device,
    fn: Any,
) -> tuple[float, Any]:
    _sync_device(device)
    tic = time.perf_counter()
    result = fn()
    _sync_device(device)
    return time.perf_counter() - tic, result


def run_single_setting(
    *,
    bench_runner: PymllmBenchRunner,
    args: BenchArgs,
    setting: BenchSetting,
    seed: int,
    record_result: bool,
) -> Optional[dict[str, Any]]:
    bench_runner.clear()
    vocab_size = getattr(bench_runner.runner, "vocab_size", 10000)
    input_ids = make_synthetic_input_ids(
        batch_size=setting.batch_size,
        input_len=setting.input_len,
        vocab_size=vocab_size,
        seed=seed,
        device=bench_runner.device,
    )

    with _maybe_profile(args=args, setting=setting, stage="prefill"):
        prefill_latency, extend_result = _timed_call(
            bench_runner.device,
            lambda: bench_runner.extend(input_ids),
        )
    next_token_ids, state = extend_result

    decode_latencies: list[float] = []
    decode_steps = max(0, setting.output_len - 1)
    profile_start_step = args.profile_start_step
    if profile_start_step is None:
        profile_start_step = decode_steps // 2 if decode_steps else 0
    profile_stop_step = profile_start_step + args.profile_steps

    for step in range(decode_steps):
        should_profile_decode = (
            _profile_stage_enabled(args, "decode")
            and profile_start_step <= step < profile_stop_step
        )
        profile_context = (
            _maybe_profile(args=args, setting=setting, stage="decode", step=step)
            if should_profile_decode
            else nullcontext()
        )
        with profile_context:
            decode_latency, decode_result = _timed_call(
                bench_runner.device,
                lambda: bench_runner.decode(next_token_ids, state),
            )
        next_token_ids, state = decode_result
        decode_latencies.append(decode_latency)

        if args.log_decode_step and (step + 1) % args.log_decode_step == 0:
            logger.info(
                "decode step %d/%d: %.6f s",
                step + 1,
                decode_steps,
                decode_latency,
            )

    if not record_result:
        return None

    return summarize_latencies(
        setting=setting,
        prefill_latency=prefill_latency,
        decode_latencies=decode_latencies,
        run_name=args.run_name,
        device=bench_runner.device,
        dtype=str(bench_runner.runner.dtype),
        cuda_graph=bench_runner.runner.graph_runner is not None,
    )


def run_benchmark(cfg: GlobalConfig, args: BenchArgs) -> list[dict[str, Any]]:
    _load_hf_config(cfg)
    logger.info(
        "bench_one_batch bypasses scheduler; max_prefill_tokens/chunked_prefill_size "
        "do not chunk this benchmark."
    )

    bench_runner = PymllmBenchRunner.create(cfg)
    try:
        settings = generate_settings(args)
        if not args.skip_warmup and settings:
            first = settings[0]
            warmup_setting = BenchSetting(
                batch_size=first.batch_size,
                input_len=first.input_len,
                output_len=min(32, first.output_len),
            )
            logger.info(
                "Warmup: batch_size=%d input_len=%d output_len=%d",
                warmup_setting.batch_size,
                warmup_setting.input_len,
                warmup_setting.output_len,
            )
            run_single_setting(
                bench_runner=bench_runner,
                args=args,
                setting=warmup_setting,
                seed=args.seed,
                record_result=False,
            )

        results: list[dict[str, Any]] = []
        for index, setting in enumerate(settings):
            logger.info(
                "Benchmark: batch_size=%d input_len=%d output_len=%d",
                setting.batch_size,
                setting.input_len,
                setting.output_len,
            )
            result = run_single_setting(
                bench_runner=bench_runner,
                args=args,
                setting=setting,
                seed=args.seed + index,
                record_result=True,
            )
            assert result is not None
            _append_jsonl(args.result_filename, result)
            logger.info("Result: %s", json.dumps(result, sort_keys=True))
            results.append(result)
        return results
    finally:
        bench_runner.shutdown()


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg, args = parse_args(argv)
    _configure_logging(cfg.server.log_level)
    run_benchmark(cfg, args)


if __name__ == "__main__":
    main()
