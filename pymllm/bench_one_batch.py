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
import sys
import time
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
    image_path: Optional[Path] = None
    prompt: str = "Describe this image."
    input_len_was_provided: bool = False
    correctness_test: bool = False


@dataclass
class DecodeState:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    mrope_position_deltas: Optional[torch.Tensor] = None


@dataclass
class ExtendResult:
    next_token_ids: torch.Tensor
    state: DecodeState
    vit_prefill_ms: Optional[float] = None
    vit_prefill_tokens: Optional[int] = None
    vit_prefill_tps: Optional[float] = None

    def __iter__(self) -> Iterator[Any]:
        # Preserve the old ``next_token_ids, state = extend(...)`` call pattern.
        yield self.next_token_ids
        yield self.state


@dataclass
class MultimodalBenchInput:
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    vit_prefill_tokens: int


@dataclass
class MultimodalProcessorBundle:
    processor_output: Any
    pad_token_id: int


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
        choices=["CPU", "GPU", "CUDA_PROFILER"],
        default=["CPU", "GPU"],
        help=(
            "CPU/GPU use the torch profiler; CUDA_PROFILER drives nsys via "
            "cudaProfilerStart/Stop (use with nsys --capture-range=cudaProfilerApi)."
        ),
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
    group.add_argument(
        "--image",
        "--image-path",
        dest="image_path",
        type=Path,
        default=None,
        help=(
            "Optional image path for multimodal benchmark mode. When set, "
            "omitted --input-len uses the processed prompt length; explicit "
            "--input-len sweeps target total multimodal prefill length "
            "(image placeholder tokens + text tokens)."
        ),
    )
    group.add_argument(
        "--prompt",
        default=BenchArgs.prompt,
        help="Prompt text used with --image in multimodal benchmark mode.",
    )
    group.add_argument(
        "--correct",
        "--correctness-test",
        dest="correctness_test",
        action="store_true",
        help=(
            "Run a single-stage smoke correctness check (encode real prompts, "
            "prefill, greedy decode, print decoded text) instead of the "
            "latency benchmark."
        ),
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


def _argv_has_option(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _bench_args_from_namespace(
    namespace: argparse.Namespace,
    *,
    input_len_was_provided: bool = False,
) -> BenchArgs:
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
        image_path=namespace.image_path,
        prompt=namespace.prompt,
        input_len_was_provided=input_len_was_provided,
        correctness_test=namespace.correctness_test,
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[GlobalConfig, BenchArgs]:
    parser = make_parser()
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    cfg = read_args(argv=cli_argv, parser=parser)
    namespace = parser.parse_args(cli_argv)
    return cfg, _bench_args_from_namespace(
        namespace,
        input_len_was_provided=_argv_has_option(cli_argv, "--input-len"),
    )


def generate_settings(args: BenchArgs) -> list[BenchSetting]:
    input_lens = (
        [0]
        if args.image_path is not None and not args.input_len_was_provided
        else args.input_len
    )
    return [
        BenchSetting(batch_size=batch_size, input_len=input_len, output_len=output_len)
        for batch_size in args.batch_size
        for input_len in input_lens
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


def make_vit_prefill_metrics(
    *,
    vit_prefill_ms: float,
    vit_prefill_tokens: int,
) -> dict[str, Any]:
    latency = float(vit_prefill_ms) / 1000.0
    throughput = _safe_div(float(vit_prefill_tokens), latency)
    return {
        "vit_prefill_latency": latency,
        "vit_prefill_ms": float(vit_prefill_ms),
        "vit_prefill_tokens": int(vit_prefill_tokens),
        "vit_prefill_throughput": throughput,
        "vit_prefill_tps": throughput,
    }


def make_multimodal_prefill_metrics(
    *,
    prefill_latency: float,
    batch_size: int,
    input_len: int,
) -> dict[str, Any]:
    tokens = int(batch_size) * int(input_len)
    throughput = _safe_div(float(tokens), float(prefill_latency))
    return {
        "multimodal_prefill_latency": float(prefill_latency),
        "multimodal_prefill_ms": float(prefill_latency) * 1000.0,
        "multimodal_prefill_tokens": tokens,
        "multimodal_prefill_throughput": throughput,
        "multimodal_prefill_tps": throughput,
    }


def _get_processor_value(processor_output: Any, key: str) -> Any:
    if hasattr(processor_output, "get"):
        return processor_output.get(key)
    return getattr(processor_output, key, None)


def _resize_multimodal_input_ids(
    input_ids: torch.Tensor,
    *,
    target_input_len: int,
    image_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    if int(pad_token_id) == int(image_token_id):
        pad_token_id = 0 if int(image_token_id) != 0 else 1
    if target_input_len <= 0:
        raise ValueError(
            f"target_input_len must be positive, got {target_input_len}"
        )
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "bench_one_batch multimodal resize expects input_ids shape [1, seq_len], "
            f"got {tuple(input_ids.shape)}"
        )

    seq = input_ids[0]
    image_mask = seq == image_token_id
    image_token_count = int(image_mask.sum().item())
    if target_input_len < image_token_count:
        raise ValueError(
            "target_input_len must be at least the number of image tokens "
            f"({image_token_count}), got {target_input_len}"
        )
    if int(seq.numel()) == target_input_len:
        return input_ids

    text_budget = target_input_len - image_token_count
    resized_tokens: list[int] = []
    kept_text = 0
    for token in seq.tolist():
        token_id = int(token)
        if token_id == image_token_id:
            resized_tokens.append(token_id)
        elif kept_text < text_budget:
            resized_tokens.append(token_id)
            kept_text += 1
        if len(resized_tokens) == target_input_len:
            break

    if len(resized_tokens) < target_input_len:
        resized_tokens.extend([int(pad_token_id)] * (target_input_len - len(resized_tokens)))

    return torch.tensor(
        [resized_tokens],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )


def make_multimodal_bench_input_from_processor_output(
    processor_output: Any,
    *,
    batch_size: int,
    image_token_id: int,
    device: str | torch.device,
    target_input_len: Optional[int] = None,
    pad_token_id: int = 0,
) -> MultimodalBenchInput:
    input_ids = _get_processor_value(processor_output, "input_ids")
    pixel_values = _get_processor_value(processor_output, "pixel_values")
    image_grid_thw = _get_processor_value(processor_output, "image_grid_thw")

    if input_ids is None:
        raise ValueError("Multimodal processor output does not contain input_ids")
    if pixel_values is None:
        raise ValueError("Multimodal processor output does not contain pixel_values")
    if image_grid_thw is None:
        raise ValueError(
            "Multimodal processor output does not contain image_grid_thw"
        )

    input_ids_t = torch.as_tensor(input_ids)
    if input_ids_t.dim() == 1:
        input_ids_t = input_ids_t.unsqueeze(0)
    if input_ids_t.shape[0] != 1:
        raise ValueError(
            "bench_one_batch multimodal mode expects one processed prompt before "
            f"batch repetition, got batch dimension {input_ids_t.shape[0]}"
        )
    if target_input_len is not None:
        input_ids_t = _resize_multimodal_input_ids(
            input_ids_t,
            target_input_len=target_input_len,
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
        )

    input_ids_t = input_ids_t.repeat(batch_size, 1).to(
        device=device, dtype=torch.int32
    )
    pixel_values_t = torch.as_tensor(pixel_values)
    pixel_values_t = pixel_values_t.repeat(
        (batch_size,) + (1,) * (pixel_values_t.dim() - 1)
    )
    image_grid_thw_t = torch.as_tensor(image_grid_thw)
    if image_grid_thw_t.dim() == 1:
        image_grid_thw_t = image_grid_thw_t.unsqueeze(0)
    image_grid_thw_t = image_grid_thw_t.repeat(batch_size, 1).to(
        device=device, dtype=torch.int64
    )

    return MultimodalBenchInput(
        input_ids=input_ids_t,
        pixel_values=pixel_values_t.to(device=device),
        image_grid_thw=image_grid_thw_t,
        vit_prefill_tokens=int((input_ids_t == image_token_id).sum().item()),
    )


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
        f"{step_part}.trace.json.gz"
    )
    return output_dir / filename


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "default"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _max_batch_size_for(runner: Any, input_len: int, output_len: int) -> int:
    """SGLang-style capacity bound on the static batch.

    Mirrors ``ModelRunner.max_batch_size`` in SGLang's bench_one_batch:
    ``max_total_num_tokens // (input_len + output_len)``.  Used to skip
    settings the KV pool cannot hold instead of failing mid-run on alloc.
    """
    total = int(getattr(runner, "max_total_num_tokens", 0) or 0)
    denom = int(input_len) + int(output_len)
    if denom <= 0:
        return 0
    return total // denom


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


def _extract_image_token_id(hf_config: Any) -> int:
    image_token_id = getattr(hf_config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Model config does not define image_token_id")
    return int(image_token_id)


def _render_multimodal_prompt(
    processor: Any,
    *,
    prompt: str,
    image_path: Path,
) -> str:
    if not hasattr(processor, "apply_chat_template"):
        return prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        logger.warning("Processor chat template failed, using raw prompt: %s", exc)
        return prompt
    if isinstance(rendered, list):
        if not rendered:
            return prompt
        return str(rendered[0])
    return str(rendered)


def _make_multimodal_processor_output(
    *,
    cfg: GlobalConfig,
    prompt: str,
    image_path: Path,
) -> MultimodalProcessorBundle:
    if cfg.server.tokenizer_path is None:
        raise ValueError("--server.tokenizer_path or --server.model_path is required")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(cfg.server.tokenizer_path),
        trust_remote_code=cfg.server.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    image = Image.open(image_path).convert("RGB")
    text = _render_multimodal_prompt(
        processor,
        prompt=prompt,
        image_path=image_path,
    )
    return MultimodalProcessorBundle(
        processor_output=processor(images=[image], text=[text], return_tensors="pt"),
        pad_token_id=int(pad_token_id),
    )


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


def _resolve_profile_output_dir() -> Path:
    output_dir = Path(os.environ.get("PYMLLM_TORCH_PROFILER_DIR", "/tmp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _start_profile(args: BenchArgs, trace_path: Path) -> Any:
    """Start profiling and return a handle.

    Mirrors SGLang's ``start_profile``: ``CUDA_PROFILER`` drives nsys via
    ``cudaProfilerStart``; otherwise a torch profiler with ``with_stack=True``
    is started so kernels can be mapped back to Python source.  Returns
    ``"cuda"`` for the nsys path, the profiler object for the torch path, or
    ``None`` when no activity is available.  ``trace_path`` is accepted for
    symmetry with ``_stop_profile`` (the torch path saves it on stop).
    """
    if "CUDA_PROFILER" in args.profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            logger.info("CUDA profiler started (nsys will begin capturing).")
        except Exception as exc:  # pragma: no cover - depends on nsys runtime
            logger.warning("Failed to start CUDA profiler: %s", exc)
        return "cuda"

    activities = _profiler_activities(args)
    if not activities:
        return None

    from torch.profiler import profile

    profiler = profile(
        activities=activities,
        with_stack=True,
        record_shapes=args.profile_record_shapes,
    )
    profiler.start()
    return profiler


def _stop_profile(handle: Any, args: BenchArgs, trace_path: Path, stage: str) -> None:
    """Stop profiling and, for the torch path, save the chrome trace.

    Mirrors SGLang's ``stop_profile``, including printing the key_averages
    table.  The trace is written as ``.trace.json.gz`` (torch gzips when the
    filename ends with ``.gz``).
    """
    if handle is None:
        return
    if handle == "cuda":
        try:
            torch.cuda.cudart().cudaProfilerStop()
            logger.info("CUDA profiler stopped for %s (nsys dumps traces).", stage)
        except Exception as exc:  # pragma: no cover - depends on nsys runtime
            logger.warning("Failed to stop CUDA profiler: %s", exc)
        return

    handle.stop()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    handle.export_chrome_trace(str(trace_path))
    try:
        sort_key = (
            "self_cuda_time_total"
            if torch.cuda.is_available()
            else "self_cpu_time_total"
        )
        print(
            handle.key_averages(
                group_by_input_shape=args.profile_record_shapes
            ).table(sort_by=sort_key)
        )
    except Exception as exc:
        logger.warning("Failed to print profiler key_averages: %s", exc)
    logger.info("Wrote torch profiler trace for %s: %s", stage, trace_path)


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

    def extend(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        benchmark_vision_timing: bool = False,
    ) -> ExtendResult:
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
        if pixel_values is not None:
            forward_batch.pixel_values = pixel_values.to(device=self.device)
        if image_grid_thw is not None:
            forward_batch.image_grid_thw = image_grid_thw.to(device=self.device)
        forward_batch.benchmark_vision_timing = benchmark_vision_timing

        logits_output = self.runner.forward(forward_batch)
        next_token_ids = self._sample_greedy(logits_output, forward_batch)
        state = DecodeState(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            mrope_position_deltas=getattr(
                forward_batch, "mrope_position_deltas", None
            ),
        )
        vit_prefill_ms = getattr(forward_batch, "vit_prefill_ms", None)
        vit_prefill_tokens = getattr(forward_batch, "vit_prefill_tokens", None)
        vit_prefill_tps = getattr(forward_batch, "vit_prefill_tps", None)
        return ExtendResult(
            next_token_ids=next_token_ids,
            state=state,
            vit_prefill_ms=(
                float(vit_prefill_ms) if vit_prefill_ms is not None else None
            ),
            vit_prefill_tokens=(
                int(vit_prefill_tokens) if vit_prefill_tokens is not None else None
            ),
            vit_prefill_tps=(
                float(vit_prefill_tps) if vit_prefill_tps is not None else None
            ),
        )

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
        # Tensorized KV-mapping write.  The production decode path
        # (orchestrator/model_runner_process.py) keeps slot/write_pos as plain
        # CPU bookkeeping and never does a per-request CUDA ``.item()`` sync.
        # Doing per-request ``.item()`` here would add 2*batch_size CPU-GPU
        # syncs inside the timed decode region that SGLang does not have,
        # biasing decode latency once batch_size > 1.  Write all rows at once:
        # req_to_token[req_pool_indices, seq_lens - 1] = out_cache_loc.
        write_positions = (seq_lens - 1).to(torch.int64)
        self.runner.req_to_token_pool.write(
            (state.req_pool_indices, write_positions),
            out_cache_loc.to(torch.int32),
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
            is_all_greedy=True,
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
    multimodal_processor_bundle: Optional[MultimodalProcessorBundle] = None,
    allow_profile: bool = True,
) -> Optional[dict[str, Any]]:
    bench_runner.clear()
    vocab_size = getattr(bench_runner.runner, "vocab_size", 10000)
    mm_input = None
    effective_setting = setting
    if multimodal_processor_bundle is None:
        input_ids = make_synthetic_input_ids(
            batch_size=setting.batch_size,
            input_len=setting.input_len,
            vocab_size=vocab_size,
            seed=seed,
            device=bench_runner.device,
        )
    else:
        hf_config = bench_runner.runner.model_config.hf_config
        image_token_id = _extract_image_token_id(hf_config)
        mm_input = make_multimodal_bench_input_from_processor_output(
            multimodal_processor_bundle.processor_output,
            batch_size=setting.batch_size,
            image_token_id=image_token_id,
            device=bench_runner.device,
            target_input_len=(
                setting.input_len
                if args.input_len_was_provided
                else None
            ),
            pad_token_id=multimodal_processor_bundle.pad_token_id,
        )
        input_ids = mm_input.input_ids
        effective_setting = BenchSetting(
            batch_size=setting.batch_size,
            input_len=int(input_ids.shape[1]),
            output_len=setting.output_len,
        )

    max_bs = _max_batch_size_for(
        bench_runner.runner,
        effective_setting.input_len,
        effective_setting.output_len,
    )
    if effective_setting.batch_size > max_bs:
        logger.info(
            "skipping (batch_size=%d, input_len=%d, output_len=%d): exceeds max "
            "batch size %d (max_total_num_tokens=%d). SGLang-style skip.",
            effective_setting.batch_size,
            effective_setting.input_len,
            effective_setting.output_len,
            max_bs,
            int(getattr(bench_runner.runner, "max_total_num_tokens", 0) or 0),
        )
        return None

    prefill_profile = allow_profile and _profile_stage_enabled(args, "prefill")
    prefill_trace: Optional[Path] = None
    prefill_handle: Any = None
    if prefill_profile:
        prefill_trace = make_profile_trace_path(
            output_dir=_resolve_profile_output_dir(),
            prefix=args.profile_filename_prefix,
            run_name=args.run_name,
            setting=effective_setting,
            stage="prefill",
        )
        prefill_handle = _start_profile(args, prefill_trace)
    prefill_latency, extend_result = _timed_call(
        bench_runner.device,
        lambda: bench_runner.extend(
            input_ids,
            pixel_values=mm_input.pixel_values if mm_input is not None else None,
            image_grid_thw=(
                mm_input.image_grid_thw if mm_input is not None else None
            ),
            benchmark_vision_timing=mm_input is not None,
        ),
    )
    if prefill_profile:
        _stop_profile(prefill_handle, args, prefill_trace, "prefill")
    next_token_ids, state = extend_result

    decode_latencies: list[float] = []
    decode_steps = max(0, setting.output_len - 1)
    decode_profile = allow_profile and _profile_stage_enabled(args, "decode")
    profile_start_step = args.profile_start_step
    if profile_start_step is None:
        # Align SGLang: default to output_len // 2.
        profile_start_step = effective_setting.output_len // 2
    profile_stop_step = profile_start_step + args.profile_steps
    decode_trace: Optional[Path] = None
    decode_handle: Any = None

    # One continuous profiler spans [profile_start_step, profile_stop_step),
    # producing a single decode trace, matching SGLang (not one file per step).
    for step in range(decode_steps):
        if decode_profile and step == profile_start_step:
            decode_trace = make_profile_trace_path(
                output_dir=_resolve_profile_output_dir(),
                prefix=args.profile_filename_prefix,
                run_name=args.run_name,
                setting=effective_setting,
                stage="decode",
            )
            decode_handle = _start_profile(args, decode_trace)

        decode_latency, decode_result = _timed_call(
            bench_runner.device,
            lambda: bench_runner.decode(next_token_ids, state),
        )

        if decode_handle is not None and step >= profile_stop_step - 1:
            _stop_profile(decode_handle, args, decode_trace, "decode")
            decode_handle = None

        next_token_ids, state = decode_result
        decode_latencies.append(decode_latency)

        if args.log_decode_step and (step + 1) % args.log_decode_step == 0:
            logger.info(
                "decode step %d/%d: %.6f s",
                step + 1,
                decode_steps,
                decode_latency,
            )

    # Save if the requested profile window ran past the final decode step.
    if decode_handle is not None:
        _stop_profile(decode_handle, args, decode_trace, "decode")
        decode_handle = None

    if not record_result:
        return None

    extra_metrics = None
    if mm_input is not None:
        extra_metrics = make_multimodal_prefill_metrics(
            prefill_latency=prefill_latency,
            batch_size=effective_setting.batch_size,
            input_len=effective_setting.input_len,
        )
        if (
            extend_result.vit_prefill_ms is not None
            and extend_result.vit_prefill_tokens is not None
        ):
            extra_metrics.update(
                make_vit_prefill_metrics(
                    vit_prefill_ms=extend_result.vit_prefill_ms,
                    vit_prefill_tokens=extend_result.vit_prefill_tokens,
                )
            )

    return summarize_latencies(
        setting=effective_setting,
        prefill_latency=prefill_latency,
        decode_latencies=decode_latencies,
        run_name=args.run_name,
        device=bench_runner.device,
        dtype=str(bench_runner.runner.dtype),
        cuda_graph=bench_runner.runner.graph_runner is not None,
        extra=extra_metrics,
    )


def _align_runner_capacity_with_batch_sizes(
    cfg: GlobalConfig, batch_sizes: Sequence[int]
) -> None:
    """Ensure the runner can hold and CUDA-graph-capture the largest batch.

    Mirrors SGLang ``main()`` which sets
    ``server_args.cuda_graph_max_bs = max(bench_args.batch_size)``.  In pymllm
    the CUDA graph capture batch sizes are derived from
    ``ModelRunner.max_running_requests`` (see ``CudaGraphRunner``), which also
    sizes ``req_to_token_pool``.  Without this, sweeping a batch size larger
    than the configured capture set makes decode silently fall off the graph
    path and run eager, biasing decode latency versus SGLang.
    """
    if not batch_sizes:
        return
    requested = max(batch_sizes)
    configured = cfg.server.max_running_requests
    if configured is None or configured < requested:
        cfg.server.max_running_requests = requested
        logger.info(
            "Raised max_running_requests to %d to cover bench batch sizes "
            "(SGLang cuda_graph_max_bs alignment).",
            requested,
        )


def run_benchmark(cfg: GlobalConfig, args: BenchArgs) -> list[dict[str, Any]]:
    _load_hf_config(cfg)
    logger.info(
        "bench_one_batch bypasses scheduler; max_prefill_tokens/chunked_prefill_size "
        "do not chunk this benchmark."
    )

    _align_runner_capacity_with_batch_sizes(cfg, args.batch_size)
    bench_runner = PymllmBenchRunner.create(cfg)
    try:
        settings = generate_settings(args)
        multimodal_processor_bundle = None
        if args.image_path is not None:
            multimodal_processor_bundle = _make_multimodal_processor_output(
                cfg=cfg,
                prompt=args.prompt,
                image_path=args.image_path,
            )
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
                multimodal_processor_bundle=multimodal_processor_bundle,
                allow_profile=False,
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
                multimodal_processor_bundle=multimodal_processor_bundle,
            )
            if result is None:
                # Setting skipped (e.g. exceeds KV pool capacity); do not record.
                continue
            _append_jsonl(args.result_filename, result)
            logger.info("Result: %s", json.dumps(result, sort_keys=True))
            results.append(result)
        return results
    finally:
        bench_runner.shutdown()


DEFAULT_CORRECTNESS_PROMPTS = (
    "The capital of France is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
)


def _load_tokenizer(cfg: GlobalConfig) -> Any:
    if cfg.server.tokenizer_path is None:
        raise ValueError("--server.tokenizer_path or --server.model_path is required")

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(cfg.server.tokenizer_path),
        trust_remote_code=cfg.server.trust_remote_code,
    )


def correctness_test(
    bench_runner: PymllmBenchRunner,
    cfg: GlobalConfig,
    args: BenchArgs,
) -> None:
    """Single-stage smoke correctness check.

    Encode a real prompt, run one full prefill at batch_size=1, greedy-decode
    ``output_len`` tokens, and print the decoded text.  Unlike SGLang's
    ``--correct`` (which exercises a cut_len two-stage prefill to test prefix-KV
    reuse), this runs each prompt as a single full prefill.  Greedy decoding
    makes the per-prompt output identical to SGLang's batched path.  The cut_len
    two-stage variant can be layered on later: ``prepare_forward_batch_extend``
    already accepts ``extend_prefix_lens > 0`` and ``req_to_token_pool.write``
    can pre-populate prefix KV indices.
    """
    tokenizer = _load_tokenizer(cfg)
    output_len = args.output_len[0]
    prompts = list(DEFAULT_CORRECTNESS_PROMPTS)

    for idx, prompt in enumerate(prompts):
        token_ids = list(tokenizer.encode(prompt))
        if not token_ids:
            logger.warning("Prompt %d encoded to an empty token list, skipping.", idx)
            continue
        input_ids = torch.tensor(
            [token_ids], dtype=torch.int32, device=bench_runner.device
        )

        bench_runner.clear()
        next_token_ids, state = bench_runner.extend(input_ids)
        output_ids = token_ids + [int(next_token_ids[0].item())]
        for _ in range(max(0, output_len - 1)):
            next_token_ids, state = bench_runner.decode(next_token_ids, state)
            output_ids.append(int(next_token_ids[0].item()))

        print(f"========== Prompt {idx} ==========")
        print(tokenizer.decode(output_ids), "\n")


def run_correctness(cfg: GlobalConfig, args: BenchArgs) -> None:
    _load_hf_config(cfg)
    bench_runner = PymllmBenchRunner.create(cfg)
    try:
        correctness_test(bench_runner, cfg, args)
    finally:
        bench_runner.shutdown()


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg, args = parse_args(argv)
    _configure_logging(cfg.server.log_level)
    if args.correctness_test:
        run_correctness(cfg, args)
    else:
        run_benchmark(cfg, args)


if __name__ == "__main__":
    main()
