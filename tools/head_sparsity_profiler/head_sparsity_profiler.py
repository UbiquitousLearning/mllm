#!/usr/bin/env python3
"""Profile per-head retention ratios with the ShadowNPU AE procedure.

The expensive ``collect`` command imports PyTorch/Transformers lazily.  The
``convert`` command and its tests only require the Python standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
AE_ARTIFACT = "https://doi.org/10.5281/zenodo.19555734"
DEFAULT_ARTIFACT_MEMBER = "ShadowNPU/data/WikiText/wiki.valid.txt"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _finite_positive(value: str, location: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{location} must be finite and positive, got {value!r}")
    return result


def read_ae_results(
    head_path: Path,
    layer_path: Path,
    requested_model: str | None = None,
) -> tuple[str, list[list[float]], list[float]]:
    """Read and validate the AE's raw head/layer text formats."""

    head_values: dict[tuple[int, int], float] = {}
    layer_values: dict[int, float] = {}
    models: set[str] = set()

    for line_number, line in enumerate(
        head_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 4:
            raise ValueError(
                f"{head_path}:{line_number}: expected MODEL LAYER HEAD PERPLEXITY"
            )
        model, layer_text, head_text, perplexity_text = fields
        try:
            layer = int(layer_text)
            head = int(head_text)
        except ValueError as error:
            raise ValueError(
                f"{head_path}:{line_number}: layer/head must be integers"
            ) from error
        if layer < 0 or head < 0:
            raise ValueError(f"{head_path}:{line_number}: negative layer/head index")
        key = (layer, head)
        if key in head_values:
            raise ValueError(f"{head_path}:{line_number}: duplicate head {key}")
        head_values[key] = _finite_positive(
            perplexity_text, f"{head_path}:{line_number}: perplexity"
        )
        models.add(model)

    for line_number, line in enumerate(
        layer_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 3:
            raise ValueError(
                f"{layer_path}:{line_number}: expected MODEL LAYER PERPLEXITY"
            )
        model, layer_text, perplexity_text = fields
        try:
            layer = int(layer_text)
        except ValueError as error:
            raise ValueError(
                f"{layer_path}:{line_number}: layer must be an integer"
            ) from error
        if layer < 0:
            raise ValueError(f"{layer_path}:{line_number}: negative layer index")
        if layer in layer_values:
            raise ValueError(f"{layer_path}:{line_number}: duplicate layer {layer}")
        layer_values[layer] = _finite_positive(
            perplexity_text, f"{layer_path}:{line_number}: perplexity"
        )
        models.add(model)

    if not head_values or not layer_values:
        raise ValueError("head and layer result files must both be non-empty")
    if requested_model is not None:
        if any(model != requested_model for model in models):
            raise ValueError(
                f"raw results contain {sorted(models)}, expected only {requested_model!r}"
            )
        model_name = requested_model
    elif len(models) != 1:
        raise ValueError(f"raw results contain multiple model labels: {sorted(models)}")
    else:
        model_name = next(iter(models))

    layer_count = max(layer_values) + 1
    expected_layers = set(range(layer_count))
    if set(layer_values) != expected_layers:
        missing = sorted(expected_layers - set(layer_values))
        raise ValueError(f"layer results are not contiguous; missing {missing}")
    head_count = max(head for _, head in head_values) + 1
    expected_heads = {
        (layer, head)
        for layer in range(layer_count)
        for head in range(head_count)
    }
    if set(head_values) != expected_heads:
        missing = sorted(expected_heads - set(head_values))
        extra = sorted(set(head_values) - expected_heads)
        raise ValueError(
            f"head result grid is incomplete; missing={missing[:8]}, extra={extra[:8]}"
        )

    heads = [
        [head_values[layer, head] for head in range(head_count)]
        for layer in range(layer_count)
    ]
    layers = [layer_values[layer] for layer in range(layer_count)]
    return model_name, heads, layers


def _validate_measurements(value: dict[str, Any]) -> tuple[str, list[list[float]], list[float]]:
    try:
        model_name = str(value["model"])
        raw_heads = value["head_perplexity"]
        raw_layers = value["layer_perplexity"]
    except (KeyError, TypeError) as error:
        raise ValueError("measurement JSON is missing model/perplexity fields") from error
    if not model_name or not isinstance(raw_heads, list) or not isinstance(raw_layers, list):
        raise ValueError("invalid measurement JSON structure")
    if not raw_heads or len(raw_heads) != len(raw_layers):
        raise ValueError("head/layer measurement counts do not match")
    head_count = len(raw_heads[0]) if isinstance(raw_heads[0], list) else 0
    if head_count == 0:
        raise ValueError("head measurement rows must be non-empty")
    heads: list[list[float]] = []
    for layer, row in enumerate(raw_heads):
        if not isinstance(row, list) or len(row) != head_count:
            raise ValueError(f"head measurement row {layer} has the wrong width")
        heads.append(
            [_finite_positive(str(item), f"head[{layer}][{head}]") for head, item in enumerate(row)]
        )
    layers = [
        _finite_positive(str(item), f"layer[{layer}]")
        for layer, item in enumerate(raw_layers)
    ]
    return model_name, heads, layers


def allocate_ae_retentions(
    head_perplexity: Sequence[Sequence[float]],
    layer_perplexity: Sequence[float],
    average_retention: float = 0.2,
    clamping_threshold: float = 102400.0,
) -> tuple[list[list[float]], list[list[float]], float]:
    """Reproduce ``offline/get_ratios.py`` from the AE artifact.

    This deliberately preserves the artifact's row-major, sequential overflow
    redistribution instead of replacing it with a mathematically cleaner
    active-set allocator.
    """

    if not math.isfinite(average_retention) or not 0.0 < average_retention <= 1.0:
        raise ValueError("average retention must be in (0, 1]")
    if not math.isfinite(clamping_threshold) or clamping_threshold <= 0.0:
        raise ValueError("clamping threshold must be finite and positive")
    layer_count = len(layer_perplexity)
    if layer_count == 0 or len(head_perplexity) != layer_count:
        raise ValueError("head and layer measurements must have equal non-zero layers")
    head_count = len(head_perplexity[0])
    if head_count == 0 or any(len(row) != head_count for row in head_perplexity):
        raise ValueError("all head measurement rows must have equal non-zero width")

    importance: list[list[float]] = []
    for layer, row in enumerate(head_perplexity):
        layer_ppl = float(layer_perplexity[layer])
        if not math.isfinite(layer_ppl) or layer_ppl <= 0.0:
            raise ValueError(f"layer perplexity {layer} must be finite and positive")
        values: list[float] = []
        for head, head_ppl in enumerate(row):
            head_ppl = float(head_ppl)
            if not math.isfinite(head_ppl) or head_ppl <= 0.0:
                raise ValueError(
                    f"head perplexity ({layer}, {head}) must be finite and positive"
                )
            values.append(min(head_ppl * layer_ppl, clamping_threshold))
        importance.append(values)

    # Keep the exact accumulation order of the artifact's ``sum_list``.  A
    # nested ``sum(sum(row) ...)`` changes the last few float bits and makes
    # regenerated paper profiles differ byte-for-byte.
    total_importance = 0.0
    for row in importance:
        for value in row:
            total_importance += value
    if total_importance <= 0.0 or not math.isfinite(total_importance):
        raise ValueError("combined importance sum must be finite and positive")
    budget = average_retention * head_count * layer_count
    retentions = [
        [
            average_retention
            * head_count
            * layer_count
            * value
            / total_importance
            for value in row
        ]
        for row in importance
    ]

    overflow = 0.0
    overflowed_count = 0
    for layer in range(layer_count):
        for head in range(head_count):
            if retentions[layer][head] > 1.0:
                overflow += retentions[layer][head] - 1.0
                retentions[layer][head] = 1.0
                overflowed_count += 1

    remaining_count = layer_count * head_count - overflowed_count
    for layer in range(layer_count):
        for head in range(head_count):
            if retentions[layer][head] < 1.0:
                addition = overflow / remaining_count
                remaining_count -= 1
                if retentions[layer][head] + addition <= 1.0:
                    retentions[layer][head] += addition
                    overflow -= addition
                else:
                    # Preserve the AE implementation literally.  It assigns
                    # one here before evaluating the subtraction expression.
                    retentions[layer][head] = 1.0
                    overflow -= 1.0 - retentions[layer][head]

    return retentions, importance, overflow


def format_retention_profile(retentions: Sequence[Sequence[float]]) -> str:
    """Use the same Python-float text representation and trailing spaces as AE."""

    return "".join("".join(f"{float(value)} " for value in row) + "\n" for row in retentions)


def _make_profile_report(
    model_name: str,
    heads: Sequence[Sequence[float]],
    layers: Sequence[float],
    average_retention: float,
    clamping_threshold: float,
) -> tuple[str, dict[str, Any]]:
    retentions, importance, leftover = allocate_ae_retentions(
        heads, layers, average_retention, clamping_threshold
    )
    actual_average = sum(sum(row) for row in retentions) / sum(
        len(row) for row in retentions
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "ShadowNPU-AE-offline-get_ratios",
        "artifact": AE_ARTIFACT,
        "model": model_name,
        "layers": len(retentions),
        "heads": len(retentions[0]),
        "requested_average_retention": average_retention,
        "actual_average_retention": actual_average,
        "global_sparsity": 1.0 - actual_average,
        "clamping_threshold": clamping_threshold,
        "unredistributed_overflow": leftover,
        "dense_heads": sum(value == 1.0 for row in retentions for value in row),
        "importance": importance,
        "retentions": retentions,
    }
    return format_retention_profile(retentions), report


def _load_calibration_text(args: argparse.Namespace) -> str:
    if args.calibration_text is not None:
        return args.calibration_text.read_text(encoding="utf-8")
    process = subprocess.run(
        ["unzip", "-p", str(args.artifact_zip), args.artifact_member],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0 or not process.stdout:
        raise RuntimeError(
            f"could not read {args.artifact_member!r} from {args.artifact_zip}: "
            f"{process.stderr.decode(errors='replace')}"
        )
    return process.stdout.decode("utf-8")


def build_calibration_windows(
    tokenizer: Any,
    text: str,
    context_length: int,
    max_samples: int = 0,
) -> list[list[int]]:
    """Tokenize WikiText exactly like the AE and return complete windows."""

    if context_length <= 1 or max_samples < 0:
        raise ValueError("context length must be >1 and max samples must be >=0")
    token_ids: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) <= 10:
            continue
        encoded = tokenizer(line)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("tokenizer returned more than one sequence for a line")
            encoded = encoded[0]
        token_ids.extend(int(token) for token in encoded)

    complete = len(token_ids) // context_length
    if max_samples:
        complete = min(complete, max_samples)
    if complete == 0:
        raise ValueError("calibration corpus contains no complete context window")
    return [
        token_ids[start : start + context_length]
        for start in range(0, complete * context_length, context_length)
    ]


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _decoder_layers(model: Any) -> Sequence[Any]:
    candidates = (
        ("model", "layers"),
        ("transformer", "layers"),
        ("transformer", "h"),
    )
    for owner_name, layers_name in candidates:
        owner = getattr(model, owner_name, None)
        layers = getattr(owner, layers_name, None) if owner is not None else None
        if layers is not None and len(layers):
            return layers
    raise RuntimeError("could not locate decoder layers on the loaded model")


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _torch_dtype(args.dtype)
    common = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, **common)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            **common,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **common,
        )
        model.config._attn_implementation = "eager"

    if args.state_dict is not None:
        state = torch.load(args.state_dict, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state, strict=True, assign=True)
        except TypeError:
            model.load_state_dict(state, strict=True)
        del state

    # The AE loads the bfloat16 model on CUDA before fake-quantizing weights.
    # Preserve that operation order because CPU/GPU low-precision rounding can
    # otherwise produce a slightly different ablation profile.
    model.to(device=args.device, dtype=dtype)

    if args.activation_scales is not None:
        from ae_fake_quant import get_ae_clip_and_scale, quantize_model_ae

        activation_distribution = json.loads(
            args.activation_scales.read_text(encoding="utf-8")
        )
        activation_scales, clipping, summary = get_ae_clip_and_scale(
            activation_distribution, args.clip_threshold
        )
        replacements = quantize_model_ae(
            model, _decoder_layers(model), activation_scales, clipping
        )
        print(
            "AE fake W8A8 quantization: "
            f"layers={replacements // 7}, projections={replacements}, "
            f"clip inputs={summary['clip_input_num']}, "
            f"clip outputs={summary['clip_output_num']}",
            flush=True,
        )

    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = "eager"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _repeat_kv(hidden: Any, groups: int) -> Any:
    if groups == 1:
        return hidden
    return hidden.repeat_interleave(groups, dim=1)


def _ae_compressed_qk(query: Any, key: Any, input_mask: Any, chunk_size: int) -> Any:
    """AE ``cal_compressed_qk`` specialized to its BLOCK_SIZE=1."""

    import torch

    batch, heads, sequence, dimension = query.shape
    query = query * input_mask.reshape(batch, 1, sequence, 1)
    key = key * input_mask.reshape(batch, 1, 1, sequence)
    shortfall = chunk_size - sequence % chunk_size
    query = torch.cat(
        (query, torch.zeros(batch, heads, shortfall, dimension, device=query.device)),
        dim=2,
    )
    key = torch.cat(
        (key, torch.zeros(batch, heads, dimension, shortfall, device=key.device)),
        dim=3,
    )
    return torch.matmul(query, key)


def _ae_blocked_mask(
    ratios: Any,
    compressed_weights: Any,
    causal_mask: Any,
    chunk_size: int,
) -> Any:
    """AE ``gen_blocked_mask`` specialized to its BLOCK_SIZE=1."""

    import torch

    batch, heads, sequence, _ = causal_mask.shape
    minimum = -torch.finfo(torch.float32).max
    shortfall = chunk_size - sequence % chunk_size
    causal = torch.cat(
        (
            causal_mask,
            torch.full(
                (batch, heads, shortfall, sequence),
                minimum,
                dtype=torch.float32,
                device=causal_mask.device,
            ),
        ),
        dim=2,
    )
    causal = torch.cat(
        (
            causal,
            torch.full(
                (batch, heads, sequence + shortfall, shortfall),
                minimum,
                dtype=torch.float32,
                device=causal_mask.device,
            ),
        ),
        dim=3,
    )
    padded_sequence = causal.shape[-1]
    weights = compressed_weights + causal
    chunked = weights.reshape(
        batch, heads, padded_sequence // chunk_size, chunk_size * padded_sequence
    )
    real_tokens = chunk_size * padded_sequence - (chunked == minimum).sum(dim=-1)
    retain = (ratios.unsqueeze(-1) * real_tokens).to(torch.int32)
    retain = retain.reshape(batch, heads, padded_sequence // chunk_size, 1)
    sorted_weights, _ = torch.sort(chunked, dim=-1, descending=True)
    threshold = torch.gather(sorted_weights, dim=-1, index=retain.to(torch.int64))
    sparse_mask = torch.zeros_like(chunked)
    sparse_mask[chunked < threshold] = minimum
    sparse_mask = sparse_mask.reshape(batch, heads, padded_sequence, padded_sequence)
    sparse_mask = sparse_mask[:, :, :sequence, :sequence]
    result = sparse_mask + causal_mask
    result[result == -float("inf")] = minimum
    return result


class _AEHeadAblator:
    """Temporarily replace a model's eager attention with the AE ratio-zero path."""

    def __init__(self, layers: Sequence[Any], chunk_size: int) -> None:
        if not hasattr(layers[0], "self_attn"):
            raise RuntimeError("decoder layers do not expose self_attn")
        module_name = type(layers[0].self_attn).__module__
        self.modeling_module = importlib.import_module(module_name)
        self.original = getattr(self.modeling_module, "eager_attention_forward", None)
        if self.original is None:
            raise RuntimeError(
                f"{module_name} has no eager_attention_forward; use "
                "--ablation-mode zero-output or a compatible Transformers version"
            )
        self.chunk_size = chunk_size
        self.target_attention: Any | None = None
        self.target_head = -1
        self.target_calls = 0
        setattr(self.modeling_module, "eager_attention_forward", self._forward)

    def target(self, attention: Any, head: int) -> None:
        self.target_attention = attention
        self.target_head = head
        self.target_calls = 0

    def dense(self) -> None:
        """Use the AE's float32 dense eager path for layer profiling."""

        self.target_attention = None
        self.target_head = -1
        self.target_calls = 0

    def _forward(
        self,
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        import torch
        import torch.nn.functional as functional

        head_profiling = self.target_attention is not None
        is_target = head_profiling and module is self.target_attention
        if is_target:
            self.target_calls += 1
        groups = int(
            getattr(
                module,
                "num_key_value_groups",
                query.shape[1] // key.shape[1],
            )
        )
        key_states = _repeat_kv(key, groups)
        value_states = _repeat_kv(value, groups)
        if attention_mask is None or query.shape[-2] != key_states.shape[-2]:
            raise RuntimeError("AE profiling requires fixed-length eager prefill")
        causal = attention_mask[:, :, :, : key_states.shape[-2]].float()
        causal = causal.expand(query.shape[0], query.shape[1], -1, -1)
        weights = torch.matmul(
            query.float(), key_states.float().transpose(2, 3)
        ) * scaling
        if is_target:
            input_mask = torch.ones(
                query.shape[0],
                query.shape[-2],
                dtype=query.dtype,
                device=query.device,
            )
            compressed = _ae_compressed_qk(
                query.float(),
                key_states.float().transpose(2, 3),
                input_mask,
                self.chunk_size,
            )
            ratios = torch.ones(
                query.shape[0],
                query.shape[1],
                dtype=torch.float32,
                device=query.device,
            )
            ratios[:, self.target_head] = 0.0
            blocked = _ae_blocked_mask(ratios, compressed, causal, self.chunk_size)
            weights = weights + causal + blocked
        else:
            # In the artifact, every non-target layer takes the sparse branch
            # with retention=1.  Its generated sparse mask is identically zero
            # on valid entries, so the expensive compressed QK + sort has no
            # effect on the result.  Use the equivalent dense float32 path.
            weights = weights + causal
        weights = functional.softmax(weights, dim=-1, dtype=torch.float32)
        weights = functional.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value_states.float())
        output = output.transpose(1, 2).contiguous().to(query.dtype)
        return output, weights

    def close(self) -> None:
        setattr(self.modeling_module, "eager_attention_forward", self.original)

    def __enter__(self) -> "_AEHeadAblator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _zero_output_hook(head: int, head_dim: int) -> Any:
    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            raise RuntimeError("attention output projection received no positional input")
        attention = args[0].clone()
        attention[..., head * head_dim : (head + 1) * head_dim] = 0
        return (attention, *args[1:])

    return hook


def _layer_bypass_hook(
    _module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> Any:
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    if hidden is None:
        raise RuntimeError("decoder layer hook did not receive hidden_states")
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _perplexity(model: Any, windows: Sequence[Sequence[int]], args: argparse.Namespace) -> float:
    """Average batch losses equally, then exponentiate, as the AE does."""

    import torch

    total_loss: Any = 0
    batch_count = 0
    with torch.inference_mode():
        for start in range(0, len(windows), args.batch_size):
            batch = torch.tensor(
                windows[start : start + args.batch_size],
                dtype=torch.long,
                device=args.device,
            )
            attention_mask = torch.ones_like(batch)
            output = model(
                input_ids=batch,
                attention_mask=attention_mask,
                labels=batch,
                use_cache=False,
            )
            # Keep accumulation and exp in the loss tensor's device/dtype.
            # The artifact does this in Torch; converting each batch loss to a
            # Python float changes the final perplexity's low bits.
            total_loss = total_loss + output.loss
            batch_count += 1
    return float(torch.exp(total_loss / batch_count))


def _new_progress(
    args: argparse.Namespace,
    model: Any,
    text: str,
    windows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    layers = _decoder_layers(model)
    heads = int(model.config.num_attention_heads)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "ShadowNPU-AE-head-layer-ablation",
        "artifact": AE_ARTIFACT,
        "model": args.model_label,
        "head_perplexity": [[None] * heads for _ in layers],
        "layer_perplexity": [None] * len(layers),
        "calibration": {
            "model_source": str(args.model),
            "model_type": args.model_type,
            "tokenizer_source": str(args.tokenizer or args.model),
            "state_dict": str(args.state_dict) if args.state_dict else None,
            "activation_scales": (
                str(args.activation_scales) if args.activation_scales else None
            ),
            "activation_scales_sha256": (
                hashlib.sha256(args.activation_scales.read_bytes()).hexdigest()
                if args.activation_scales
                else None
            ),
            "clip_threshold": args.clip_threshold,
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "samples": len(windows),
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "seed": args.seed,
            "ablation_mode": args.ablation_mode,
            "chunk_size": args.chunk_size,
        },
    }


def _check_resume(progress: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in ("schema_version", "method", "model", "calibration"):
        if progress.get(field) != expected.get(field):
            raise ValueError(
                f"resume metadata mismatch for {field}; use a fresh output directory"
            )
    if len(progress.get("head_perplexity", [])) != len(expected["head_perplexity"]):
        raise ValueError("resume layer count does not match the loaded model")


def _write_raw_results(output_dir: Path, progress: dict[str, Any]) -> None:
    model_name = progress["model"]
    head_lines: list[str] = []
    for layer, row in enumerate(progress["head_perplexity"]):
        for head, value in enumerate(row):
            if value is not None:
                head_lines.append(f"{model_name} {layer} {head} {value}\n")
    layer_lines = [
        f"{model_name} {layer} {value}\n"
        for layer, value in enumerate(progress["layer_perplexity"])
        if value is not None
    ]
    _atomic_text(output_dir / "heads.txt", "".join(head_lines))
    _atomic_text(output_dir / "layers.txt", "".join(layer_lines))


def _checkpoint(output_dir: Path, progress: dict[str, Any]) -> None:
    _atomic_json(output_dir / "measurements.json", progress)
    _write_raw_results(output_dir, progress)


def _collect(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("loading model", flush=True)
    model, tokenizer = _load_model(args)
    layers = _decoder_layers(model)
    text = _load_calibration_text(args)
    windows = build_calibration_windows(
        tokenizer, text, args.context_length, args.max_samples
    )
    print(
        f"calibration: {len(windows)} x {args.context_length}; "
        f"layers={len(layers)}, heads={model.config.num_attention_heads}",
        flush=True,
    )

    expected = _new_progress(args, model, text, windows)
    measurement_path = args.output_dir / "measurements.json"
    if args.resume and measurement_path.exists():
        progress = json.loads(measurement_path.read_text(encoding="utf-8"))
        _check_resume(progress, expected)
    else:
        progress = expected
        _checkpoint(args.output_dir, progress)

    heads = int(model.config.num_attention_heads)
    configured_head_dim = getattr(model.config, "head_dim", None)
    head_dim = int(configured_head_dim or model.config.hidden_size // heads)
    started = time.perf_counter()
    ae_ablation: contextlib.AbstractContextManager[Any]
    if args.ablation_mode == "ae":
        ae_ablation = _AEHeadAblator(layers, args.chunk_size)
    else:
        ae_ablation = contextlib.nullcontext(None)

    with ae_ablation as ablator:
        for layer_index, layer in enumerate(layers):
            for head_index in range(heads):
                if progress["head_perplexity"][layer_index][head_index] is not None:
                    continue
                if args.ablation_mode == "ae":
                    ablator.target(layer.self_attn, head_index)
                    value = _perplexity(model, windows, args)
                    if ablator.target_calls == 0:
                        raise RuntimeError(
                            "eager attention patch was not invoked; verify "
                            "attn_implementation='eager' or use --ablation-mode zero-output"
                        )
                else:
                    handle = layer.self_attn.o_proj.register_forward_pre_hook(
                        _zero_output_hook(head_index, head_dim)
                    )
                    try:
                        value = _perplexity(model, windows, args)
                    finally:
                        handle.remove()
                progress["head_perplexity"][layer_index][head_index] = value
                _checkpoint(args.output_dir, progress)
                print(
                    f"head layer={layer_index:02d} head={head_index:02d} "
                    f"ppl={value:.9g}",
                    flush=True,
                )

        if args.ablation_mode == "ae":
            ablator.dense()
        for layer_index, layer in enumerate(layers):
            if progress["layer_perplexity"][layer_index] is not None:
                continue
            handle = layer.register_forward_hook(_layer_bypass_hook, with_kwargs=True)
            try:
                value = _perplexity(model, windows, args)
            finally:
                handle.remove()
            progress["layer_perplexity"][layer_index] = value
            _checkpoint(args.output_dir, progress)
            print(f"layer {layer_index:02d} bypass ppl={value:.9g}", flush=True)

    _, complete_heads, complete_layers = _validate_measurements(progress)
    profile, report = _make_profile_report(
        args.model_label,
        complete_heads,
        complete_layers,
        args.average_retention,
        args.clamping_threshold,
    )
    _atomic_text(args.output_dir / "head-retention.txt", profile)
    report["measurement_file"] = str(measurement_path)
    report["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(args.output_dir / "profile-report.json", report)
    print(
        f"profile: {args.output_dir / 'head-retention.txt'}; "
        f"actual retention={report['actual_average_retention']:.9g}",
        flush=True,
    )


def _convert(args: argparse.Namespace) -> None:
    if args.measurements is not None:
        value = json.loads(args.measurements.read_text(encoding="utf-8"))
        model_name, heads, layers = _validate_measurements(value)
        if args.model_label is not None and args.model_label != model_name:
            raise ValueError(
                f"measurement model is {model_name!r}, not {args.model_label!r}"
            )
    else:
        if args.head_results is None or args.layer_results is None:
            raise ValueError(
                "provide --measurements, or both --head-results and --layer-results"
            )
        model_name, heads, layers = read_ae_results(
            args.head_results, args.layer_results, args.model_label
        )
    profile, report = _make_profile_report(
        model_name,
        heads,
        layers,
        args.average_retention,
        args.clamping_threshold,
    )
    _atomic_text(args.output, profile)
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".json")
    _atomic_json(report_path, report)
    print(
        f"wrote {args.output}: {report['layers']} layers x {report['heads']} heads, "
        f"retention={report['actual_average_retention']:.9g}, "
        f"sparsity={report['global_sparsity']:.9g}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile per-head retention using the ShadowNPU AE procedure."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect", help="run head/layer ablations and emit a retention profile"
    )
    collect.add_argument("--model", required=True, help="local model path or HF id")
    collect.add_argument("--model-label", required=True, help="label stored in AE text files")
    collect.add_argument("--model-type", choices=("auto", "qwen2"), default="auto")
    collect.add_argument("--tokenizer", help="optional tokenizer path or HF id")
    collect.add_argument("--state-dict", type=Path)
    collect.add_argument(
        "--activation-scales",
        type=Path,
        help="AE activation-distribution JSON for static fake W8A8 quantization",
    )
    collect.add_argument("--clip-threshold", type=float, default=64.0)
    corpus = collect.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--calibration-text", type=Path)
    corpus.add_argument("--artifact-zip", type=Path)
    collect.add_argument("--artifact-member", default=DEFAULT_ARTIFACT_MEMBER)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--context-length", type=int, default=128)
    collect.add_argument("--batch-size", type=int, default=128)
    collect.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 uses every complete WikiText window, matching the AE",
    )
    collect.add_argument("--device", default="cuda")
    collect.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument(
        "--ablation-mode",
        choices=("ae", "zero-output"),
        default="ae",
        help="ae reproduces ratio=0 block masking; zero-output is a simpler fallback",
    )
    collect.add_argument("--chunk-size", type=int, default=32)
    collect.add_argument("--average-retention", type=float, default=0.2)
    collect.add_argument("--clamping-threshold", type=float, default=102400.0)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--local-files-only", action="store_true")
    collect.add_argument("--trust-remote-code", action="store_true")

    convert = subparsers.add_parser(
        "convert", help="convert AE raw results or measurements.json to a profile"
    )
    source = convert.add_mutually_exclusive_group(required=True)
    source.add_argument("--measurements", type=Path)
    source.add_argument("--head-results", type=Path)
    convert.add_argument("--layer-results", type=Path)
    convert.add_argument("--model-label")
    convert.add_argument("--average-retention", type=float, default=0.2)
    convert.add_argument("--clamping-threshold", type=float, default=102400.0)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect":
        if args.context_length <= 1 or args.batch_size <= 0 or args.max_samples < 0:
            raise ValueError("invalid context length, batch size, or max samples")
        if args.chunk_size <= 0:
            raise ValueError("chunk size must be positive")
        if not math.isfinite(args.clip_threshold) or args.clip_threshold <= 0.0:
            raise ValueError("clip threshold must be finite and positive")
        _collect(args)
    else:
        _convert(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
