#!/usr/bin/env python3
"""Collect, aggregate, and validate ShadowNPU three-stage latency profiles.

The device-side measurements are emitted by the native
``profile_hmx_pipeline_latency`` executable.  This host-side tool makes that
workflow reproducible and rejects profiles that do not cover the exact
NPU/Top-k/sparse grid required by the runtime scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "1"
LAYERS = 28
HEADS = 12
FIRST_PROFILED_LAYER = 2
HEAD_DIM = 128
OPERATOR_CAPACITY = 4160
BUCKET_COUNT = 9
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _finite_positive(text: str, location: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{location} must be finite and positive")
    return value


def _parse_cpu_list(value: str, location: str) -> tuple[int, ...]:
    try:
        cpus = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{location} must be a comma-separated CPU list") from error
    if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError(f"{location} must contain unique non-negative CPUs")
    return cpus


def key_lengths(max_key_len: int, query_len: int) -> tuple[int, ...]:
    if query_len <= 0 or max_key_len < query_len or max_key_len > OPERATOR_CAPACITY:
        raise ValueError("invalid query/max key length")
    values = list(range(query_len, min(4096, max_key_len) + 1, query_len))
    if max_key_len == OPERATOR_CAPACITY:
        values.append(OPERATOR_CAPACITY)
    elif max_key_len % query_len != 0:
        raise ValueError(
            "max key length must be a query-length multiple or exactly 4160"
        )
    return tuple(values)


def load_head_profile(path: Path) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = tuple(float(field) for field in line.split())
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid retention") from error
        if len(row) != HEADS:
            raise ValueError(
                f"{path}:{line_number}: three-stage v1 requires a "
                f"{LAYERS}x{HEADS} Qwen profile; row width is {len(row)}"
            )
        if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in row):
            raise ValueError(f"{path}:{line_number}: retention must be in (0, 1]")
        rows.append(row)
    if len(rows) != LAYERS:
        raise ValueError(
            f"three-stage v1 requires a {LAYERS}x{HEADS} Qwen profile; "
            f"got {len(rows)}x{len(rows[0]) if rows else 0}"
        )
    return tuple(rows)


def validate_pipeline_head_profile(
    retentions: tuple[tuple[float, ...], ...], path: Path
) -> None:
    non_dense = [
        (layer, head, value)
        for layer in range(FIRST_PROFILED_LAYER)
        for head, value in enumerate(retentions[layer])
        if not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-7)
    ]
    if non_dense:
        layer, head, value = non_dense[0]
        raise ValueError(
            f"{path}: native three-stage v1 profiles L{FIRST_PROFILED_LAYER}--"
            f"L{LAYERS - 1}; L{layer}H{head} retention is {value}, not dense"
        )


@dataclass(frozen=True)
class ManifestInfo:
    model: str
    head_counts: tuple[int, ...]
    buckets: tuple[tuple[float, float], ...]
    operator_rows: int


def load_manifest(path: Path, model: str, query_len: int) -> ManifestInfo:
    rows: dict[int, set[tuple[float, float]]] = {}
    models: set[str] = set()
    operator_rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[0] == "model":
            continue
        if len(fields) < 11:
            raise ValueError(f"{path}:{line_number}: manifest row needs 11 fields")
        row_model = fields[0]
        models.add(row_model)
        if row_model != model:
            continue
        try:
            heads = int(fields[1])
            shape = tuple(int(fields[index]) for index in (4, 5, 6))
            pair = (float(fields[7]), float(fields[8]))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid numeric field") from error
        if heads <= 0 or heads > HEADS:
            raise ValueError(f"{path}:{line_number}: invalid fused head count {heads}")
        if shape != (query_len, HEAD_DIM, OPERATOR_CAPACITY):
            raise ValueError(
                f"{path}:{line_number}: expected M/K/N="
                f"{query_len}/{HEAD_DIM}/{OPERATOR_CAPACITY}, got {shape}"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in pair):
            raise ValueError(f"{path}:{line_number}: scales must be positive")
        bucket_set = rows.setdefault(heads, set())
        if pair in bucket_set:
            raise ValueError(
                f"{path}:{line_number}: duplicate H={heads} scale pair {pair}"
            )
        bucket_set.add(pair)
        operator_rows += 1
    if models != {model}:
        raise ValueError(
            f"native profiler requires a single-model manifest {model!r}; "
            f"found {sorted(models)}"
        )
    if not rows:
        raise ValueError(f"manifest contains no operators for {model!r}")
    first = next(iter(rows.values()))
    if len(first) != BUCKET_COUNT:
        raise ValueError(f"manifest requires exactly {BUCKET_COUNT} Q/K buckets")
    for heads, pairs in rows.items():
        if pairs != first:
            raise ValueError(f"H={heads} does not contain the same bucket grid")
    return ManifestInfo(
        model=model,
        head_counts=tuple(sorted(rows)),
        buckets=tuple(sorted(first)),
        operator_rows=operator_rows,
    )


def inspect_inputs(
    manifest: Path,
    head_profile: Path,
    model: str,
    query_len: int,
    max_key_len: int,
    repetitions: int,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    retentions = load_head_profile(head_profile)
    validate_pipeline_head_profile(retentions, head_profile)
    manifest_info = load_manifest(manifest, model, query_len)
    lengths = key_lengths(max_key_len, query_len)
    npu_rows = len(lengths) * len(manifest_info.head_counts) * BUCKET_COUNT
    rows_per_head_stage = (
        len(lengths) * (LAYERS - FIRST_PROFILED_LAYER) * HEADS
    )
    return {
        "schema_version": 1,
        "model": model,
        "shape": {
            "layers": len(retentions),
            "heads": len(retentions[0]),
            "query_len": query_len,
            "head_dim": HEAD_DIM,
            "max_key_len": max_key_len,
        },
        "profiled_layers": [FIRST_PROFILED_LAYER, LAYERS - 1],
        "key_length_count": len(lengths),
        "key_lengths": list(lengths),
        "npu_head_counts": list(manifest_info.head_counts),
        "bucket_count": len(manifest_info.buckets),
        "operator_rows": manifest_info.operator_rows,
        "aggregate_rows": {
            "npu": npu_rows,
            "topk": rows_per_head_stage,
            "sparse": rows_per_head_stage,
            "total": npu_rows + 2 * rows_per_head_stage,
        },
        "minimum_raw_rows": repetitions * (npu_rows + 2 * rows_per_head_stage),
        "manifest_sha256": _sha256(manifest),
        "head_profile_sha256": _sha256(head_profile),
    }


def _parse_metadata_and_rows(
    path: Path,
) -> tuple[dict[str, str], list[tuple[str, tuple[str, ...], int]]]:
    metadata: dict[str, str] = {}
    rows: list[tuple[str, tuple[str, ...], int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        if line.startswith("#"):
            fields = line[1:].split("\t")
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"{path}:{line_number}: invalid metadata")
            if fields[0] in metadata:
                raise ValueError(f"{path}:{line_number}: duplicate metadata {fields[0]}")
            metadata[fields[0]] = fields[1]
            continue
        fields = tuple(line.split("\t"))
        if len(fields) != 7 or fields[0] not in {"npu", "topk", "sparse"}:
            raise ValueError(f"{path}:{line_number}: invalid stage row")
        rows.append((fields[0], fields, line_number))
    return metadata, rows


def validate_profile(
    profile: Path,
    manifest: Path | None = None,
    head_profile: Path | None = None,
    binary: Path | None = None,
    expected_model: str | None = None,
    expected_device: str | None = None,
    expected_main_cpu: str | None = None,
    expected_topk_cpus: str | None = None,
    expected_sparse_cpus: str | None = None,
) -> dict[str, Any]:
    metadata, rows = _parse_metadata_and_rows(profile)
    required = {
        "schema_version", "model", "device", "query_len", "head_dim",
        "max_key_len", "main_cpu", "topk_cpus", "sparse_cpus",
        "manifest_sha256", "head_profile_sha256", "binary_sha256",
        "minimum_samples", "statistic", "npu_key_cache_policy",
        "npu_execution_scope", "npu_heads",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"profile is missing metadata: {missing}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if metadata["statistic"] != "p50":
        raise ValueError("profile statistic must be p50")
    if metadata["npu_key_cache_policy"] != "incremental":
        raise ValueError("NPU key cache policy must be incremental")
    if metadata["npu_execution_scope"] != "attention":
        raise ValueError("NPU execution scope must be attention")
    query_len = int(metadata["query_len"])
    head_dim = int(metadata["head_dim"])
    max_key_len = int(metadata["max_key_len"])
    if head_dim != HEAD_DIM or int(metadata["minimum_samples"]) <= 0:
        raise ValueError("invalid shape or minimum_samples metadata")
    lengths = key_lengths(max_key_len, query_len)
    npu_heads = _parse_cpu_list(metadata["npu_heads"], "npu_heads")
    if any(head <= 0 or head > HEADS for head in npu_heads):
        raise ValueError("npu_heads must be in [1, 12]")
    _parse_cpu_list(metadata["topk_cpus"], "topk_cpus")
    _parse_cpu_list(metadata["sparse_cpus"], "sparse_cpus")
    if len(_parse_cpu_list(metadata["main_cpu"], "main_cpu")) != 1:
        raise ValueError("main_cpu must contain exactly one CPU")

    bindings = {
        "model": expected_model,
        "device": expected_device,
        "main_cpu": expected_main_cpu,
        "topk_cpus": expected_topk_cpus,
        "sparse_cpus": expected_sparse_cpus,
    }
    for key, expected in bindings.items():
        if expected is not None and metadata[key] != expected:
            raise ValueError(
                f"profile {key} mismatch: {metadata[key]!r} != {expected!r}"
            )
    if manifest is not None and metadata["manifest_sha256"] != _sha256(manifest):
        raise ValueError("manifest SHA-256 mismatch")
    if head_profile is not None and metadata["head_profile_sha256"] != _sha256(head_profile):
        raise ValueError("head profile SHA-256 mismatch")
    if binary is not None and metadata["binary_sha256"] != _sha256(binary):
        raise ValueError("profiler binary SHA-256 mismatch")

    retentions = load_head_profile(head_profile) if head_profile is not None else None
    if retentions is not None:
        validate_pipeline_head_profile(retentions, head_profile)
    manifest_info = (
        load_manifest(manifest, metadata["model"], query_len)
        if manifest is not None else None
    )
    if manifest_info is not None and npu_heads != manifest_info.head_counts:
        raise ValueError("npu_heads metadata does not match manifest")

    npu: set[tuple[int, int, float, float]] = set()
    topk: set[tuple[int, int, int]] = set()
    sparse: set[tuple[int, int, int]] = set()
    npu_pairs: dict[int, set[tuple[float, float]]] = {}
    p50_values: dict[str, list[float]] = {"npu": [], "topk": [], "sparse": []}
    for stage, fields, line_number in rows:
        try:
            if stage == "npu":
                key_len, heads = int(fields[1]), int(fields[2])
                q_scale = _finite_positive(fields[3], "q_scale")
                k_scale = _finite_positive(fields[4], "k_scale")
                key: Any = (key_len, heads, q_scale, k_scale)
                target = npu
                npu_pairs.setdefault(heads, set()).add((q_scale, k_scale))
            else:
                key_len, layer, head = map(int, fields[1:4])
                retention = _finite_positive(fields[4], "retention")
                if retention > 1.0:
                    raise ValueError("retention exceeds 1")
                if retentions is not None and not math.isclose(
                    retention, retentions[layer][head], rel_tol=1e-5, abs_tol=1e-7
                ):
                    raise ValueError("retention does not match head profile")
                key = (key_len, layer, head)
                target = topk if stage == "topk" else sparse
            p50 = _finite_positive(fields[5], "p50")
            p95 = _finite_positive(fields[6], "p95")
            if p95 < p50:
                raise ValueError("p95 is below p50")
        except (IndexError, ValueError) as error:
            raise ValueError(f"{profile}:{line_number}: {error}") from error
        if key in target:
            raise ValueError(f"{profile}:{line_number}: duplicate {stage} row")
        target.add(key)
        p50_values[stage].append(p50)

    if manifest_info is not None:
        buckets = set(manifest_info.buckets)
    else:
        if set(npu_pairs) != set(npu_heads):
            raise ValueError("NPU row head counts do not match metadata")
        bucket_sets = list(npu_pairs.values())
        if not bucket_sets or len(bucket_sets[0]) != BUCKET_COUNT:
            raise ValueError(f"NPU rows require exactly {BUCKET_COUNT} buckets")
        if any(pairs != bucket_sets[0] for pairs in bucket_sets[1:]):
            raise ValueError("NPU head counts have inconsistent bucket grids")
        buckets = bucket_sets[0]
    expected_npu = {
        (key_len, heads, q_scale, k_scale)
        for key_len in lengths
        for heads in npu_heads
        for q_scale, k_scale in buckets
    }
    expected_heads = {
        (key_len, layer, head)
        for key_len in lengths
        for layer in range(FIRST_PROFILED_LAYER, LAYERS)
        for head in range(HEADS)
    }
    for stage, actual, expected in (
        ("npu", npu, expected_npu),
        ("topk", topk, expected_heads),
        ("sparse", sparse, expected_heads),
    ):
        if actual != expected:
            missing_rows = sorted(expected - actual)[:5]
            extra_rows = sorted(actual - expected)[:5]
            raise ValueError(
                f"{stage} grid is incomplete: missing={missing_rows}, extra={extra_rows}"
            )
    return {
        "valid": True,
        "profile": str(profile),
        "metadata": metadata,
        "rows": {"npu": len(npu), "topk": len(topk), "sparse": len(sparse)},
        "median_p50_us": {
            stage: statistics.median(values)
            for stage, values in p50_values.items()
        },
        "profile_sha256": _sha256(profile),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    info = load_manifest(args.manifest, args.model, args.query_len)
    retentions = load_head_profile(args.head_profile)
    validate_pipeline_head_profile(retentions, args.head_profile)
    if not args.binary.is_file():
        raise ValueError(f"profiler binary does not exist: {args.binary}")
    if args.minimum_samples <= 0:
        raise ValueError("minimum samples must be positive")

    npu_samples: dict[tuple[int, int, float, float], list[float]] = defaultdict(list)
    head_samples: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    sources = [(args.raw, {"npu", "topk", "sparse"})]
    if args.head_raw is not None:
        sources = [
            (args.raw, {"npu"}),
            (args.head_raw, {"topk", "sparse"}),
        ]
    for source_path, accepted_stages in sources:
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            fields = line.split("\t")
            if not fields or fields[0] not in accepted_stages:
                continue
            try:
                if fields[0] == "npu" and len(fields) == 6:
                    key_len, heads = int(fields[1]), int(fields[2])
                    measured_pair = (float(fields[3]), float(fields[4]))
                    pair = min(
                        info.buckets,
                        key=lambda candidate: (
                            (candidate[0] - measured_pair[0]) ** 2
                            + (candidate[1] - measured_pair[1]) ** 2
                        ),
                    )
                    key: Any = (key_len, heads, pair[0], pair[1])
                elif fields[0] in {"topk", "sparse"} and len(fields) == 6:
                    key_len, layer, head = map(int, fields[1:4])
                    retention = float(fields[4])
                    if not math.isclose(
                        retention, retentions[layer][head],
                        rel_tol=1e-5, abs_tol=1e-7,
                    ):
                        raise ValueError("retention does not match head profile")
                    key = (fields[0], key_len, layer, head)
                else:
                    raise ValueError("invalid raw timing row")
                elapsed = _finite_positive(fields[5], "elapsed time")
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"{source_path}:{line_number}: {error}"
                ) from error
            if fields[0] == "npu":
                npu_samples[key].append(elapsed)
            else:
                head_samples[key].append(elapsed)

    lengths = key_lengths(args.max_key_len, args.query_len)
    for key_len in lengths:
        for heads in info.head_counts:
            for q_scale, k_scale in info.buckets:
                count = len(npu_samples[(key_len, heads, q_scale, k_scale)])
                if count < args.minimum_samples:
                    raise ValueError(
                        f"NPU samples for N={key_len}, H={heads}, "
                        f"q={q_scale}, k={k_scale} are {count}; "
                        f"need {args.minimum_samples}"
                    )
        for layer in range(FIRST_PROFILED_LAYER, LAYERS):
            for head in range(HEADS):
                for stage in ("topk", "sparse"):
                    count = len(head_samples[(stage, key_len, layer, head)])
                    if count < args.minimum_samples:
                        raise ValueError(
                            f"{stage} samples for N={key_len}, L={layer}, "
                            f"H={head} are {count}; need {args.minimum_samples}"
                        )

    def percentile(values: Sequence[float], percent: float) -> float:
        ordered = sorted(values)
        rank = (len(ordered) - 1) * percent / 100.0
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)

    metadata = (
        ("schema_version", SCHEMA_VERSION),
        ("model", args.model),
        ("device", args.device),
        ("query_len", str(args.query_len)),
        ("head_dim", str(HEAD_DIM)),
        ("max_key_len", str(args.max_key_len)),
        ("main_cpu", args.main_cpu),
        ("topk_cpus", args.topk_cpus),
        ("sparse_cpus", args.sparse_cpus),
        ("manifest_sha256", _sha256(args.manifest)),
        ("head_profile_sha256", _sha256(args.head_profile)),
        ("binary_sha256", _sha256(args.binary)),
        ("minimum_samples", str(args.minimum_samples)),
        ("statistic", "p50"),
        ("npu_key_cache_policy", "incremental"),
        ("npu_execution_scope", "attention"),
        ("npu_heads", ",".join(map(str, info.head_counts))),
    )
    lines = [f"#{name}\t{value}" for name, value in metadata]
    for (key_len, heads, q_scale, k_scale), values in sorted(npu_samples.items()):
        lines.append(
            f"npu\t{key_len}\t{heads}\t{q_scale:.17g}\t{k_scale:.17g}\t"
            f"{percentile(values, 50):.9g}\t{percentile(values, 95):.9g}"
        )
    for (stage, key_len, layer, head), values in sorted(head_samples.items()):
        lines.append(
            f"{stage}\t{key_len}\t{layer}\t{head}\t"
            f"{retentions[layer][head]:.17g}\t"
            f"{percentile(values, 50):.9g}\t{percentile(values, 95):.9g}"
        )
    text = "\n".join(lines) + "\n"
    _atomic_text(args.output, text)
    return validate_profile(
        args.output,
        args.manifest,
        args.head_profile,
        args.binary,
        args.model,
        args.device,
        args.main_cpu,
        args.topk_cpus,
        args.sparse_cpus,
    )


def _parse_extra_env(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not _ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid environment assignment {value!r}")
        if name in result:
            raise ValueError(f"duplicate environment assignment {name}")
        result[name] = setting
    return result


def make_device_command(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    topk_cpus = _parse_cpu_list(args.topk_cpus, "topk_cpus")
    sparse_cpus = _parse_cpu_list(args.sparse_cpus, "sparse_cpus")
    _parse_cpu_list(args.main_cpu, "main_cpu")
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    key_lengths(args.max_key_len, args.query_len)
    environment = {
        "MLLM_HMX_PIPELINE_PROFILE_STAGES": args.stages,
        "MLLM_HMX_PIPELINE_MAIN_CPU": args.main_cpu,
        "MLLM_HMX_PIPELINE_TOPK_CPU": args.topk_cpus,
        "MLLM_HMX_PIPELINE_TOPK_WORKERS": str(len(topk_cpus)),
        "MLLM_HMX_PIPELINE_SPARSE_CPU": args.sparse_cpus,
        "MLLM_HMX_PIPELINE_SPARSE_WORKERS": str(len(sparse_cpus)),
        "MLLM_HMX_PIPELINE_TOPK_MODE": args.topk_mode,
    }
    extra_environment = _parse_extra_env(args.env)
    conflicting = sorted(set(environment) & set(extra_environment))
    if conflicting:
        raise ValueError(
            "--env may not override resource settings: "
            + ", ".join(conflicting)
        )
    environment.update(extra_environment)
    tokens = ["env"]
    tokens.extend(f"{key}={value}" for key, value in environment.items())
    tokens.extend(
        [
            args.device_binary,
            args.device_manifest,
            args.device_head_profile,
            args.device_raw,
            str(args.max_key_len),
            str(args.warmup),
            str(args.repetitions),
            str(args.query_len),
        ]
    )
    return " ".join(shlex.quote(token) for token in tokens), environment


def collect_adb(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise ValueError(f"output already exists (use --force): {args.output}")
    remote_command, environment = make_device_command(args)
    adb_prefix = [args.adb]
    if args.serial:
        adb_prefix.extend(["-s", args.serial])
    subprocess.run(adb_prefix + ["get-state"], check=True)
    completed = subprocess.run(adb_prefix + ["shell", remote_command], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"device profiler failed with status {completed.returncode}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        adb_prefix + ["pull", args.device_raw, str(args.output)], check=True
    )
    metadata = {
        "schema_version": 1,
        "runner": "adb",
        "serial": args.serial,
        "remote_command": remote_command,
        "environment": environment,
        "raw_output": str(args.output),
        "raw_sha256": _sha256(args.output),
        "max_key_len": args.max_key_len,
        "query_len": args.query_len,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "stages": args.stages,
    }
    _atomic_json(args.output.with_suffix(args.output.suffix + ".json"), metadata)
    return metadata


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _add_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen2_1p5b")
    parser.add_argument("--device", required=True)
    parser.add_argument("--main-cpu", default="2")
    parser.add_argument("--topk-cpus", default="4,6")
    parser.add_argument("--sparse-cpus", default="7")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ShadowNPU NPU estimation, Top-k, and sparse QKV latency"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate inputs and print the required grid")
    inspect_parser.add_argument("--manifest", type=Path, required=True)
    inspect_parser.add_argument("--head-profile", type=Path, required=True)
    inspect_parser.add_argument("--model", default="qwen2_1p5b")
    inspect_parser.add_argument("--query-len", type=int, default=512)
    inspect_parser.add_argument("--max-key-len", type=int, default=4160)
    inspect_parser.add_argument("--repetitions", type=int, default=20)

    collect_parser = subparsers.add_parser("collect-adb", help="run an already-deployed native profiler and pull raw TSV")
    collect_parser.add_argument("--adb", default="adb")
    collect_parser.add_argument("--serial")
    collect_parser.add_argument("--device-binary", required=True)
    collect_parser.add_argument("--device-manifest", required=True)
    collect_parser.add_argument("--device-head-profile", required=True)
    collect_parser.add_argument("--device-raw", required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--stages", choices=("all", "npu", "heads"), default="all")
    collect_parser.add_argument("--query-len", type=int, default=512)
    collect_parser.add_argument("--max-key-len", type=int, default=4160)
    collect_parser.add_argument("--warmup", type=int, default=3)
    collect_parser.add_argument("--repetitions", type=int, default=20)
    collect_parser.add_argument("--main-cpu", default="2")
    collect_parser.add_argument("--topk-cpus", default="4,6")
    collect_parser.add_argument("--sparse-cpus", default="7")
    collect_parser.add_argument("--topk-mode", choices=("cooperative", "independent"), default="cooperative")
    collect_parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    collect_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="turn raw samples into strict runtime schema-v1")
    aggregate_parser.add_argument("--raw", type=Path, required=True)
    aggregate_parser.add_argument("--head-raw", type=Path)
    aggregate_parser.add_argument("--manifest", type=Path, required=True)
    aggregate_parser.add_argument("--head-profile", type=Path, required=True)
    aggregate_parser.add_argument("--binary", type=Path, required=True)
    _add_binding_arguments(aggregate_parser)
    aggregate_parser.add_argument("--query-len", type=int, default=512)
    aggregate_parser.add_argument("--max-key-len", type=int, default=4160)
    aggregate_parser.add_argument("--minimum-samples", type=int, default=20)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="strictly validate a runtime profile")
    validate_parser.add_argument("--profile", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path)
    validate_parser.add_argument("--head-profile", type=Path)
    validate_parser.add_argument("--binary", type=Path)
    validate_parser.add_argument("--model")
    validate_parser.add_argument("--device")
    validate_parser.add_argument("--main-cpu")
    validate_parser.add_argument("--topk-cpus")
    validate_parser.add_argument("--sparse-cpus")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_inputs(
                args.manifest, args.head_profile, args.model,
                args.query_len, args.max_key_len, args.repetitions,
            )
        elif args.command == "collect-adb":
            result = collect_adb(args)
        elif args.command == "aggregate":
            result = aggregate(args)
        else:
            result = validate_profile(
                args.profile, args.manifest, args.head_profile, args.binary,
                args.model, args.device, args.main_cpu,
                args.topk_cpus, args.sparse_cpus,
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    _print_json(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
