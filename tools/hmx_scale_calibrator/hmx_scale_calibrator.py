#!/usr/bin/env python3
"""Calibrate model-bound HMX INT8 Q/K/output scales from device logs."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FIELD_PATTERN = re.compile(rf"\b([A-Za-z_][A-Za-z0-9_]*)=({FLOAT_PATTERN}|[^\s]+)")
MATRIX_SCALE_PATTERN = re.compile(
    rf"\b(\d+)\(q=({FLOAT_PATTERN}),k=({FLOAT_PATTERN})\)"
)


@dataclass(frozen=True)
class QKObservation:
    source: str
    layer: int | None
    head: int
    key_len: int | None
    q_scale: float
    k_scale: float


@dataclass(frozen=True)
class OutputObservation:
    source: str
    layer: int | None
    head: int
    key_len: int | None
    query_begin: int | None
    q_scale: float
    k_scale: float
    output_scale: float
    requant_scale: float
    peak: int
    attempts: int


@dataclass(frozen=True)
class RecallObservation:
    source: str
    layer: int | None
    head: int
    key_len: int | None
    recall: float
    unique_scores: int
    zero_fraction: float
    saturation_fraction: float
    score_mismatch_fraction: float


@dataclass
class ParsedCalibration:
    qk: list[QKObservation]
    output: list[OutputObservation]
    recall: list[RecallObservation]
    quality_results: list[bool]
    sources: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value: str | float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _fraction(value: str | float, name: str, allow_zero: bool = False) -> float:
    parsed = float(value)
    lower_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not math.isfinite(parsed) or not lower_ok or parsed > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}")
    return parsed


def _optional_int(fields: dict[str, str], name: str) -> int | None:
    if name not in fields:
        return None
    return int(fields[name])


def _fields(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in FIELD_PATTERN.finditer(line)}


def _sidecar_metadata(path: Path) -> tuple[str | None, str | None]:
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        return None, None
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid collection sidecar {sidecar}: {error}") from error
    model = value.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"collection sidecar {sidecar} has no model ID")
    return model, _sha256(sidecar)


def parse_calibration_logs(
    paths: Sequence[Path],
    model: str,
    require_model_sidecar: bool = False,
) -> ParsedCalibration:
    if not paths:
        raise ValueError("at least one diagnostic log is required")
    qk: list[QKObservation] = []
    output: list[OutputObservation] = []
    recall: list[RecallObservation] = []
    quality_results: list[bool] = []
    sources: list[dict[str, Any]] = []

    for path in paths:
        if not path.is_file():
            raise ValueError(f"diagnostic log does not exist: {path}")
        sidecar_model, sidecar_sha256 = _sidecar_metadata(path)
        if require_model_sidecar and sidecar_model is None:
            raise ValueError(f"diagnostic log has no model sidecar: {path}.json")
        if sidecar_model is not None and sidecar_model != model:
            raise ValueError(
                f"diagnostic log {path} belongs to {sidecar_model!r}, not {model!r}"
            )
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "model_sidecar": sidecar_model,
                "model_sidecar_sha256": sidecar_sha256,
            }
        )
        text = path.read_bytes().replace(b"\0", b"").decode(
            "utf-8", errors="replace"
        )
        bucket_qk: list[QKObservation] = []
        recall_qk: list[QKObservation] = []
        for line in text.splitlines():
            fields = _fields(line)
            layer = _optional_int(fields, "layer")
            key_len = _optional_int(fields, "key_len")
            if "[HMX_INT8_BUCKET]" in line or "[HMX_INT8_BUCKET_FALLBACK]" in line:
                for match in MATRIX_SCALE_PATTERN.finditer(line):
                    bucket_qk.append(
                        QKObservation(
                            str(path), layer, int(match.group(1)), key_len,
                            _positive(match.group(2), "measured Q scale"),
                            _positive(match.group(3), "measured K scale"),
                        )
                    )
            elif "[HMX_INT8_OUTPUT_SCALE]" in line:
                required = ("matrix", "q_scale", "k_scale", "output", "requant", "peak", "attempts")
                missing = [name for name in required if name not in fields]
                if missing:
                    raise ValueError(
                        f"malformed output-scale record in {path}: missing {missing}"
                    )
                output.append(
                    OutputObservation(
                        str(path), layer, int(fields["matrix"]), key_len,
                        _optional_int(fields, "query_begin"),
                        _positive(fields["q_scale"], "output Q scale"),
                        _positive(fields["k_scale"], "output K scale"),
                        _positive(fields["output"], "dynamic output scale"),
                        _positive(fields["requant"], "dynamic requant scale"),
                        int(fields["peak"]), int(fields["attempts"]),
                    )
                )
            elif "[HMX_INT8_RECALL]" in line and "error=" not in line:
                required = (
                    "matrix", "recall", "unique_scores", "zero_fraction",
                    "saturation_fraction", "score_mismatch_fraction",
                    "measured_q", "measured_k",
                )
                missing = [name for name in required if name not in fields]
                if missing:
                    raise ValueError(
                        f"malformed recall record in {path}: missing {missing}"
                    )
                head = int(fields["matrix"])
                recall.append(
                    RecallObservation(
                        str(path), layer, head, key_len,
                        _fraction(fields["recall"], "Top-k recall", allow_zero=True),
                        int(fields["unique_scores"]),
                        _fraction(fields["zero_fraction"], "zero fraction", allow_zero=True),
                        _fraction(fields["saturation_fraction"], "saturation fraction", allow_zero=True),
                        _fraction(fields["score_mismatch_fraction"], "score mismatch fraction", allow_zero=True),
                    )
                )
                recall_qk.append(
                    QKObservation(
                        str(path), layer, head, key_len,
                        _positive(fields["measured_q"], "measured Q scale"),
                        _positive(fields["measured_k"], "measured K scale"),
                    )
                )
            elif "QUALITY_RESULT" in line and "retrieval_exact" in fields:
                quality_results.append(fields["retrieval_exact"] == "1")
        # Recall diagnostics repeat the same Q/K observations. Prefer the much
        # cheaper bucket records whenever the log contains them.
        qk.extend(bucket_qk if bucket_qk else recall_qk)

    return ParsedCalibration(qk, output, recall, quality_results, sources)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    quantile = _fraction(quantile, "quantile", allow_zero=True)
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(values),
        "min": min(values),
        "p05": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "mean": statistics.fmean(values),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _parse_multipliers(value: str) -> list[float]:
    result = []
    for item in re.split(r"[,;\s]+", value.strip()):
        if item:
            result.append(_positive(item, "bucket multiplier"))
    if not result:
        raise ValueError("at least one bucket multiplier is required")
    if len(set(result)) != len(result):
        raise ValueError("bucket multipliers must be unique")
    return sorted(result)


def _dispatcher_diagnostics(
    observations: Sequence[QKObservation],
    q_buckets: Sequence[float],
    k_buckets: Sequence[float],
) -> dict[str, Any]:
    clipped_q = clipped_k = clipped_either = 0
    q_relative_error: list[float] = []
    k_relative_error: list[float] = []
    assignments: dict[str, int] = {}
    pairs = [(q, k) for q in q_buckets for k in k_buckets]
    for item in observations:
        q_selected, k_selected = min(
            pairs,
            key=lambda pair: (
                (pair[0] - item.q_scale) ** 2 + (pair[1] - item.k_scale) ** 2,
                pair[0], pair[1],
            ),
        )
        q_clip = item.q_scale > q_selected
        k_clip = item.k_scale > k_selected
        clipped_q += q_clip
        clipped_k += k_clip
        clipped_either += q_clip or k_clip
        q_relative_error.append(abs(q_selected - item.q_scale) / item.q_scale)
        k_relative_error.append(abs(k_selected - item.k_scale) / item.k_scale)
        key = f"q={q_selected:.17g},k={k_selected:.17g}"
        assignments[key] = assignments.get(key, 0) + 1
    count = len(observations)
    return {
        "observations": count,
        "q_clipping_fraction": clipped_q / count,
        "k_clipping_fraction": clipped_k / count,
        "either_clipping_fraction": clipped_either / count,
        "q_relative_error_mean": statistics.fmean(q_relative_error),
        "k_relative_error_mean": statistics.fmean(k_relative_error),
        "assignments": dict(sorted(assignments.items())),
    }


def make_model_scale_profile(
    parsed: ParsedCalibration,
    model: str,
    head_dim: int,
    multipliers: Sequence[float],
    output_quantile: float,
    output_margin: float,
    model_sha256: str | None = None,
) -> dict[str, Any]:
    if not model or any(character.isspace() for character in model):
        raise ValueError("model must be a non-empty ID without whitespace")
    if head_dim <= 0:
        raise ValueError("head dimension must be positive")
    if not parsed.qk:
        raise ValueError("logs contain no HMX Q/K scale diagnostics")
    if not parsed.output:
        raise ValueError(
            "logs contain no dynamic output-scale diagnostics; collect with "
            "MLLM_HMX_INT8_DYNAMIC_OUTPUT_SCALE=1 and "
            "MLLM_HMX_INT8_BUCKET_DIAGNOSTICS=1"
        )
    output_quantile = _fraction(output_quantile, "output quantile", allow_zero=True)
    output_margin = _positive(output_margin, "output margin")
    if output_margin < 1.0:
        raise ValueError("output margin must be at least one")

    q_values = [item.q_scale for item in parsed.qk]
    k_values = [item.k_scale for item in parsed.qk]
    output_values = [item.output_scale for item in parsed.output]
    requant_values = [item.requant_scale for item in parsed.output]
    q_base = statistics.fmean(q_values)
    k_base = statistics.fmean(k_values)
    q_buckets = [q_base * value for value in multipliers]
    k_buckets = [k_base * value for value in multipliers]
    fixed_output = _percentile(output_values, output_quantile) * output_margin
    # A smaller requant multiplier is safer against saturation. The fifth
    # percentile retains the dynamic probe's ranking resolution while avoiding
    # one-off minima from dominating every compiled Q/K pair.
    target_requant = _percentile(requant_values, 0.05) / output_margin

    output_summary = _summary(output_values)
    output_range_ratio = float(output_summary["p95"]) / float(output_summary["p05"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": "hmx-device-model-scale-calibration",
        "model": model,
        "model_sha256": model_sha256,
        "head_dim": head_dim,
        "sources": parsed.sources,
        "q": {"statistics": _summary(q_values), "base_scale": q_base, "buckets": q_buckets},
        "k": {"statistics": _summary(k_values), "base_scale": k_base, "buckets": k_buckets},
        "bucket_multipliers": list(multipliers),
        "dispatcher_fit": _dispatcher_diagnostics(parsed.qk, q_buckets, k_buckets),
        "output": {
            "statistics": output_summary,
            "requant_statistics": _summary(requant_values),
            "fixed_scale_quantile": output_quantile,
            "fixed_scale_margin": output_margin,
            "fixed_scale": fixed_output,
            "target_requant_scale": target_requant,
            "p95_to_p05_ratio": output_range_ratio,
            "recommended_mode": (
                "target-requant-per-qk-pair" if output_range_ratio > 2.0
                else "fixed-output"
            ),
        },
        "dynamic_probe": {
            "records": len(parsed.output),
            "peak": _summary([float(item.peak) for item in parsed.output]),
            "attempts": _summary([float(item.attempts) for item in parsed.output]),
        },
    }
    if parsed.recall:
        result["input_recall"] = recall_summary(parsed.recall)
    if parsed.quality_results:
        result["input_quality"] = {
            "runs": len(parsed.quality_results),
            "all_retrieval_exact": all(parsed.quality_results),
        }
    return result


def recall_summary(records: Sequence[RecallObservation]) -> dict[str, Any]:
    if not records:
        raise ValueError("no recall records")
    return {
        "records": len(records),
        "mean": statistics.fmean(item.recall for item in records),
        "minimum": min(item.recall for item in records),
        "p05": _percentile([item.recall for item in records], 0.05),
        "mean_unique_scores": statistics.fmean(item.unique_scores for item in records),
        "mean_zero_fraction": statistics.fmean(item.zero_fraction for item in records),
        "mean_saturation_fraction": statistics.fmean(
            item.saturation_fraction for item in records
        ),
        "mean_score_mismatch_fraction": statistics.fmean(
            item.score_mismatch_fraction for item in records
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_catalog(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "model head_dim q_base_scale k_base_scale output_scale\n"
        f"{profile['model']} {profile['head_dim']} "
        f"{profile['q']['base_scale']:.17g} {profile['k']['base_scale']:.17g} "
        f"{profile['output']['fixed_scale']:.17g}\n",
        encoding="utf-8",
    )


def _write_build_env(
    path: Path,
    profile: dict[str, Any],
    catalog: Path,
    bank_mode: str,
) -> None:
    q_scales = " ".join(f"{value:.17g}" for value in profile["q"]["buckets"])
    k_scales = " ".join(f"{value:.17g}" for value in profile["k"]["buckets"])
    lines = [
        f"HMX_OPERATOR_CATALOG_FILE={shlex.quote(str(catalog.resolve()))}",
        f"HMX_OPERATOR_MODELS={shlex.quote(profile['model'])}",
        f"HMX_OPERATOR_Q_SCALES={shlex.quote(q_scales)}",
        f"HMX_OPERATOR_K_SCALES={shlex.quote(k_scales)}",
    ]
    if bank_mode == "target-requant":
        lines.append(
            "HMX_OPERATOR_TARGET_REQUANT_SCALE="
            f"{profile['output']['target_requant_scale']:.17g}"
        )
    else:
        lines.append(
            "HMX_OPERATOR_OUTPUT_SCALES="
            f"{shlex.quote(format(profile['output']['fixed_scale'], '.17g'))}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_env(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"environment override must be KEY=VALUE: {value!r}")
        key, item = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment name: {key!r}")
        result[key] = item
    return result


def _run_collect_adb(args: argparse.Namespace) -> None:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("collect-adb requires a device command after --")
    environment = _parse_env(args.env)
    forced = {
        "MLLM_HMX_INT8_BUCKET_DIAGNOSTICS": "1",
        "MLLM_HMX_INT8_DYNAMIC_OUTPUT_SCALE": "1",
        "MLLM_HMX_INT8_RECALL_DIAGNOSTICS": "1" if args.recall else "0",
        "MLLM_HMX_INT8_TOPK_OVERSAMPLE": "1",
    }
    environment.update(forced)
    remote = ["env"] + [f"{key}={value}" for key, value in sorted(environment.items())]
    remote.extend(command)
    adb = [args.adb]
    if args.serial:
        adb.extend(["-s", args.serial])
    adb.extend(["shell", shlex.join(remote)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        process = subprocess.Popen(adb, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for block in iter(lambda: process.stdout.read(65536), b""):
            output.write(block)
            output.flush()
            sys.stdout.buffer.write(block.replace(b"\0", b""))
            sys.stdout.buffer.flush()
        status = process.wait()
    sidecar = {
        "schema_version": 1,
        "model": args.model,
        "serial": args.serial,
        "command": command,
        "environment": environment,
        "exit_status": status,
        "log_sha256": _sha256(args.output),
    }
    _write_json(Path(str(args.output) + ".json"), sidecar)
    if status != 0:
        raise ValueError(f"device calibration command failed with status {status}")
    parsed = parse_calibration_logs([args.output], args.model, True)
    if not parsed.qk or not parsed.output:
        raise ValueError(
            "device run completed but did not emit Q/K and dynamic output diagnostics"
        )
    print(
        f"collected model={args.model} qk_records={len(parsed.qk)} "
        f"output_records={len(parsed.output)} log={args.output}"
    )


def _run_calibrate(args: argparse.Namespace) -> None:
    multipliers = _parse_multipliers(args.bucket_multipliers)
    parsed = parse_calibration_logs(args.log, args.model, args.require_model_sidecar)
    model_sha = args.model_sha256
    if args.model_file:
        computed = _sha256(args.model_file)
        if model_sha and model_sha != computed:
            raise ValueError("--model-sha256 does not match --model-file")
        model_sha = computed
    profile = make_model_scale_profile(
        parsed, args.model, args.head_dim, multipliers,
        args.output_quantile, args.output_margin, model_sha,
    )
    _write_json(args.output, profile)
    catalog = args.catalog or args.output.with_suffix(".catalog.txt")
    build_env = args.build_env or args.output.with_suffix(".build.env")
    _write_catalog(catalog, profile)
    _write_build_env(build_env, profile, catalog, args.bank_mode)
    print(
        f"wrote model-bound scale profile {args.output} "
        f"(model={args.model}, Q={profile['q']['base_scale']:.9g}, "
        f"K={profile['k']['base_scale']:.9g}, "
        f"fixed_output={profile['output']['fixed_scale']:.9g}, "
        f"target_requant={profile['output']['target_requant_scale']:.9g})"
    )
    print(f"wrote operator catalog {catalog}")
    print(f"wrote bank build environment {build_env}")


def _run_validate(args: argparse.Namespace) -> None:
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    model = profile.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("scale profile has no model ID")
    parsed = parse_calibration_logs(args.log, model, args.require_model_sidecar)
    result: dict[str, Any] = {
        "schema_version": 1,
        "model": model,
        "profile_sha256": _sha256(args.profile),
        "logs": parsed.sources,
        "quality_runs": len(parsed.quality_results),
        "quality_all_retrieval_exact": (
            all(parsed.quality_results) if parsed.quality_results else None
        ),
    }
    passed = True
    if args.require_quality and (
        not parsed.quality_results or not all(parsed.quality_results)
    ):
        passed = False
    if parsed.recall:
        result["recall"] = recall_summary(parsed.recall)
        passed = passed and result["recall"]["mean"] >= args.minimum_mean_recall
        passed = passed and result["recall"]["minimum"] >= args.minimum_head_recall
        passed = passed and (
            result["recall"]["mean_score_mismatch_fraction"]
            <= args.maximum_score_mismatch
        )
    elif args.require_recall:
        passed = False
    if parsed.qk:
        result["dispatcher_fit"] = _dispatcher_diagnostics(
            parsed.qk, profile["q"]["buckets"], profile["k"]["buckets"]
        )
    result["passed"] = passed
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise ValueError("scale validation failed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect, calibrate, and validate model-specific HMX INT8 scales."
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    collect = subparsers.add_parser(
        "collect-adb", help="run an existing device benchmark with scale diagnostics"
    )
    collect.add_argument("--adb", default="adb")
    collect.add_argument("--serial")
    collect.add_argument("--model", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--env", action="append", default=[])
    collect.add_argument("--recall", action="store_true")
    collect.add_argument("command", nargs=argparse.REMAINDER)
    collect.set_defaults(function=_run_collect_adb)

    calibrate = subparsers.add_parser(
        "calibrate", help="fit a model-bound static Q/K and output-scale profile"
    )
    calibrate.add_argument("--log", type=Path, action="append", required=True)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--model-file", type=Path)
    calibrate.add_argument("--model-sha256")
    calibrate.add_argument("--head-dim", type=int, required=True)
    calibrate.add_argument("--bucket-multipliers", default="0.5,1,2")
    calibrate.add_argument("--output-quantile", type=float, default=1.0)
    calibrate.add_argument("--output-margin", type=float, default=1.05)
    calibrate.add_argument(
        "--bank-mode", choices=("target-requant", "fixed-output"),
        default="target-requant",
    )
    calibrate.add_argument("--require-model-sidecar", action="store_true")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--catalog", type=Path)
    calibrate.add_argument("--build-env", type=Path)
    calibrate.set_defaults(function=_run_calibrate)

    validate = subparsers.add_parser(
        "validate", help="validate a calibrated profile against a post-build run"
    )
    validate.add_argument("--profile", type=Path, required=True)
    validate.add_argument("--log", type=Path, action="append", required=True)
    validate.add_argument("--minimum-mean-recall", type=float, default=0.90)
    validate.add_argument("--minimum-head-recall", type=float, default=0.50)
    validate.add_argument("--maximum-score-mismatch", type=float, default=0.0)
    validate.add_argument("--require-quality", action="store_true")
    validate.add_argument("--require-recall", action="store_true")
    validate.add_argument("--require-model-sidecar", action="store_true")
    validate.add_argument("--output", type=Path)
    validate.set_defaults(function=_run_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
