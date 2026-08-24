#!/usr/bin/env python3
"""Validate benchmark logs and summarize matched dense/sparse medians."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


NAME = re.compile(r"(?P<mode>dense|sparse)-p(?P<prompt>\d+)-run\d+\.log$")


def fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for path in args.logs:
        match = NAME.search(path.name)
        if match is None:
            parser.error(f"unexpected log name: {path}")
        text = path.read_text(encoding="utf-8", errors="replace").replace("\0", "")
        bench_lines = [line for line in text.splitlines() if line.startswith("BENCH_RESULT")]
        profile_lines = [line for line in text.splitlines() if line.startswith("QWEN_ATTN_PROFILE")]
        quality_lines = [line for line in text.splitlines() if line.startswith("QUALITY_RESULT")]
        if len(bench_lines) != 1 or len(profile_lines) != 1 or len(quality_lines) != 1:
            parser.error(f"{path}: expected one result, profile, and quality line")
        bench, profile, quality = map(fields, (bench_lines[0], profile_lines[0], quality_lines[0]))
        if quality.get("retrieval_exact") != "1":
            parser.error(f"{path}: retrieval_exact is not 1")
        if "FastRPC redirect" not in text:
            parser.error(f"{path}: session redirect was not observed")
        if re.search(r"FastRPC.*(?:error|failed)|QNN.*(?:error|failed)", text, re.I):
            parser.error(f"{path}: FastRPC/QNN error found")
        mode = match.group("mode")
        expected_attention = "dense" if mode == "dense" else "hmx-topk"
        if bench.get("attention_mode") != expected_attention:
            parser.error(f"{path}: unexpected attention mode")
        expected_main_cpu = "7" if mode == "dense" else "2"
        if bench.get("main_cpu") != expected_main_cpu:
            parser.error(
                f"{path}: expected main_cpu={expected_main_cpu} for {mode}"
            )
        worker_observed = "[CPU_ATTN_WORKER] actual_cpu=7" in text
        if mode == "dense" and worker_observed:
            parser.error(f"{path}: dense baseline unexpectedly used attention worker")
        if mode == "sparse" and not worker_observed:
            parser.error(f"{path}: sparse attention worker placement was not observed")
        key = (int(match.group("prompt")), mode)
        runs.setdefault(key, []).append(
            (float(bench["ttft_ms"]), float(profile["profiled_attention_wall_ms"]))
        )

    lines = [
        "prompt_tokens\tdense_ttft_ms\tsparse_ttft_ms\tttft_speedup\t"
        "dense_attention_ms\tsparse_attention_ms\tattention_speedup"
    ]
    for prompt in sorted({prompt for prompt, _ in runs}):
        if (prompt, "dense") not in runs or (prompt, "sparse") not in runs:
            parser.error(f"prompt {prompt}: both dense and sparse runs are required")
        dense = runs[prompt, "dense"]
        sparse = runs[prompt, "sparse"]
        dense_ttft = statistics.median(value[0] for value in dense)
        sparse_ttft = statistics.median(value[0] for value in sparse)
        dense_attention = statistics.median(value[1] for value in dense)
        sparse_attention = statistics.median(value[1] for value in sparse)
        lines.append(
            f"{prompt}\t{dense_ttft:.3f}\t{sparse_ttft:.3f}\t"
            f"{dense_ttft / sparse_ttft:.3f}\t{dense_attention:.3f}\t"
            f"{sparse_attention:.3f}\t{dense_attention / sparse_attention:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
