#!/usr/bin/env python3
"""
Benchmark pymllm-server prefill (prompt-processing) and decode (token-gen)
throughput, in the style of ``llama-bench``.

Methodology (per prefill size P, repeated ``--repeat`` times):
  * PREFILL (pp):  send a prompt of ~P tokens with ``max_tokens=1`` and measure
    pure prefill latency.  Prefill time is taken from the server's own
    ``debug_timing`` block when available (start the server with
    ``--server.enable_debug_timing``); otherwise it falls back to the
    client-side TTFT minus one decode step.
  * DECODE (tg):   send the same prompt with ``max_tokens=D`` (``--decode-tokens``)
    and measure *steady-state* decode throughput from the inter-token arrival
    timestamps, discarding the first token (which still carries prefill +
    first-step warmup).  This mirrors how llama-bench reports tg.

Results are aggregated over repetitions (mean / std / min / max) and printed as
a table, plus an overall summary that is directly comparable to llama.cpp's
``pp`` / ``tg`` numbers.

Usage:
    # Start the server first (debug timing gives the most accurate prefill):
    #   PYMLLM_GDN_EXTEND_BACKEND=cuda_chunkwise pymllm-server \
    #       --server.model_path /path/to/Qwen3.5-2B --server.enable_debug_timing
    python benchmarks/bench_serve.py
    python benchmarks/bench_serve.py --prefill-sizes 256,512,1024 --decode-tokens 128 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

# A neutral filler sentence (~16 tokens) repeated to build prompts of a target
# token length.  Decode prompts additionally ask for a long answer so greedy
# decoding does not stop early before ``--decode-tokens`` tokens are produced.
_FILLER = (
    "The quick brown fox jumps over the lazy dog while the curious cat watches "
    "from the warm windowsill on a bright and cloudless autumn morning. "
)
_DECODE_INSTRUCTION = (
    "Write an extremely long, detailed, and continuous essay. Do not stop early. "
)


def build_prompt(approx_tokens: int, *, for_decode: bool = False) -> str:
    """Build a prompt of roughly ``approx_tokens`` tokens (~0.75 tok/word)."""
    target_words = max(1, int(approx_tokens * 0.78))
    words: List[str] = []
    filler_words = _FILLER.split()
    while len(words) < target_words:
        words.extend(filler_words)
    text = " ".join(words[:target_words])
    if for_decode:
        text = _DECODE_INSTRUCTION + text
    return text


# ---------------------------------------------------------------------------
# Single streaming request
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    prompt_tokens: int
    gen_tokens: int
    ttft_s: float                  # client-side time to first token
    token_times: List[float]       # perf_counter() at each received chunk
    server_prefill_ms: Optional[float] = None  # from debug_timing if present
    server_decode_ms: Optional[float] = None
    server_decode_tps: Optional[float] = None  # decode_phase_output_tps if present

    @property
    def decode_tps(self) -> Optional[float]:
        """Decode throughput, apples-to-apples with llama-bench ``tg``.

        Prefers the server-side decode-loop throughput (excludes HTTP /
        detokenizer / network jitter, like llama-bench's pure-compute tg);
        falls back to client-side inter-token gaps when unavailable.
        """
        if self.server_decode_tps is not None and self.server_decode_tps > 0:
            return self.server_decode_tps
        return self.steady_decode_tps

    @property
    def steady_decode_tps(self) -> Optional[float]:
        """Client-side decode t/s from inter-token gaps, discarding token 1."""
        if len(self.token_times) < 3:
            return None
        span = self.token_times[-1] - self.token_times[1]
        n = len(self.token_times) - 2  # gaps between tokens 2..N
        if span <= 0 or n <= 0:
            return None
        return n / span

    def prefill_tps(self) -> Optional[float]:
        ms = self.prefill_ms()
        if ms is None or ms <= 0:
            return None
        return self.prompt_tokens / (ms / 1000.0)

    def prefill_ms(self) -> Optional[float]:
        if self.server_prefill_ms is not None and self.server_prefill_ms > 0:
            return self.server_prefill_ms
        # Fall back to TTFT minus one steady-state decode step.
        ttft_ms = self.ttft_s * 1000.0
        tps = self.steady_decode_tps
        one_step = (1000.0 / tps) if tps else 0.0
        return max(ttft_ms - one_step, ttft_ms * 0.5)


def stream_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> StreamResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    token_times: List[float] = []
    prompt_tokens = 0
    gen_tokens = 0
    server_prefill_ms: Optional[float] = None
    server_decode_ms: Optional[float] = None
    server_decode_tps: Optional[float] = None

    resp = requests.post(
        f"{url}/v1/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=900,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # A chunk carrying generated text counts as one decode token.
        choices = data.get("choices") or []
        has_text = any(c.get("text") for c in choices)
        if has_text:
            token_times.append(time.perf_counter())

        usage = data.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            gen_tokens = usage.get("completion_tokens", gen_tokens)

        dbg = data.get("debug_timing")
        if dbg:
            if dbg.get("experimental_llm_prefill_ms"):
                server_prefill_ms = dbg["experimental_llm_prefill_ms"]
            if dbg.get("decode_phase_wall_ms"):
                server_decode_ms = dbg["decode_phase_wall_ms"]
            if dbg.get("decode_phase_output_tps"):
                server_decode_tps = dbg["decode_phase_output_tps"]

    ttft_s = (token_times[0] - t0) if token_times else (time.perf_counter() - t0)
    if gen_tokens <= 0:
        gen_tokens = len(token_times)
    if prompt_tokens <= 0:
        prompt_tokens = max(1, int(len(prompt.split()) / 0.78))

    return StreamResult(
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        ttft_s=ttft_s,
        token_times=token_times,
        server_prefill_ms=server_prefill_ms,
        server_decode_ms=server_decode_ms,
        server_decode_tps=server_decode_tps,
    )


def server_timed_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
):
    """Non-streaming request that returns the server's own ``debug_timing``.

    This is the rigorous, llama-bench-aligned path:
      * prefill: ``experimental_llm_prefill_ms`` = pure model-forward time for
        the prompt (no sampling / detokenize / HTTP), matching llama-bench pp.
      * decode : ``decode_phase_output_tps`` = server-side decode-loop
        throughput (no HTTP / network jitter), matching llama-bench tg.

    Returns ``(prompt_tokens, prefill_ms, decode_tps)`` with ``None`` for any
    field the server did not provide.
    """
    r = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=900,
    )
    r.raise_for_status()
    data = r.json()
    dbg = data.get("debug_timing") or {}
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or dbg.get("prefill_tokens") or 0
    return (
        prompt_tokens,
        dbg.get("experimental_llm_prefill_ms"),
        dbg.get("decode_phase_output_tps"),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class SizeStats:
    target_tokens: int
    prompt_tokens: int = 0
    prefill_tps: List[float] = field(default_factory=list)
    decode_tps: List[float] = field(default_factory=list)
    gen_tokens: List[int] = field(default_factory=list)
    server_timing: bool = False          # prefill timing from server
    decode_server_timing: bool = False   # decode timing from server


def _fmt(values: List[float]) -> str:
    if not values:
        return "       n/a"
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:7.1f} ± {s:5.1f}"


def print_report(stats: List[SizeStats], decode_tokens: int):
    print()
    print("=" * 92)
    print(f"{'Prompt tok':>11} | {'Prefill (pp) t/s':>22} | {'Decode (tg) t/s':>22} | {'gen':>5} | pp/tg src")
    print("-" * 92)
    all_pref: List[float] = []
    all_dec: List[float] = []
    for st in stats:
        src = f"{'srv' if st.server_timing else 'cli'}/{'srv' if st.decode_server_timing else 'cli'}"
        gen = f"{int(statistics.mean(st.gen_tokens))}" if st.gen_tokens else "-"
        print(f"{st.prompt_tokens:>11} | {_fmt(st.prefill_tps):>22} | "
              f"{_fmt(st.decode_tps):>22} | {gen:>5} | {src}")
        all_pref.extend(st.prefill_tps)
        all_dec.extend(st.decode_tps)
    print("-" * 92)
    if all_pref:
        print(f"  Overall prefill: mean {statistics.mean(all_pref):.1f} t/s   "
              f"(max {max(all_pref):.1f})")
    if all_dec:
        print(f"  Overall decode : mean {statistics.mean(all_dec):.1f} t/s   "
              f"(max {max(all_dec):.1f})")
    print("=" * 92)
    print("  Reference llama.cpp (2B): prefill ~1300-1400 t/s, decode ~30-40 t/s")
    print("=" * 92)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark pymllm-server (llama-bench style)")
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--prefill-sizes", default="256,512,1024",
                   help="Comma-separated prompt token sizes (default: 256,512,1024)")
    p.add_argument("--decode-tokens", type=int, default=128,
                   help="Tokens to generate when measuring decode (default: 128)")
    p.add_argument("--repeat", type=int, default=5,
                   help="Measured repetitions per size (default: 5)")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup requests before measuring (default: 2)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy/deterministic (default: 0)")
    p.add_argument("--client-timing", action="store_true",
                   help="Force client-side (HTTP) timing instead of the server's "
                        "compute-only debug_timing. Default uses server timing "
                        "(rigorous, llama-bench-aligned) when available.")
    args = p.parse_args()

    sizes = [int(s) for s in args.prefill_sizes.split(",") if s.strip()]

    print("pymllm-server benchmark (llama-bench style)")
    print(f"  URL            : {args.url}")
    print(f"  Model          : {args.model}")
    print(f"  Prefill sizes  : {sizes}")
    print(f"  Decode tokens  : {args.decode_tokens}")
    print(f"  Repeat / warmup: {args.repeat} / {args.warmup}")
    print(f"  Temperature    : {args.temperature}")

    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        print(f"  Server health  : {r.status_code}")
    except Exception:
        print("  Server health  : (no /health, proceeding)")

    # Warmup (pays JIT/compile + CUDA-graph capture costs once).
    if args.warmup > 0:
        print(f"\n--- Warmup ({args.warmup}) ---")
        for i in range(args.warmup):
            wr = stream_request(args.url, args.model,
                                build_prompt(max(sizes), for_decode=True),
                                max_tokens=32, temperature=args.temperature)
            note = " (JIT/capture)" if wr.ttft_s > 5.0 else ""
            print(f"  warmup {i+1}/{args.warmup}: ttft={wr.ttft_s*1000:.0f}ms{note}")

    stats: List[SizeStats] = []
    print(f"\n--- Benchmark ---")
    for size in sizes:
        st = SizeStats(target_tokens=size)
        pp_prompt = build_prompt(size, for_decode=False)
        tg_prompt = build_prompt(size, for_decode=True)
        for rep in range(args.repeat):
            # --- Prefill (pp) ---
            tps = None
            if not args.client_timing:
                # Rigorous path: server compute-only prefill (debug_timing).
                ptok, pms, _ = server_timed_request(
                    args.url, args.model, pp_prompt,
                    max_tokens=1, temperature=args.temperature)
                if pms and pms > 0:
                    st.prompt_tokens = ptok
                    st.server_timing = True
                    tps = ptok / (pms / 1000.0)
            if tps is None:
                # Fallback: client-side TTFT (streaming).
                pp = stream_request(args.url, args.model, pp_prompt,
                                    max_tokens=1, temperature=args.temperature)
                st.prompt_tokens = pp.prompt_tokens
                tps = pp.prefill_tps()
            if tps:
                st.prefill_tps.append(tps)

            # --- Decode (tg) ---
            dtps = None
            gen = args.decode_tokens
            if not args.client_timing:
                # Rigorous path: server-side decode-loop throughput.
                _, _, sdtps = server_timed_request(
                    args.url, args.model, tg_prompt,
                    max_tokens=args.decode_tokens, temperature=args.temperature)
                if sdtps and sdtps > 0:
                    st.decode_server_timing = True
                    dtps = sdtps
            if dtps is None:
                # Fallback: client-side inter-token gaps (streaming).
                tg = stream_request(args.url, args.model, tg_prompt,
                                    max_tokens=args.decode_tokens, temperature=args.temperature)
                dtps = tg.decode_tps
                gen = tg.gen_tokens
            if dtps:
                st.decode_tps.append(dtps)
            st.gen_tokens.append(gen)
            print(f"  [{size:>5} tok] rep {rep+1}/{args.repeat}: "
                  f"prefill={tps or 0:7.1f} t/s  decode={dtps or 0:5.1f} t/s  "
                  f"(prompt={st.prompt_tokens})", flush=True)
        stats.append(st)

    print_report(stats, args.decode_tokens)


if __name__ == "__main__":
    main()
