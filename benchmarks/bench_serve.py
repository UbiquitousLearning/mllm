#!/usr/bin/env python3
"""
Benchmark pymllm server: prefill and decode throughput.

Sends a list of prompts sequentially to the /v1/completions endpoint,
measures Time-To-First-Token (TTFT, i.e. prefill latency) and decode
throughput for each request, and prints a summary table.

Usage:
    # Start the server first, then:
    python benchmarks/bench_server.py

    # Custom server URL:
    python benchmarks/bench_server.py --url http://192.168.1.100:30000

    # Adjust max tokens:
    python benchmarks/bench_server.py --max-tokens 200
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List

import requests


# ---------------------------------------------------------------------------
# Benchmark prompts — short / medium / long to test different prefill sizes
# ---------------------------------------------------------------------------

PROMPTS: List[str] = [
    # ~256 tokens
    "You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "the API contracts, and how the services communicate with each other. Also discuss deployment discuss",

    "You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "the API contracts, and how the services communicate with each other. Also discuss deployment "
    "strategies, monitoring, and fault tolerance. You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "the API contracts, and how the services communicate with each other.",

    "You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "the API contracts, and how the services communicate with each other. Also discuss deployment "
    "strategies, monitoring, and fault tolerance. You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "the API contracts, and how the services communicate with each other. Also discuss deployment "
    "strategies, monitoring, and fault tolerance. You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "the API contracts, and how the services communicate with each other. Also discuss deployment "
    "strategies, monitoring, and fault tolerance. You are an expert historian. Please provide a comprehensive overview of the Industrial Revolution, "
    "including its origins in 18th century Britain, the key technological innovations such as the steam "
    "engine and spinning jenny, the social and economic impacts on the working class, urbanization trends, "
    "As a senior software architect, design a complete microservices architecture for an e-commerce "
    "platform. The platform should support user authentication and authorization, product catalog "
    "management with search and filtering, shopping cart functionality, order processing and payment "
    "integration, inventory management, shipping and logistics tracking, customer reviews and ratings, "
    "recommendation engine, analytics dashboard, and notification services. For each microservice, "
    "describe its responsibilities, the technology stack you would choose, the database type, "
    "describe its responsibilities"
]



# ---------------------------------------------------------------------------
# Result data
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float          # time to first token (prefill latency), client-side
    prefill_ms: float       # estimated pure prefill time = ttft - one_decode_step
    total_time_ms: float    # total request time
    decode_time_ms: float   # total_time - ttft
    prefill_tps: float      # prompt_tokens / prefill_ms
    decode_tps: float       # generated_tokens / decode_time


@dataclass
class BenchmarkSummary:
    results: List[RequestResult] = field(default_factory=list)

    def add(self, r: RequestResult):
        self.results.append(r)

    def print_table(self):
        print()
        print("=" * 120)
        print(f"{'#':>3}  {'Prompt Tok':>10}  {'Gen Tok':>8}  "
              f"{'TTFT(ms)':>10}  {'Prefill(ms)':>12}  {'Prefill t/s':>12}  "
              f"{'Decode(ms)':>10}  {'Decode t/s':>11}  "
              f"{'Total(ms)':>10}")
        print("-" * 120)

        for i, r in enumerate(self.results):
            print(
                f"{i+1:>3}  {r.prompt_tokens:>10}  {r.generated_tokens:>8}  "
                f"{r.ttft_ms:>10.1f}  {r.prefill_ms:>12.1f}  {r.prefill_tps:>12.1f}  "
                f"{r.decode_time_ms:>10.1f}  {r.decode_tps:>11.1f}  "
                f"{r.total_time_ms:>10.1f}"
            )

        print("-" * 120)

        if not self.results:
            return

        total_prompt = sum(r.prompt_tokens for r in self.results)
        total_gen = sum(r.generated_tokens for r in self.results)
        total_time = sum(r.total_time_ms for r in self.results)

        print(f"SUM  {total_prompt:>10}  {total_gen:>8}  "
              f"{'':>10}  {'':>12}  {'':>12}  "
              f"{'':>10}  {'':>11}  {total_time:>10.1f}")
        print("=" * 120)


# ---------------------------------------------------------------------------
# Streaming request with TTFT measurement
# ---------------------------------------------------------------------------

def run_one_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    """Send a streaming completions request and measure TTFT + decode speed."""

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_token = None
    chunk_count = 0         # SSE data chunks received (≈ generated tokens)
    api_prompt_tokens = 0   # from usage field if server reports it
    api_gen_tokens = 0      # from usage field if server reports it

    resp = requests.post(
        f"{url}/v1/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=300,
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

        # Record TTFT on the very first data chunk
        if t_first_token is None:
            t_first_token = time.perf_counter()

        chunk_count += 1

        # Extract usage if the server provides it (usually in the last chunk)
        usage = data.get("usage")
        if usage:
            api_prompt_tokens = usage.get("prompt_tokens", api_prompt_tokens)
            api_gen_tokens = usage.get("completion_tokens", api_gen_tokens)

    t_end = time.perf_counter()

    if t_first_token is None:
        t_first_token = t_end

    # Use server-reported counts when available; fall back to chunk count
    prompt_tokens = api_prompt_tokens if api_prompt_tokens > 0 else max(1, len(prompt.split()) * 2)
    generated_tokens = api_gen_tokens if api_gen_tokens > 0 else max(chunk_count, 1)

    ttft_ms = (t_first_token - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    decode_ms = total_ms - ttft_ms

    decode_tps = generated_tokens / (decode_ms / 1000) if decode_ms > 0 else 0

    # Estimate pure prefill time by subtracting one decode step from TTFT.
    # TTFT (client-side) = network_rtt + server_queue + prefill + first_decode_step + network_rtt.
    # On localhost network_rtt ≈ 0.  We estimate first_decode_step ≈ decode_ms / generated_tokens.
    one_decode_step_ms = (decode_ms / generated_tokens) if generated_tokens > 0 else 0
    prefill_ms = max(ttft_ms - one_decode_step_ms, ttft_ms * 0.5)

    prefill_tps = prompt_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0

    return RequestResult(
        prompt=prompt[:60] + ("..." if len(prompt) > 60 else ""),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        total_time_ms=total_ms,
        decode_time_ms=decode_ms,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark pymllm server")
    parser.add_argument("--url", default="http://127.0.0.1:30000",
                        help="Server base URL (default: http://127.0.0.1:30000)")
    parser.add_argument("--model", default="Qwen3.5-2B",
                        help="Model name for the API request")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate per request (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup requests before benchmarking (default: 1)")
    args = parser.parse_args()

    print(f"pymllm Server Benchmark")
    print(f"  URL:         {args.url}")
    print(f"  Model:       {args.model}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Prompts:     {len(PROMPTS)}")
    print()

    # Check server is alive
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        print(f"Server health: {r.status_code}")
    except Exception:
        print(f"Server health: endpoint not found (proceeding anyway)")

    # Warmup — ensures JIT-compiled CUDA kernels (e.g. cuda_chunkwise) are
    # already compiled before benchmark requests start.  On Jetson Orin the
    # first compile can take ~10-15 s; this cost is paid here, not during
    # benchmark measurement.
    if args.warmup > 0:
        print(f"\n--- Warmup ({args.warmup} request(s), first may be slow due to JIT) ---")
        for i in range(args.warmup):
            print(f"  warmup {i+1}/{args.warmup} ...", end=" ", flush=True)
            wr = run_one_request(
                args.url, args.model,
                "Please explain the GDN linear attention mechanism briefly.",
                max_tokens=20, temperature=args.temperature,
            )
            print(f"done ({wr.total_time_ms:.0f} ms)"
                  + (" ← JIT compile included" if wr.ttft_ms > 5000 else ""))

    # Benchmark
    print(f"\n--- Benchmark ({len(PROMPTS)} requests) ---")
    summary = BenchmarkSummary()

    for i, prompt in enumerate(PROMPTS):
        short = prompt[:50] + ("..." if len(prompt) > 50 else "")
        print(f"  [{i+1}/{len(PROMPTS)}] \"{short}\"", flush=True)

        result = run_one_request(
            args.url, args.model, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        summary.add(result)

        print(f"         TTFT={result.ttft_ms:.0f}ms  prefill_est={result.prefill_ms:.0f}ms  "
              f"prefill={result.prefill_tps:.0f} t/s  "
              f"decode={result.decode_tps:.1f} t/s  "
              f"gen={result.generated_tokens} tokens")

    summary.print_table()


if __name__ == "__main__":
    main()
