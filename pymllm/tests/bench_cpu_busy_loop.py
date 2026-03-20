#!/usr/bin/env python3
"""Benchmark: CPU busy-loop vs brief-poll in the scheduler event loop.

Simulates the scheduler's decode loop (poll → "forward" → poll → ...)
and measures CPU usage under both strategies.

Usage:
    python pymllm/tests/bench_cpu_busy_loop.py

What to look for:
    - "CPU usage" percentage: spin-poll should be ~100%, brief-poll should be <10%
    - "Wall time" should be similar (brief-poll adds ~1ms per iteration)
    - "Throughput" (iterations/sec) shows the latency cost of the brief poll
"""

import os
import time

import zmq


def run_loop(poller, sock, poll_timeout_ms: int, duration_s: float = 2.0):
    """Run the scheduler-style poll loop for *duration_s* seconds.

    The loop body does NO simulated work — this isolates the poll overhead,
    which is exactly what happens in the real scheduler between GPU kernel
    launches (the CPU thread is free while the GPU computes; it's the poll
    call that either spins or yields).

    Returns (wall_time, cpu_time, iterations).
    """
    iterations = 0
    t0_wall = time.monotonic()
    t0_cpu = time.process_time()
    deadline = t0_wall + duration_s

    while time.monotonic() < deadline:
        # Poll for new requests (this is where CPU spins or yields)
        timeout = poll_timeout_ms
        while True:
            events = dict(poller.poll(timeout=timeout))
            if sock not in events:
                break
            timeout = 0  # drain remaining
            sock.recv(zmq.NOBLOCK)  # consume message
        iterations += 1

    wall = time.monotonic() - t0_wall
    cpu = time.process_time() - t0_cpu
    return wall, cpu, iterations


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    addr = f"inproc://bench-{os.getpid()}"
    sock.bind(addr)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    duration = 3.0  # seconds per test

    print("=" * 64)
    print("Scheduler CPU Busy-Loop Benchmark")
    print("=" * 64)
    print(f"Each test runs for {duration:.0f}s simulating the scheduler poll loop")
    print(f"(poll for requests → loop back, no simulated GPU work)")
    print()

    # --- Spin poll (timeout=0) ---
    print("Running SPIN POLL (timeout=0) ...")
    spin_wall, spin_cpu, spin_iters = run_loop(poller, sock, 0, duration)
    spin_pct = 100.0 * spin_cpu / max(spin_wall, 1e-9)
    spin_throughput = spin_iters / max(spin_wall, 1e-9)

    # --- Brief poll (timeout=1ms) ---
    print("Running BRIEF POLL (timeout=1ms) ...")
    brief_wall, brief_cpu, brief_iters = run_loop(poller, sock, 1, duration)
    brief_pct = 100.0 * brief_cpu / max(brief_wall, 1e-9)
    brief_throughput = brief_iters / max(brief_wall, 1e-9)

    sock.close()
    ctx.term()

    # --- Results ---
    print()
    print("-" * 64)
    print(f"{'Metric':<30} {'Spin (before)':>15} {'Brief (after)':>15}")
    print("-" * 64)
    print(f"{'Wall time (s)':<30} {spin_wall:>15.3f} {brief_wall:>15.3f}")
    print(f"{'CPU time (s)':<30} {spin_cpu:>15.3f} {brief_cpu:>15.3f}")
    print(f"{'CPU usage (%)':<30} {spin_pct:>14.1f}% {brief_pct:>14.1f}%")
    print(f"{'Iterations':<30} {spin_iters:>15d} {brief_iters:>15d}")
    print(f"{'Throughput (iter/s)':<30} {spin_throughput:>15.1f} {brief_throughput:>15.1f}")
    print("-" * 64)

    reduction = spin_pct - brief_pct
    throughput_cost = 100.0 * (1 - brief_throughput / max(spin_throughput, 1)) if spin_throughput > 0 else 0
    print()
    print(f"CPU usage reduction:   {reduction:+.1f} percentage points")
    print(f"Throughput cost:       {throughput_cost:.1f}% fewer iterations/sec")
    print()
    if reduction > 20:
        print("RESULT: Significant CPU savings with negligible throughput cost.")
    elif reduction > 5:
        print("RESULT: Moderate CPU savings.")
    else:
        print("RESULT: Minimal difference (forward pass dominates loop time).")


if __name__ == "__main__":
    main()
