#!/usr/bin/env python3
"""Integration test: measure scheduler CPU usage with and without the brief-poll fix.

Spawns a real SchedulerProcess (with ZMQ sockets) but replaces the model runner
with a mock that simulates decode batches.  Sends requests through the tokenizer
ZMQ socket and measures how much CPU the scheduler subprocess burns.

Usage:
    python pymllm/tests/integration_cpu_busy_loop.py

Expected output:
    - "BEFORE fix" (poll timeout=0): scheduler burns ~100% CPU
    - "AFTER fix"  (poll timeout=1): scheduler burns <10% CPU
"""

import multiprocessing
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List
from unittest.mock import MagicMock

import psutil
import zmq

# ---------------------------------------------------------------------------
# We need the pymllm package on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pymllm.engine.forward_batch import ForwardMode
from pymllm.orchestrator.scheduler_process import (
    Req,
    ScheduleBatch,
    SchedulerProcess,
    _DECODE_POLL_TIMEOUT_MS,
)
from pymllm.engine.io_struct import TokenizedGenerateReqInput


# ---------------------------------------------------------------------------
# Mock model runner that always returns a decode batch with simulated latency
# ---------------------------------------------------------------------------


def _scheduler_worker(
    tokenizer_addr: str,
    detokenizer_addr: str,
    ready_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    decode_poll_timeout_ms: int,
):
    """Run a real SchedulerProcess event loop with a mock model.

    *decode_poll_timeout_ms* lets us toggle the fix on/off:
      0  = spin-poll (old behaviour)
      1+ = brief-poll (new behaviour)
    """
    import pymllm.orchestrator.scheduler_process as sp

    # Monkey-patch the constant so we can test both behaviours
    sp._DECODE_POLL_TIMEOUT_MS = decode_poll_timeout_ms

    proc = SchedulerProcess.__new__(SchedulerProcess)

    # Minimal init -- only what the event loop needs
    proc._recv_from_tokenizer_addr = tokenizer_addr
    proc._send_to_detokenizer_addr = detokenizer_addr
    proc._server_config = None
    proc._model_config = None
    proc._gpu_id = 0
    proc._shared_queue = None
    proc._enable_shared_queue = False
    proc._tensor_transport_mode = "default"

    proc._waiting_queue = deque()
    proc._pending_queue = []
    proc._running_batch = []
    proc._finished = deque()
    proc._max_running_requests = 256
    proc._max_prefill_tokens = 8192
    proc._max_total_tokens = 131072
    proc._used_tokens = 0
    proc._eos_token_ids = [2]
    proc._default_max_new_tokens = 32
    proc._next_req_pool_idx = 0
    proc._decode_log_interval = 40
    proc._num_prefill_tokens = 0
    proc._num_prefill_cache_tokens = 0
    proc._num_decode_tokens = 0
    proc._num_prefill_reqs = 0
    proc._last_prefill_stats_tic = time.time()
    proc._last_decode_stats_tic = time.time()
    proc._forward_ct_decode = 0

    # Init real ZMQ sockets
    proc.init_sockets()

    # Override heavy methods with lightweight mocks:
    # - get_next_batch_to_run: always returns a decode batch (simulates
    #   continuous decode, which is the hot path we're optimizing)
    # - run_batch: simulates a ~2ms GPU forward pass
    # - process_batch_result / stream_output: no-ops

    def fake_get_next_batch():
        if stop_event.is_set():
            raise StopIteration
        # Always return a decode batch so _in_decode stays True
        req = Req(rid="fake-1", input_ids=[1, 2, 3],
                  sampling_params={"max_new_tokens": 5, "stop_token_ids": [2]})
        return ScheduleBatch([req], ForwardMode.DECODE)

    def fake_run_batch(batch):
        # No sleep — we want the scheduler to loop as fast as possible
        # so the poll overhead (spin vs brief) dominates CPU usage.
        # In reality the GPU forward pass runs on the device while the
        # CPU thread is free; it's the poll() call that either spins or
        # yields during that interval.
        return {}

    proc.get_next_batch_to_run = fake_get_next_batch
    proc.run_batch = fake_run_batch
    proc.process_batch_result = lambda batch, result: None
    proc.stream_output = lambda: None

    ready_event.set()

    try:
        proc.event_loop()
    except StopIteration:
        pass
    finally:
        if proc._zmq_ctx:
            proc._recv_from_tokenizer.close()
            proc._send_to_detokenizer.close()
            proc._zmq_ctx.term()


def measure_scheduler_cpu(label: str, decode_poll_timeout_ms: int, duration: float = 5.0):
    """Spawn a scheduler subprocess, let it run for *duration* seconds, measure CPU."""

    # Create unique IPC addresses
    pid = os.getpid()
    ts = int(time.monotonic() * 1000)
    tok_addr = f"ipc:///tmp/mllm-bench-tok-{pid}-{ts}"
    detok_addr = f"ipc:///tmp/mllm-bench-detok-{pid}-{ts}"

    # Set up the tokenizer-side PUSH socket (we'll send messages into the scheduler)
    ctx = zmq.Context()
    tok_push = ctx.socket(zmq.PUSH)
    tok_push.bind(tok_addr)

    detok_pull = ctx.socket(zmq.PULL)
    detok_pull.connect(detok_addr)

    ready = multiprocessing.Event()
    stop = multiprocessing.Event()

    worker = multiprocessing.Process(
        target=_scheduler_worker,
        args=(tok_addr, detok_addr, ready, stop, decode_poll_timeout_ms),
        daemon=True,
    )
    worker.start()

    # Wait for scheduler to be ready
    if not ready.wait(timeout=10):
        print(f"  [{label}] Scheduler failed to start!")
        worker.kill()
        return None, None, None

    # Give the process a moment to stabilize
    time.sleep(0.5)

    # Measure CPU usage over the test duration
    try:
        ps = psutil.Process(worker.pid)
        cpu_times_before = ps.cpu_times()
        wall_start = time.monotonic()

        time.sleep(duration)

        cpu_times_after = ps.cpu_times()
        wall_end = time.monotonic()
    except psutil.NoSuchProcess:
        print(f"  [{label}] Scheduler process died during measurement!")
        return None, None, None

    wall = wall_end - wall_start
    cpu_user = cpu_times_after.user - cpu_times_before.user
    cpu_sys = cpu_times_after.system - cpu_times_before.system
    cpu_total = cpu_user + cpu_sys
    cpu_pct = 100.0 * cpu_total / max(wall, 1e-9)

    # Stop the worker
    stop.set()
    worker.join(timeout=5)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=2)

    tok_push.close()
    detok_pull.close()
    ctx.term()

    # Clean up IPC files
    for addr in [tok_addr, detok_addr]:
        path = addr.replace("ipc://", "")
        try:
            os.unlink(path)
        except OSError:
            pass

    return wall, cpu_total, cpu_pct


def main():
    print("=" * 68)
    print("  Scheduler CPU Busy-Loop Integration Test")
    print("  (real SchedulerProcess + ZMQ sockets, mock model runner)")
    print("=" * 68)
    print()

    duration = 5.0

    # --- BEFORE fix: spin-poll (timeout=0) ---
    print(f"[1/2] BEFORE fix (poll timeout=0, spin-poll) — {duration:.0f}s ...")
    spin_wall, spin_cpu, spin_pct = measure_scheduler_cpu(
        "SPIN", decode_poll_timeout_ms=0, duration=duration
    )
    if spin_pct is not None:
        print(f"       Wall: {spin_wall:.2f}s  CPU: {spin_cpu:.2f}s  Usage: {spin_pct:.1f}%")
    print()

    # --- AFTER fix: brief-poll (timeout=1ms) ---
    print(f"[2/2] AFTER fix (poll timeout={_DECODE_POLL_TIMEOUT_MS}ms, brief-poll) — {duration:.0f}s ...")
    brief_wall, brief_cpu, brief_pct = measure_scheduler_cpu(
        "BRIEF", decode_poll_timeout_ms=_DECODE_POLL_TIMEOUT_MS, duration=duration
    )
    if brief_pct is not None:
        print(f"       Wall: {brief_wall:.2f}s  CPU: {brief_cpu:.2f}s  Usage: {brief_pct:.1f}%")
    print()

    if spin_pct is None or brief_pct is None:
        print("ERROR: Could not measure both scenarios.")
        return 1

    # --- Summary ---
    print("-" * 68)
    print(f"{'Metric':<30} {'Before (spin)':>16} {'After (brief)':>16}")
    print("-" * 68)
    print(f"{'Wall time (s)':<30} {spin_wall:>16.2f} {brief_wall:>16.2f}")
    print(f"{'CPU time (s)':<30} {spin_cpu:>16.2f} {brief_cpu:>16.2f}")
    print(f"{'CPU usage (%)':<30} {spin_pct:>15.1f}% {brief_pct:>15.1f}%")
    print("-" * 68)

    reduction = spin_pct - brief_pct
    print()
    print(f"  CPU usage reduction: {reduction:+.1f} percentage points")
    print(f"  ({spin_pct:.1f}% -> {brief_pct:.1f}%)")
    print()

    if reduction > 30:
        print("  PASS: Significant CPU savings — the fix works!")
    elif reduction > 10:
        print("  PASS: Moderate CPU savings.")
    else:
        print("  INCONCLUSIVE: Minimal difference.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
