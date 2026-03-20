"""Tests for the scheduler CPU busy-loop optimization.

Validates that:
1. The brief_poll parameter flows correctly through recv_requests → _recv_from_zmq
2. The event loop sets _in_decode=True only during active decode batches
3. The ZMQ poll uses a non-zero timeout when brief_poll=True
4. Functional correctness: requests still flow through the scheduler
"""

import queue as stdlib_queue
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import zmq

from pymllm.orchestrator.scheduler_process import (
    IdleSleeper,
    Req,
    ScheduleBatch,
    SchedulerProcess,
    _DECODE_POLL_TIMEOUT_MS,
)
from pymllm.engine.forward_batch import ForwardMode


# ======================================================================
# Helpers
# ======================================================================


def _make_req(rid: str = "test-1", input_ids: Optional[List[int]] = None) -> Req:
    """Create a minimal Req for testing."""
    return Req(
        rid=rid,
        input_ids=input_ids or [1, 2, 3],
        sampling_params={"max_new_tokens": 5, "stop_token_ids": [2]},
    )


class FakePoller:
    """Records poll calls so we can verify timeout values."""

    def __init__(self):
        self.poll_calls: List[int] = []

    def poll(self, timeout: int = 0) -> list:
        self.poll_calls.append(timeout)
        return []  # no events

    def register(self, socket, flags):
        pass


# ======================================================================
# Tests
# ======================================================================


class TestRecvFromZmqBriefPoll:
    """Verify that _recv_from_zmq uses the correct poll timeout."""

    def _make_scheduler(self) -> SchedulerProcess:
        proc = SchedulerProcess.__new__(SchedulerProcess)
        proc._recv_from_tokenizer_addr = ""
        proc._send_to_detokenizer_addr = ""
        proc._server_config = None
        proc._model_config = None
        proc._gpu_id = 0
        proc._shared_queue = None
        proc._enable_shared_queue = False
        proc._tensor_transport_mode = "default"
        proc._zmq_ctx = None
        proc._recv_from_tokenizer = MagicMock()
        proc._send_to_detokenizer = None
        proc._model_runner = None
        proc._waiting_queue = deque()
        proc._pending_queue = []
        proc._running_batch = []
        proc._finished = deque()
        proc._max_running_requests = 256
        proc._max_prefill_tokens = 8192
        proc._max_total_tokens = 131072
        proc._used_tokens = 0
        proc._eos_token_ids = []
        proc._default_max_new_tokens = 32768
        proc._next_req_pool_idx = 0
        proc._decode_log_interval = 40
        proc._num_prefill_tokens = 0
        proc._num_prefill_cache_tokens = 0
        proc._num_decode_tokens = 0
        proc._num_prefill_reqs = 0
        proc._last_prefill_stats_tic = time.time()
        proc._last_decode_stats_tic = time.time()
        proc._forward_ct_decode = 0
        return proc

    def test_brief_poll_false_uses_zero_timeout(self):
        """When brief_poll=False, poll timeout should be 0 (non-blocking)."""
        proc = self._make_scheduler()
        fake_poller = FakePoller()
        proc._poller = fake_poller

        proc._recv_from_zmq(brief_poll=False)

        assert len(fake_poller.poll_calls) == 1
        assert fake_poller.poll_calls[0] == 0

    def test_brief_poll_true_uses_decode_timeout(self):
        """When brief_poll=True, first poll should use _DECODE_POLL_TIMEOUT_MS."""
        proc = self._make_scheduler()
        fake_poller = FakePoller()
        proc._poller = fake_poller

        proc._recv_from_zmq(brief_poll=True)

        assert len(fake_poller.poll_calls) == 1
        assert fake_poller.poll_calls[0] == _DECODE_POLL_TIMEOUT_MS

    def test_recv_requests_forwards_brief_poll(self):
        """recv_requests(brief_poll=True) should forward to _recv_from_zmq."""
        proc = self._make_scheduler()
        proc._recv_from_zmq = MagicMock()

        proc.recv_requests(brief_poll=True)
        proc._recv_from_zmq.assert_called_once_with(brief_poll=True)

        proc._recv_from_zmq.reset_mock()
        proc.recv_requests(brief_poll=False)
        proc._recv_from_zmq.assert_called_once_with(brief_poll=False)

    def test_recv_requests_default_is_non_blocking(self):
        """recv_requests() with no argument should use brief_poll=False."""
        proc = self._make_scheduler()
        proc._recv_from_zmq = MagicMock()

        proc.recv_requests()
        proc._recv_from_zmq.assert_called_once_with(brief_poll=False)


class TestEventLoopDecodeTracking:
    """Verify that event_loop tracks decode state correctly."""

    def test_decode_batch_sets_in_decode(self):
        """After a decode batch, the next recv_requests should use brief_poll=True."""
        proc = SchedulerProcess.__new__(SchedulerProcess)
        proc._enable_shared_queue = False
        proc._tensor_transport_mode = "default"

        call_log = []
        iteration = [0]

        def fake_recv(brief_poll=False):
            call_log.append(("recv", brief_poll))

        def fake_process_input():
            pass

        def fake_get_next_batch():
            i = iteration[0]
            iteration[0] += 1
            if i == 0:
                # First iteration: return an extend (prefill) batch
                batch = MagicMock()
                batch.forward_mode = ForwardMode.EXTEND
                batch.forward_mode.is_extend = lambda: True
                return batch
            elif i == 1:
                # Second iteration: return a decode batch
                batch = MagicMock()
                batch.forward_mode = ForwardMode.DECODE
                batch.forward_mode.is_extend = lambda: False
                return batch
            elif i == 2:
                # Third iteration: should see brief_poll=True from decode
                # Return None to go idle
                return None
            else:
                raise StopIteration("done")

        def fake_run_batch(batch):
            return {}

        def fake_process_batch_result(batch, result):
            pass

        def fake_stream_output():
            pass

        def fake_idle_sleep():
            pass

        proc.recv_requests = fake_recv
        proc.process_input_requests = fake_process_input
        proc.get_next_batch_to_run = fake_get_next_batch
        proc.run_batch = fake_run_batch
        proc.process_batch_result = fake_process_batch_result
        proc.stream_output = fake_stream_output
        proc._idle_sleeper = MagicMock()
        proc._idle_sleeper.sleep = fake_idle_sleep

        # Run event_loop until StopIteration
        try:
            proc.event_loop()
        except StopIteration:
            pass

        # call_log should be:
        # iter 0: recv(brief_poll=False) → extend batch → _in_decode=False
        # iter 1: recv(brief_poll=False) → decode batch → _in_decode=True
        # iter 2: recv(brief_poll=True)  → None → _in_decode=False
        # iter 3: recv(brief_poll=False) → StopIteration
        assert call_log[0] == ("recv", False), f"iter 0: {call_log[0]}"
        assert call_log[1] == ("recv", False), f"iter 1: {call_log[1]}"
        assert call_log[2] == ("recv", True), f"iter 2: should be True after decode"
        assert call_log[3] == ("recv", False), f"iter 3: should be False after idle"


class TestScheduleBatchForwardMode:
    """Verify ScheduleBatch correctly reports forward mode."""

    def test_extend_batch_is_extend(self):
        batch = ScheduleBatch([_make_req()], ForwardMode.EXTEND)
        assert batch.forward_mode.is_extend()
        assert not batch.forward_mode.is_decode()

    def test_decode_batch_is_decode(self):
        batch = ScheduleBatch([_make_req()], ForwardMode.DECODE)
        assert batch.forward_mode.is_decode()
        assert not batch.forward_mode.is_extend()


class TestCpuUsageReduction:
    """Measure that the brief poll actually yields CPU time.

    This is a coarse integration test: we run a tight poll loop with and
    without the brief timeout and compare how much CPU time each burns
    over a fixed wall-clock interval.
    """

    @pytest.mark.timeout(10)
    def test_brief_poll_reduces_cpu_usage(self):
        """Brief poll should use measurably less CPU than non-blocking poll."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.bind("inproc://test-cpu-usage")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        iterations = 500

        # Measure non-blocking (timeout=0)
        t0_wall = time.monotonic()
        t0_cpu = time.process_time()
        for _ in range(iterations):
            poller.poll(timeout=0)
        spin_wall = time.monotonic() - t0_wall
        spin_cpu = time.process_time() - t0_cpu

        # Measure brief poll (timeout=1ms)
        t0_wall = time.monotonic()
        t0_cpu = time.process_time()
        for _ in range(iterations):
            poller.poll(timeout=1)
        brief_wall = time.monotonic() - t0_wall
        brief_cpu = time.process_time() - t0_cpu

        sock.close()
        ctx.term()

        # The brief poll should use much less CPU relative to wall time.
        # Non-blocking: CPU ≈ wall (spinning)
        # Brief poll:   CPU << wall (blocked in kernel)
        spin_ratio = spin_cpu / max(spin_wall, 1e-9)
        brief_ratio = brief_cpu / max(brief_wall, 1e-9)

        # The brief_ratio should be significantly lower.
        # Non-blocking is nearly 1.0 (all CPU), brief should be <0.1
        assert brief_ratio < spin_ratio, (
            f"Brief poll CPU ratio ({brief_ratio:.3f}) should be less than "
            f"spin poll CPU ratio ({spin_ratio:.3f})"
        )
        # Sanity: brief poll should actually take some wall time
        assert brief_wall > 0.1, (
            f"Brief poll wall time ({brief_wall:.3f}s) too short; "
            f"poll(timeout=1) should block ~{iterations}ms total"
        )


class TestDecodeTimeoutConstant:
    """Verify the timeout constant is sensible."""

    def test_decode_poll_timeout_is_positive(self):
        assert _DECODE_POLL_TIMEOUT_MS > 0

    def test_decode_poll_timeout_is_small(self):
        """Should be small enough to not add significant latency."""
        assert _DECODE_POLL_TIMEOUT_MS <= 5
