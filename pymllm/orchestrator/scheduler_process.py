"""
SchedulerProcess -- the central scheduling and inference hub.

Receives tokenized requests from the TokenizerProcess, organises them into
batches, runs model forward passes via the **in-process** model runner,
and streams finished token IDs to the DetokenizerProcess.

Architecture: the scheduler owns the :class:`ModelRunnerProcess` directly
(same process, direct function calls).  GPU resources (KV cache, req pool
slots) are freed immediately when requests finish — no cross-process
communication needed.

Request ingestion supports two modes:
    1. ZMQ path: Receive TokenizedGenerateReqInput via ZMQ recv_pyobj
    2. Shared queue fast path: Read from shared memory + multiprocessing queue

The main ``event_loop``::

    while True:
        recv_requests()
        process_input_requests()
        batch = get_next_batch_to_run()   # also frees finished GPU resources
        if batch:
            result = run_batch(batch)      # direct call to model runner
            process_batch_result(batch, result)
        else:
            idle_sleeper.sleep()           # block until ZMQ data or timeout
        stream_output()
"""

import logging
import os
import queue as stdlib_queue
import time
from collections import deque
from multiprocessing.connection import Connection
from typing import Any, Deque, Dict, List, Optional

import zmq

from pymllm.engine.forward_batch import ForwardMode
from pymllm.engine.io_struct import BatchTokenIDOutput, TokenizedGenerateReqInput
from pymllm.orchestrator.cuda_ipc_transport import (
    TensorTransportMode,
    unwrap_mm_inputs_from_ipc,
)
from pymllm.orchestrator.ipc_utils import create_zmq_socket, setup_subprocess_logging
from pymllm.orchestrator.shared_memory_queue import SharedMemoryManager, TensorQueue

logger = logging.getLogger(__name__)

# Default scheduling limits
_DEFAULT_MAX_RUNNING_REQUESTS = 256
_DEFAULT_IDLE_POLL_TIMEOUT_MS = 1000
_DEFAULT_MAX_PREFILL_TOKENS = 8192
_DEFAULT_MAX_TOTAL_TOKENS = 131072
_DEFAULT_MAX_NEW_TOKENS = 32768

# Brief poll timeout (ms) used between decode batches to avoid 100% CPU spin.
# 1 ms is enough to yield the CPU core to the OS scheduler while adding
# negligible latency (decode steps typically take >1 ms on the GPU anyway).
# Override via MLLM_DECODE_POLL_TIMEOUT_MS env var for testing.
_DECODE_POLL_TIMEOUT_MS = int(os.environ.get("MLLM_DECODE_POLL_TIMEOUT_MS", "1"))


# ======================================================================
# IdleSleeper -- avoid busy-looping when no work is available
# ======================================================================


class IdleSleeper:
    """Block the scheduler thread when idle using ZMQ Poller.

    Avoids 100% CPU spinning when no requests are pending.  The poller
    wakes immediately when data arrives on any registered socket, so
    request latency is not affected.
    """

    def __init__(
        self, sockets: list, poll_timeout_ms: int = _DEFAULT_IDLE_POLL_TIMEOUT_MS
    ):
        self.poller = zmq.Poller()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)
        self.poll_timeout_ms = poll_timeout_ms

    def sleep(self) -> None:
        """Block until data arrives on any registered socket, or timeout."""
        self.poller.poll(self.poll_timeout_ms)


# ======================================================================
# Req -- per-request state tracker
# ======================================================================


class Req:
    """Tracks a single request through its lifecycle (prefill -> decode -> finish).

    Created by :meth:`SchedulerProcess.process_input_requests` from a
    :class:`~pymllm.engine.io_struct.TokenizedGenerateReqInput`.
    """

    __slots__ = (
        "rid",
        "input_ids",
        "input_text",
        "sampling_params",
        "mm_inputs",
        "stream",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        # KV-cache state
        "req_pool_idx",
        "seq_len",
        # Prefix-cache hit (set during scheduling when radix cache is active)
        "prefix_len",
        # Generation state
        "output_ids",
        "finished_reason",
        "is_prefilled",
        # Sampling parameters (parsed)
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_token_ids",
        # Streaming
        "read_offset",
        # Prompt length (for token accounting)
        "prompt_len",
    )

    def __init__(
        self,
        rid: str,
        input_ids: List[int],
        input_text: str = "",
        sampling_params: Optional[Dict[str, Any]] = None,
        mm_inputs: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        return_logprob: bool = False,
        logprob_start_len: int = -1,
        top_logprobs_num: int = 0,
    ):
        self.rid = rid
        self.input_ids = list(input_ids)
        self.input_text = input_text
        self.mm_inputs = mm_inputs
        self.stream = stream
        self.return_logprob = return_logprob
        self.logprob_start_len = logprob_start_len
        self.top_logprobs_num = top_logprobs_num

        # Parse sampling params
        sp = sampling_params or {}
        self.sampling_params = sp
        self.max_new_tokens: int = sp.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
        self.temperature: float = sp.get("temperature", 1.0)
        self.top_p: float = sp.get("top_p", 1.0)
        self.top_k: int = sp.get("top_k", -1)
        self.stop_token_ids: List[int] = list(sp.get("stop_token_ids", []))

        # KV-cache state (assigned during scheduling)
        self.req_pool_idx: int = -1
        self.seq_len: int = len(input_ids)
        # Number of prefix tokens served from the radix/KV cache (0 = no hit).
        # Updated by process_batch_result when the model runner reports a
        # prefix cache hit.  Used in _free_req_resources to correctly
        # release the token budget.
        self.prefix_len: int = 0

        # Generation state
        self.output_ids: List[int] = []
        self.finished_reason: Optional[str] = None
        self.is_prefilled: bool = False

        # Streaming
        self.read_offset: int = 0

        # Prompt length
        self.prompt_len: int = len(input_ids)

    def check_finished(self) -> bool:
        """Check if this request has reached a finish condition.

        Sets ``finished_reason`` and returns True if finished.
        Checks:
        1. Stop token (EOS tokens are merged into stop_token_ids during
           :meth:`SchedulerProcess.process_input_requests`)
        2. ``max_new_tokens`` reached
        """
        if self.finished_reason is not None:
            return True

        if self.output_ids:
            last_token = self.output_ids[-1]
            if last_token in self.stop_token_ids:
                self.finished_reason = "eos"
                return True

        # Check max_new_tokens
        if len(self.output_ids) >= self.max_new_tokens:
            self.finished_reason = "length"
            return True

        return False

    @property
    def is_finished(self) -> bool:
        return self.finished_reason is not None

    def abort(self) -> None:
        """Mark this request as aborted."""
        self.finished_reason = "abort"

    def __repr__(self) -> str:
        return (
            f"Req(rid={self.rid!r}, seq_len={self.seq_len}, "
            f"out={len(self.output_ids)}, finished={self.finished_reason})"
        )


# ======================================================================
# ScheduleBatch -- batch container
# ======================================================================


class ScheduleBatch:
    """Wraps a list of :class:`Req` objects for a single forward pass.

    Provides helpers to assemble the batch dict sent to the ModelRunnerProcess
    in the format expected by :class:`~pymllm.engine.forward_batch.ForwardBatch`.
    """

    def __init__(self, reqs: List[Req], forward_mode: ForwardMode):
        self.reqs = reqs
        self.forward_mode = forward_mode

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def prepare_for_extend(self) -> Dict[str, Any]:
        """Assemble a batch dict for prefill / extend forward pass.

        Returns a dict with flattened ``input_ids``, per-request ``positions``,
        ``req_pool_indices``, ``seq_lens``, ``extend_seq_lens``,
        ``extend_prefix_lens``, and request metadata.

        Note: The scheduler sends the **full** input_ids (no prefix trimming).
        The ModelRunnerProcess performs radix cache prefix matching and
        rebuilds the tensors with actual prefix lengths before the forward
        pass.  The ``extend_prefix_lens`` here are always 0 from the
        scheduler; they serve as placeholders.
        """
        all_input_ids: List[int] = []
        all_positions: List[int] = []
        req_pool_indices: List[int] = []
        seq_lens: List[int] = []
        extend_seq_lens: List[int] = []
        extend_prefix_lens: List[int] = []
        requests_meta: List[Dict[str, Any]] = []

        for req in self.reqs:
            input_len = len(req.input_ids)

            # Send full input_ids; model runner will trim based on prefix
            all_input_ids.extend(req.input_ids)
            all_positions.extend(range(input_len))
            req_pool_indices.append(req.req_pool_idx)
            seq_lens.append(req.seq_len)
            extend_seq_lens.append(input_len)
            extend_prefix_lens.append(0)
            requests_meta.append(
                {
                    "rid": req.rid,
                    "input_ids": req.input_ids,
                    "mm_inputs": req.mm_inputs,
                    "sampling_params": req.sampling_params,
                    "return_logprob": req.return_logprob,
                    "logprob_start_len": req.logprob_start_len,
                    "top_logprobs_num": req.top_logprobs_num,
                }
            )

        return {
            "forward_mode": "extend",
            "batch_size": self.batch_size,
            "input_ids": all_input_ids,
            "positions": all_positions,
            "req_pool_indices": req_pool_indices,
            "seq_lens": seq_lens,
            "extend_seq_lens": extend_seq_lens,
            "extend_prefix_lens": extend_prefix_lens,
            "requests": requests_meta,
            "batch_id": id(self),
            "created_at": time.time(),
        }

    def prepare_for_decode(self) -> Dict[str, Any]:
        """Assemble a batch dict for decode forward pass (one token per request).

        Returns a dict with one input token per request (the last generated
        token), positions at ``seq_len``, and request metadata.
        """
        all_input_ids: List[int] = []
        all_positions: List[int] = []
        req_pool_indices: List[int] = []
        seq_lens: List[int] = []
        requests_meta: List[Dict[str, Any]] = []

        for req in self.reqs:
            # For decode, the input is the last generated token
            if req.output_ids:
                all_input_ids.append(req.output_ids[-1])
            else:
                # Fallback: last input token (shouldn't happen normally)
                all_input_ids.append(req.input_ids[-1])
            all_positions.append(req.seq_len)
            req_pool_indices.append(req.req_pool_idx)
            seq_lens.append(req.seq_len)
            requests_meta.append(
                {
                    "rid": req.rid,
                    "sampling_params": req.sampling_params,
                    "return_logprob": req.return_logprob,
                    "logprob_start_len": req.logprob_start_len,
                    "top_logprobs_num": req.top_logprobs_num,
                }
            )

        return {
            "forward_mode": "decode",
            "batch_size": self.batch_size,
            "input_ids": all_input_ids,
            "positions": all_positions,
            "req_pool_indices": req_pool_indices,
            "seq_lens": seq_lens,
            "requests": requests_meta,
            "batch_id": id(self),
            "created_at": time.time(),
        }

    def to_batch_dict(self) -> Dict[str, Any]:
        """Build the batch dict appropriate for the current forward mode."""
        if self.forward_mode.is_extend():
            return self.prepare_for_extend()
        else:
            return self.prepare_for_decode()

    def __repr__(self) -> str:
        return f"ScheduleBatch(mode={self.forward_mode.name}, size={self.batch_size})"


# ======================================================================
# SchedulerProcess
# ======================================================================


class SchedulerProcess:
    """Runs inside a subprocess.  Central hub that drives the inference loop."""

    def __init__(
        self,
        recv_from_tokenizer_addr: str,
        send_to_detokenizer_addr: str,
        server_config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        gpu_id: int = 0,
        shared_queue: Optional[TensorQueue] = None,
        enable_shared_queue: bool = False,
        tensor_transport_mode: TensorTransportMode = "default",
        # Scheduling limits
        max_running_requests: int = _DEFAULT_MAX_RUNNING_REQUESTS,
        max_prefill_tokens: int = _DEFAULT_MAX_PREFILL_TOKENS,
        max_total_tokens: int = _DEFAULT_MAX_TOTAL_TOKENS,
        eos_token_ids: Optional[List[int]] = None,
        default_max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ):
        # ZMQ addresses (tokenizer + detokenizer only)
        self._recv_from_tokenizer_addr = recv_from_tokenizer_addr
        self._send_to_detokenizer_addr = send_to_detokenizer_addr

        # Model config (for in-process model runner)
        self._server_config = server_config
        self._model_config = model_config
        self._gpu_id = gpu_id

        # Shared queue configuration
        self._shared_queue = shared_queue
        self._enable_shared_queue = enable_shared_queue
        self._tensor_transport_mode = tensor_transport_mode

        # ZMQ runtime objects (initialised in init_sockets)
        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_tokenizer: Optional[zmq.Socket] = None
        self._send_to_detokenizer: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

        # In-process model runner (initialised in init_model)
        self._model_runner = None

        # Request management -- three-stage pipeline
        self._waiting_queue: Deque[TokenizedGenerateReqInput] = deque()
        self._pending_queue: List[Req] = []
        self._running_batch: List[Req] = []
        self._finished: Deque[Dict[str, Any]] = deque()

        # Scheduling limits
        self._max_running_requests = max_running_requests
        self._max_prefill_tokens = max_prefill_tokens

        # KV-cache token budget (simplified single-GPU tracking).
        self._max_total_tokens = max_total_tokens
        self._used_tokens: int = 0

        # EOS token(s) for finish detection
        self._eos_token_ids: List[int] = list(eos_token_ids) if eos_token_ids else []

        # Default max_new_tokens (from model config or fallback)
        self._default_max_new_tokens = default_max_new_tokens

        # Monotonic request-slot counter (simplified; no GPU pool access)
        self._next_req_pool_idx: int = 0

        # ------ Throughput metrics ------
        # How often (in decode batches) to log throughput stats.
        self._decode_log_interval: int = (
            server_config.decode_log_interval
            if server_config is not None
            and hasattr(server_config, "decode_log_interval")
            else 40
        )
        # Accumulators reset at each log interval
        self._num_prefill_tokens: int = 0  # new prefill tokens (excluding cache hits)
        self._num_prefill_cache_tokens: int = 0  # prefill tokens served from cache
        self._num_decode_tokens: int = 0  # generated decode tokens
        self._num_prefill_reqs: int = 0  # prefill requests count
        # Timestamps for throughput calculation
        self._last_prefill_stats_tic: float = time.time()
        self._last_decode_stats_tic: float = time.time()
        # Forward pass counters
        self._forward_ct_decode: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_sockets(self) -> None:
        self._zmq_ctx = zmq.Context()

        self._recv_from_tokenizer = create_zmq_socket(
            self._zmq_ctx,
            zmq.PULL,
            self._recv_from_tokenizer_addr,
            bind=False,
        )
        self._send_to_detokenizer = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_detokenizer_addr,
            bind=True,
        )

        # Poller for non-blocking recv from tokenizer
        self._poller = zmq.Poller()
        self._poller.register(self._recv_from_tokenizer, zmq.POLLIN)

        # Idle sleeper: blocks the event loop when no batch is ready,
        # wakes immediately on incoming ZMQ messages.
        self._idle_sleeper = IdleSleeper([self._recv_from_tokenizer])

    def init_model(self) -> None:
        """Create and initialise the in-process model runner.

        Must be called after ``init_sockets`` and inside the subprocess
        (after spawn) since it performs CUDA initialisation.
        """
        from pymllm.orchestrator.model_runner_process import ModelRunnerProcess

        self._model_runner = ModelRunnerProcess(
            gpu_id=self._gpu_id,
            server_config=self._server_config,
            model_config=self._model_config,
        )
        self._model_runner.init_model()
        logger.info("In-process model runner initialised on GPU %d", self._gpu_id)

    def event_loop(self) -> None:
        """Infinite scheduling loop.

        When decode batches are active the loop would otherwise spin at
        100 % CPU doing non-blocking ZMQ polls between GPU forward passes.
        We track whether the previous iteration ran a decode batch and, if
        so, use a brief poll timeout (default 1 ms) in ``recv_requests``
        so the OS can schedule other work on this core.
        """
        logger.info(
            "SchedulerProcess event loop started (shared_queue=%s, transport=%s)",
            self._enable_shared_queue,
            self._tensor_transport_mode,
        )
        _in_decode = False
        while True:
            self.recv_requests(brief_poll=_in_decode)
            self.process_input_requests()
            batch = self.get_next_batch_to_run()
            if batch is not None:
                _in_decode = not batch.forward_mode.is_extend()
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                _in_decode = False
                # No work available -- sleep until a new request arrives
                # on the ZMQ socket (or timeout).  Avoids busy-looping.
                self._idle_sleeper.sleep()
            self.stream_output()

    # ------------------------------------------------------------------
    # Step 1: receive tokenized requests (non-blocking)
    # ------------------------------------------------------------------

    def recv_requests(self, brief_poll: bool = False) -> None:
        """Non-blocking receive of tokenized requests from TokenizerProcess.

        Supports two modes:
        1. Legacy ZMQ: Uses ``zmq.Poller`` with a short timeout
        2. Shared queue: Non-blocking get from multiprocessing.Queue

        When *brief_poll* is ``True`` (typically during active decode), the
        first poll uses a small timeout (``_DECODE_POLL_TIMEOUT_MS``) instead
        of zero.  This yields the CPU core to the OS scheduler between decode
        batches while adding negligible latency.

        Messages are either:
        * A :class:`~pymllm.engine.io_struct.TokenizedGenerateReqInput`
          dataclass - appended to ``_waiting_queue``.
        * A plain abort sentinel dict ``{"rid": ..., "abort": True}`` - handled
          inline by removing the matching rid from the waiting queue.
        """
        if self._enable_shared_queue and self._shared_queue is not None:
            self._recv_from_shared_queue(brief_poll=brief_poll)
        else:
            self._recv_from_zmq(brief_poll=brief_poll)

    def _recv_from_zmq(self, brief_poll: bool = False) -> None:
        """Receive requests via legacy ZMQ path."""
        # On the first poll, use a brief timeout if requested (decode path)
        # to yield the CPU.  After draining the first message, switch to
        # non-blocking for any remaining queued messages.
        poll_timeout = _DECODE_POLL_TIMEOUT_MS if brief_poll else 0
        while True:
            events = dict(self._poller.poll(timeout=poll_timeout))
            if self._recv_from_tokenizer not in events:
                break
            poll_timeout = 0  # drain remaining messages without blocking
            msg = self._recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            # Abort sentinel: plain dict with "abort" key.
            if isinstance(msg, dict) and msg.get("abort"):
                rid = msg.get("rid")
                logger.debug("Scheduler received abort for rid=%s", rid)
                self._waiting_queue = type(self._waiting_queue)(
                    r for r in self._waiting_queue if r.rid != rid
                )
                # Also abort from pending queue
                self._abort_request(rid)
            else:
                self._waiting_queue.append(msg)

    def _recv_from_shared_queue(self, brief_poll: bool = False) -> None:
        """Receive requests via shared memory + shared queue fast path.

        After reading a ``(rid, shm_name, mm_inputs)`` tuple from the queue:
        1. The tokenized metadata is read from the POSIX shared memory segment.
        2. If CUDA IPC is enabled, ``mm_inputs`` may contain
           :class:`~pymllm.orchestrator.cuda_ipc_transport.CudaIpcTensorTransportProxy`
           or :class:`~pymllm.orchestrator.cuda_ipc_transport.TransportProxyTensor`
           objects that are reconstructed by calling
           :func:`~pymllm.orchestrator.cuda_ipc_transport.unwrap_mm_inputs_from_ipc`.
           This step also increments sync flags so the sender can recycle pool chunks.
        3. A full ``TokenizedGenerateReqInput`` is assembled and appended to
           ``_waiting_queue``.
        """
        # Use a slightly longer timeout on the first get when in decode mode
        # to yield CPU; subsequent gets use a short timeout to drain the queue.
        get_timeout = _DECODE_POLL_TIMEOUT_MS / 1000.0 if brief_poll else 0.002
        while True:
            try:
                rid, shm_name, mm_inputs = self._shared_queue.get(timeout=get_timeout)
                get_timeout = 0.002  # drain remaining without extra delay

                # Read metadata from shared memory (and unlink immediately)
                metadata: TokenizedGenerateReqInput = SharedMemoryManager.read_metadata(
                    shm_name, unlink=True
                )

                # Reconstruct GPU tensors from CUDA IPC handles (if any)
                if self._tensor_transport_mode in ("cuda_ipc", "cuda_ipc_pool"):
                    mm_inputs = unwrap_mm_inputs_from_ipc(mm_inputs)

                # Reassemble the full request
                full_request = TokenizedGenerateReqInput(
                    rid=metadata.rid,
                    input_text=metadata.input_text,
                    input_ids=metadata.input_ids,
                    mm_inputs=mm_inputs,
                    sampling_params=metadata.sampling_params,
                    stream=metadata.stream,
                    return_logprob=metadata.return_logprob,
                    logprob_start_len=metadata.logprob_start_len,
                    top_logprobs_num=metadata.top_logprobs_num,
                    lora_path=metadata.lora_path,
                    session_params=metadata.session_params,
                )

                self._waiting_queue.append(full_request)
                logger.debug("Received request %s from shared queue", rid)

            except stdlib_queue.Empty:
                break
            except Exception as exc:
                logger.error(
                    "Error receiving from shared queue: %s", exc, exc_info=True
                )
                try:
                    if "shm_name" in locals():
                        SharedMemoryManager.cleanup(shm_name)
                except Exception:
                    pass
                break

    # ------------------------------------------------------------------
    # Step 2: process input requests
    # ------------------------------------------------------------------

    def process_input_requests(self) -> None:
        """Convert raw :class:`TokenizedGenerateReqInput` in ``_waiting_queue``
        into :class:`Req` objects and move them to ``_pending_queue``.

        For each request:
        1. Parse sampling params (max_new_tokens, temperature, top_p, top_k,
           stop_token_ids with defaults from EOS token).
        2. Create a ``Req`` object.
        3. Move from ``_waiting_queue`` to ``_pending_queue``.
        """
        while self._waiting_queue:
            raw = self._waiting_queue.popleft()

            # Merge EOS token into stop_token_ids if not already present
            sp = dict(raw.sampling_params) if raw.sampling_params else {}
            # Inject model-aware default for max_new_tokens when not provided
            if "max_new_tokens" not in sp:
                sp["max_new_tokens"] = self._default_max_new_tokens
            stop_ids = list(sp.get("stop_token_ids", []))
            for eid in self._eos_token_ids:
                if eid not in stop_ids:
                    stop_ids.append(eid)
            sp["stop_token_ids"] = stop_ids

            req = Req(
                rid=raw.rid,
                input_ids=raw.input_ids,
                input_text=raw.input_text,
                sampling_params=sp,
                mm_inputs=raw.mm_inputs,
                stream=raw.stream,
                return_logprob=raw.return_logprob,
                logprob_start_len=raw.logprob_start_len,
                top_logprobs_num=raw.top_logprobs_num,
            )
            self._pending_queue.append(req)
            logger.debug("Processed input request %s (len=%d)", req.rid, req.seq_len)

    # ------------------------------------------------------------------
    # Step 3: build the next batch
    # ------------------------------------------------------------------

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """Implements continuous batching with two phases.

        1. **Filter finished**: Remove finished requests from
           ``_running_batch`` and free their token budget.
        2. **Schedule new prefills**: From ``_pending_queue``, admit
           requests that fit within the token budget and
           ``max_running_requests``.
        3. **Build batch**:
           - If new prefill requests exist -> EXTEND batch
           - Else if running decode requests exist -> DECODE batch
           - Else -> None (idle)

        Note on prefix cache: The actual prefix matching is done by the
        ModelRunnerProcess (which owns the RadixCache).  The scheduler
        uses ``input_len`` as a conservative budget estimate.  The model
        runner reports back actual ``prefix_len`` in results, and the
        scheduler adjusts ``_used_tokens`` accordingly in
        ``process_batch_result``.
        """
        # Phase 1: filter finished requests from running batch
        still_running: List[Req] = []
        for req in self._running_batch:
            if req.is_finished:
                self._model_runner._free_rid_resources(req.rid)
                self._free_req_resources(req)
            else:
                still_running.append(req)
        self._running_batch = still_running

        # Phase 2: schedule new prefill requests from pending queue
        new_prefill: List[Req] = []
        remaining_pending: List[Req] = []
        prefill_token_budget = self._max_prefill_tokens

        for req in self._pending_queue:
            input_len = len(req.input_ids)
            total_running = len(self._running_batch) + len(new_prefill)

            # Check capacity constraints.
            # We reserve the full input_len as KV budget (conservative).
            # If the model runner finds a prefix cache hit, some tokens
            # won't need new KV allocation; the budget is corrected in
            # process_batch_result.
            can_fit_request = total_running < self._max_running_requests
            can_fit_tokens = (self._used_tokens + input_len) <= self._max_total_tokens
            can_fit_prefill = input_len <= prefill_token_budget

            if can_fit_request and can_fit_tokens and can_fit_prefill:
                # Allocate req pool slot
                req.req_pool_idx = self._next_req_pool_idx
                self._next_req_pool_idx += 1
                # Reserve token budget (full input_len as conservative estimate)
                self._used_tokens += input_len
                prefill_token_budget -= input_len
                new_prefill.append(req)
                logger.debug(
                    "Scheduled prefill for %s (len=%d, used=%d/%d)",
                    req.rid,
                    input_len,
                    self._used_tokens,
                    self._max_total_tokens,
                )
            else:
                remaining_pending.append(req)

        self._pending_queue = remaining_pending

        # Phase 3: build batch
        if new_prefill:
            return ScheduleBatch(new_prefill, ForwardMode.EXTEND)
        elif self._running_batch:
            return ScheduleBatch(self._running_batch, ForwardMode.DECODE)
        else:
            return None

    # ------------------------------------------------------------------
    # Step 4: run the batch via ModelRunnerProcess
    # ------------------------------------------------------------------

    def run_batch(self, batch: ScheduleBatch) -> Dict[str, Any]:
        """Execute the batch via the in-process model runner.

        Direct function call — no ZMQ serialisation overhead.
        """
        batch_dict = batch.to_batch_dict()
        return self._model_runner._forward_batch(batch_dict)

    # ------------------------------------------------------------------
    # Step 5: process batch result
    # ------------------------------------------------------------------

    def process_batch_result(
        self, batch: ScheduleBatch, result: Dict[str, Any]
    ) -> None:
        """Handle the result returned by the ModelRunnerProcess.

        For each request in the result:
        1. Update ``prefix_len`` from the model runner's radix cache hit.
        2. Adjust ``_used_tokens`` if a prefix cache hit was found (the
           scheduler over-reserved during scheduling).
        3. Append new token(s) to ``req.output_ids``.
        4. Increment ``req.seq_len``.
        5. Call ``req.check_finished()`` (EOS token, max_new_tokens).
        6. If prefill request: mark ``req.is_prefilled = True``, move to
           running batch for decode.
        7. If finished: collect for output, free KV-cache budget.
        """
        # Build a rid -> Req lookup for the batch
        rid_to_req: Dict[str, Req] = {req.rid: req for req in batch.reqs}

        # The result may contain per-request outputs in "finished" and
        # "unfinished" lists, or a flat "outputs" list. Handle both.
        output_items: List[Dict[str, Any]] = []
        output_items.extend(result.get("finished", []))
        output_items.extend(result.get("unfinished", []))
        if "outputs" in result:
            output_items.extend(result["outputs"])

        for out in output_items:
            rid = out.get("rid")
            req = rid_to_req.get(rid)
            if req is None:
                logger.warning("Result for unknown rid=%s, skipping", rid)
                continue

            # Update prefix_len from model runner's radix cache matching.
            # The model runner reports the actual prefix_len it found.
            # The scheduler originally reserved full input_len in
            # get_next_batch_to_run; correct the over-reservation now.
            if "prefix_len" in out and batch.forward_mode.is_extend():
                actual_prefix_len = out["prefix_len"]
                if actual_prefix_len > req.prefix_len:
                    saved = actual_prefix_len - req.prefix_len
                    req.prefix_len = actual_prefix_len
                    # Give back the over-reserved tokens.  The model runner
                    # reused cached KV for `saved` tokens, so those tokens
                    # do not consume new KV pool slots.
                    self._used_tokens = max(0, self._used_tokens - saved)
                    logger.info(
                        "Prefix cache hit for rid=%s: %d tokens reused, "
                        "budget adjusted by -%d (used=%d/%d)",
                        rid,
                        actual_prefix_len,
                        saved,
                        self._used_tokens,
                        self._max_total_tokens,
                    )

            # Append generated token(s)
            new_token_ids = out.get("output_token_ids", [])
            if isinstance(new_token_ids, int):
                new_token_ids = [new_token_ids]
            req.output_ids.extend(new_token_ids)
            req.seq_len += len(new_token_ids)

            # Update token budget for newly generated tokens
            self._used_tokens += len(new_token_ids)

            # Check finish conditions (EOS tokens already in stop_token_ids)
            req.check_finished()

        # Process batch requests based on forward mode
        if batch.forward_mode.is_extend():
            # Prefill batch: mark as prefilled and route
            for req in batch.reqs:
                req.is_prefilled = True
                if req.is_finished:
                    self._collect_finished_output(req)
                    self._model_runner._free_rid_resources(req.rid)
                    self._free_req_resources(req)
                else:
                    self._running_batch.append(req)

            # --- Accumulate prefill metrics ---
            total_input = 0
            total_cached = 0
            for req in batch.reqs:
                total_input += req.prompt_len
                total_cached += req.prefix_len
            self._num_prefill_tokens += total_input - total_cached
            self._num_prefill_cache_tokens += total_cached
            self._num_prefill_reqs += len(batch.reqs)
            self._log_prefill_stats()
        else:
            # Decode batch: check finish and collect
            new_running: List[Req] = []
            for req in batch.reqs:
                if req.is_finished:
                    self._collect_finished_output(req)
                    self._model_runner._free_rid_resources(req.rid)
                    self._free_req_resources(req)
                else:
                    new_running.append(req)
            self._running_batch = new_running

            # --- Accumulate decode metrics ---
            self._num_decode_tokens += batch.batch_size  # 1 token per request
            self._forward_ct_decode += 1
            if (
                self._decode_log_interval > 0
                and self._forward_ct_decode % self._decode_log_interval == 0
            ):
                self._log_decode_stats()

    # ------------------------------------------------------------------
    # Step 6: stream output to DetokenizerProcess
    # ------------------------------------------------------------------

    def stream_output(self) -> None:
        """Send finished/streaming outputs to the DetokenizerProcess.

        Produces :class:`~pymllm.engine.io_struct.BatchTokenIDOutput`-compatible
        dicts.  For streaming requests, intermediate tokens are also sent.
        """
        # Collect streaming outputs from running requests (skip aborted)
        for req in self._running_batch:
            if req.finished_reason == "abort":
                continue
            if req.stream and len(req.output_ids) > req.read_offset:
                decode_ids = req.output_ids[req.read_offset :]
                output = {
                    "rids": [req.rid],
                    "finished_reasons": [None],
                    "decode_ids": decode_ids,
                    "read_offsets": [req.read_offset],
                    "output_ids": list(req.output_ids),
                    "skip_special_tokens": [True],
                    "prompt_tokens": [req.prompt_len],
                    "completion_tokens": [len(req.output_ids)],
                }
                req.read_offset = len(req.output_ids)
                self._send_to_detokenizer.send_pyobj(output)

        # Send finished outputs
        while self._finished:
            item = self._finished.popleft()
            self._send_to_detokenizer.send_pyobj(item)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_prefill_stats(self) -> None:
        """Log prefill throughput at INFO level (called after each prefill batch)."""
        now = time.time()
        elapsed = now - self._last_prefill_stats_tic
        self._last_prefill_stats_tic = now

        if elapsed > 0:
            input_throughput = self._num_prefill_tokens / elapsed
        else:
            input_throughput = 0.0

        logger.info(
            "Prefill batch: %d reqs, "
            "new tokens: %d, "
            "cached tokens: %d, "
            "input throughput: %.2f token/s",
            self._num_prefill_reqs,
            self._num_prefill_tokens,
            self._num_prefill_cache_tokens,
            input_throughput,
        )
        # Reset accumulators
        self._num_prefill_tokens = 0
        self._num_prefill_cache_tokens = 0
        self._num_prefill_reqs = 0

    def _log_decode_stats(self) -> None:
        """Log decode throughput at INFO level (called every decode_log_interval batches)."""
        now = time.time()
        elapsed = now - self._last_decode_stats_tic
        self._last_decode_stats_tic = now

        if elapsed > 0:
            gen_throughput = self._num_decode_tokens / elapsed
        else:
            gen_throughput = 0.0

        logger.info(
            "Decode: %d steps, "
            "gen tokens: %d, "
            "running: %d reqs, "
            "gen throughput: %.2f token/s",
            self._forward_ct_decode,
            self._num_decode_tokens,
            len(self._running_batch),
            gen_throughput,
        )
        # Reset accumulators
        self._num_decode_tokens = 0
        self._forward_ct_decode = 0

    def _collect_finished_output(self, req: Req) -> None:
        """Build a finished output dict and add it to ``_finished``."""
        decode_ids = req.output_ids[req.read_offset :]
        output: Dict[str, Any] = {
            "rids": [req.rid],
            "finished_reasons": [req.finished_reason],
            "decode_ids": decode_ids,
            "read_offsets": [req.read_offset],
            "output_ids": list(req.output_ids),
            "skip_special_tokens": [True],
            "prompt_tokens": [req.prompt_len],
            "completion_tokens": [len(req.output_ids)],
        }
        self._finished.append(output)
        logger.debug(
            "Request %s finished: reason=%s, tokens=%d",
            req.rid,
            req.finished_reason,
            len(req.output_ids),
        )

    def _free_req_resources(self, req: Req) -> None:
        """Release KV-cache token budget for a finished request.

        The budget was charged as follows:
        - At scheduling: ``+input_len`` (full prompt as conservative estimate)
        - After prefix correction: ``-prefix_len`` (cached prefix doesn't need
          new KV allocation; model runner manages those via radix cache)
        - At each decode step: ``+1`` per generated token

        So the net charge for this request is:
            ``(input_len - prefix_len) + num_decode_tokens``
            = ``seq_len - prefix_len``

        We release exactly that amount.
        """
        tokens_to_free = req.seq_len - req.prefix_len
        self._used_tokens = max(0, self._used_tokens - tokens_to_free)
        req.req_pool_idx = -1

    def _abort_request(self, rid: str) -> None:
        """Abort a request by rid from pending or running queues."""
        # Remove from pending queue
        self._pending_queue = [r for r in self._pending_queue if r.rid != rid]
        # Abort in running batch
        for req in self._running_batch:
            if req.rid == rid:
                req.abort()
                break

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._model_runner is not None:
            self._model_runner.shutdown()
        for sock in (
            self._recv_from_tokenizer,
            self._send_to_detokenizer,
        ):
            if sock is not None:
                sock.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_scheduler_process(
    recv_from_tokenizer_addr: str,
    send_to_detokenizer_addr: str,
    pipe_writer: Connection,
    shared_queue: Optional[TensorQueue] = None,
    enable_shared_queue: bool = False,
    tensor_transport_mode: TensorTransportMode = "default",
    log_level: str = "info",
    default_max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    eos_token_ids: Optional[List[int]] = None,
    server_config: Optional[Any] = None,
    model_config: Optional[Any] = None,
    gpu_id: int = 0,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``.

    The scheduler process now also owns the model runner,
    so model initialisation happens here.
    """
    setup_subprocess_logging(log_level)

    # Extract scheduling limits from server_config (fall back to defaults)
    max_running = _DEFAULT_MAX_RUNNING_REQUESTS
    max_prefill = _DEFAULT_MAX_PREFILL_TOKENS
    max_total = _DEFAULT_MAX_TOTAL_TOKENS
    if server_config is not None:
        if getattr(server_config, "max_running_requests", None) is not None:
            max_running = server_config.max_running_requests
        if getattr(server_config, "max_prefill_tokens", None) is not None:
            max_prefill = server_config.max_prefill_tokens
        if getattr(server_config, "max_total_tokens", None) is not None:
            max_total = server_config.max_total_tokens

    proc = SchedulerProcess(
        recv_from_tokenizer_addr,
        send_to_detokenizer_addr,
        server_config=server_config,
        model_config=model_config,
        gpu_id=gpu_id,
        shared_queue=shared_queue,
        enable_shared_queue=enable_shared_queue,
        tensor_transport_mode=tensor_transport_mode,
        max_running_requests=max_running,
        max_prefill_tokens=max_prefill,
        max_total_tokens=max_total,
        default_max_new_tokens=default_max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    proc.init_sockets()
    proc.init_model()

    pipe_writer.send({"status": "ready", "process": "scheduler"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
