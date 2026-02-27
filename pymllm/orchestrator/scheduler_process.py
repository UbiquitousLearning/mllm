"""
SchedulerProcess -- the central scheduling hub.

Receives tokenized requests from the TokenizerProcess, organises them into
batches, dispatches batches to the ModelRunnerProcess for forward passes,
collects results, and streams finished token IDs to the DetokenizerProcess.

Supports two modes:
    1. Legacy ZMQ path: Receive TokenizedGenerateReqInput via ZMQ recv_pyobj
    2. Shared queue fast path: Read rid from shared queue and metadata from shared memory

The main ``event_loop`` scheduler flow::

    while True:
        recv_requests()
        process_input_requests()
        batch = get_next_batch_to_run()
        if batch:
            run_batch(batch)
            process_batch_result(batch)
        stream_output()
"""

import logging
import queue as stdlib_queue
import time
from collections import deque
from multiprocessing.connection import Connection
from typing import Any, Deque, Dict, List, Optional

import zmq

from pymllm.engine.io_struct import TokenizedGenerateReqInput
from pymllm.orchestrator.ipc_utils import create_zmq_socket
from pymllm.orchestrator.shared_memory_queue import SharedMemoryManager, TensorQueue

logger = logging.getLogger(__name__)


class SchedulerProcess:
    """Runs inside a subprocess.  Central hub that drives the inference loop."""

    def __init__(
        self,
        recv_from_tokenizer_addr: str,
        send_to_model_runner_addr: str,
        recv_from_model_runner_addr: str,
        send_to_detokenizer_addr: str,
        shared_queue: Optional[TensorQueue] = None,
        enable_shared_queue: bool = False,
    ):
        # ZMQ addresses
        self._recv_from_tokenizer_addr = recv_from_tokenizer_addr
        self._send_to_model_runner_addr = send_to_model_runner_addr
        self._recv_from_model_runner_addr = recv_from_model_runner_addr
        self._send_to_detokenizer_addr = send_to_detokenizer_addr

        # Shared queue configuration
        self._shared_queue = shared_queue
        self._enable_shared_queue = enable_shared_queue

        # ZMQ runtime objects (initialised in init_sockets)
        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_tokenizer: Optional[zmq.Socket] = None
        self._send_to_model_runner: Optional[zmq.Socket] = None
        self._recv_from_model_runner: Optional[zmq.Socket] = None
        self._send_to_detokenizer: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

        # Request management
        self._waiting_queue: Deque[TokenizedGenerateReqInput] = deque()
        self._running_batch: Optional[Dict[str, Any]] = None
        self._finished: List[Dict[str, Any]] = []

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
        self._send_to_model_runner = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_model_runner_addr,
            bind=True,
        )
        self._recv_from_model_runner = create_zmq_socket(
            self._zmq_ctx,
            zmq.PULL,
            self._recv_from_model_runner_addr,
            bind=True,
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

    def event_loop(self) -> None:
        """Infinite scheduling loop."""
        logger.info(
            "SchedulerProcess event loop started (shared_queue=%s)",
            self._enable_shared_queue,
        )
        while True:
            self.recv_requests()
            self.process_input_requests()
            batch = self.get_next_batch_to_run()
            if batch is not None:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            self.stream_output()

    # ------------------------------------------------------------------
    # Step 1: receive tokenized requests (non-blocking)
    # ------------------------------------------------------------------

    def recv_requests(self) -> None:
        """Non-blocking receive of tokenized requests from TokenizerProcess.

        Supports two modes:
        1. Legacy ZMQ: Uses ``zmq.Poller`` with a short timeout
        2. Shared queue: Non-blocking get from multiprocessing.Queue

        Messages are either:
        * A :class:`~pymllm.engine.io_struct.TokenizedGenerateReqInput`
          dataclass – appended to ``_waiting_queue``.
        * A plain abort sentinel dict ``{"rid": ..., "abort": True}`` – handled
          inline by removing the matching rid from the waiting queue.
        """
        if self._enable_shared_queue and self._shared_queue is not None:
            self._recv_from_shared_queue()
        else:
            self._recv_from_zmq()

    def _recv_from_zmq(self) -> None:
        """Receive requests via legacy ZMQ path."""
        while True:
            events = dict(self._poller.poll(timeout=0))  # non-blocking
            if self._recv_from_tokenizer not in events:
                break
            msg = self._recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            # Abort sentinel: plain dict with "abort" key.
            if isinstance(msg, dict) and msg.get("abort"):
                rid = msg.get("rid")
                logger.debug("Scheduler received abort for rid=%s", rid)
                self._waiting_queue = type(self._waiting_queue)(
                    r for r in self._waiting_queue if r.rid != rid
                )
            else:
                self._waiting_queue.append(msg)

    def _recv_from_shared_queue(self) -> None:
        """Receive requests via shared memory + shared queue fast path."""
        while True:
            try:
                # Non-blocking get from shared queue
                rid, shm_name, mm_inputs = self._shared_queue.get(timeout=0.0001)

                # Read metadata from shared memory (and unlink immediately)
                metadata: TokenizedGenerateReqInput = SharedMemoryManager.read_metadata(
                    shm_name, unlink=True
                )

                # Reconstruct the full TokenizedGenerateReqInput with mm_inputs
                full_request = TokenizedGenerateReqInput(
                    rid=metadata.rid,
                    input_text=metadata.input_text,
                    input_ids=metadata.input_ids,
                    mm_inputs=mm_inputs,  # Restored from shared queue
                    sampling_params=metadata.sampling_params,
                    stream=metadata.stream,
                    return_logprob=metadata.return_logprob,
                    logprob_start_len=metadata.logprob_start_len,
                    top_logprobs_num=metadata.top_logprobs_num,
                    lora_path=metadata.lora_path,
                    session_params=metadata.session_params,
                )

                self._waiting_queue.append(full_request)
                logger.debug(f"Received request {rid} from shared queue")

            except stdlib_queue.Empty:
                # No more requests available
                break
            except Exception as e:
                logger.error(f"Error receiving from shared queue: {e}", exc_info=True)
                # Try to cleanup shared memory if possible
                try:
                    if "shm_name" in locals():
                        SharedMemoryManager.cleanup(shm_name)
                except:
                    pass
                break

    # ------------------------------------------------------------------
    # Step 2: process input requests
    # ------------------------------------------------------------------

    def process_input_requests(self) -> None:
        """Pre-process and validate requests sitting in ``_waiting_queue``.

        TODO: attach sampling params, allocate KV-cache slots, etc.
        """
        pass

    # ------------------------------------------------------------------
    # Step 3: build the next batch
    # ------------------------------------------------------------------

    def get_next_batch_to_run(self) -> Optional[Dict[str, Any]]:
        """Select requests from ``_waiting_queue`` and form a batch.

        TODO: implement real batching / scheduling policy.
        """
        if not self._waiting_queue:
            return None

        batch_requests: List[Dict[str, Any]] = []
        # TODO: respect max_running_requests, memory budget, etc.
        while self._waiting_queue:
            batch_requests.append(self._waiting_queue.popleft())

        batch = {
            "requests": batch_requests,
            "batch_id": id(batch_requests),
            "created_at": time.time(),
        }
        return batch

    # ------------------------------------------------------------------
    # Step 4: run the batch via ModelRunnerProcess
    # ------------------------------------------------------------------

    def run_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Send *batch* to ModelRunnerProcess and wait for the result.

        This is a **blocking** call: the scheduler is synchronous with the
        model runner for simplicity.  Overlap scheduling can be added later.
        """
        self._send_to_model_runner.send_pyobj(batch)
        result = self._recv_from_model_runner.recv_pyobj()
        return result

    # ------------------------------------------------------------------
    # Step 5: process batch result
    # ------------------------------------------------------------------

    def process_batch_result(
        self, batch: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Handle the result returned by the ModelRunnerProcess.

        TODO: check completion status (EOS, max_tokens), manage KV-cache,
        split finished vs. unfinished requests.
        """
        finished_requests = result.get("finished", [])
        unfinished_requests = result.get("unfinished", [])

        self._finished.extend(finished_requests)

        # Put unfinished requests back for the next iteration
        for req in unfinished_requests:
            self._waiting_queue.appendleft(req)

    # ------------------------------------------------------------------
    # Step 6: stream output to DetokenizerProcess
    # ------------------------------------------------------------------

    def stream_output(self) -> None:
        """Send finished token-ID outputs to the DetokenizerProcess."""
        while self._finished:
            item = self._finished.pop(0)
            self._send_to_detokenizer.send_pyobj(item)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        for sock in (
            self._recv_from_tokenizer,
            self._send_to_model_runner,
            self._recv_from_model_runner,
            self._send_to_detokenizer,
        ):
            if sock is not None:
                sock.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_scheduler_process(
    recv_from_tokenizer_addr: str,
    send_to_model_runner_addr: str,
    recv_from_model_runner_addr: str,
    send_to_detokenizer_addr: str,
    pipe_writer: Connection,
    shared_queue: Optional[TensorQueue] = None,
    enable_shared_queue: bool = False,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = SchedulerProcess(
        recv_from_tokenizer_addr,
        send_to_model_runner_addr,
        recv_from_model_runner_addr,
        send_to_detokenizer_addr,
        shared_queue=shared_queue,
        enable_shared_queue=enable_shared_queue,
    )
    proc.init_sockets()

    pipe_writer.send({"status": "ready", "process": "scheduler"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
