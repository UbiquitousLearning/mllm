"""
SchedulerProcess -- the central scheduling hub.

Receives tokenized requests from the TokenizerProcess, organises them into
batches, dispatches batches to the ModelRunnerProcess for forward passes,
collects results, and streams finished token IDs to the DetokenizerProcess.

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
import time
from collections import deque
from multiprocessing.connection import Connection
from typing import Any, Deque, Dict, List, Optional

import zmq

from pymllm.orchestrator.ipc_utils import create_zmq_socket

logger = logging.getLogger(__name__)


class SchedulerProcess:
    """Runs inside a subprocess.  Central hub that drives the inference loop."""

    def __init__(
        self,
        recv_from_tokenizer_addr: str,
        send_to_model_runner_addr: str,
        recv_from_model_runner_addr: str,
        send_to_detokenizer_addr: str,
    ):
        # ZMQ addresses
        self._recv_from_tokenizer_addr = recv_from_tokenizer_addr
        self._send_to_model_runner_addr = send_to_model_runner_addr
        self._recv_from_model_runner_addr = recv_from_model_runner_addr
        self._send_to_detokenizer_addr = send_to_detokenizer_addr

        # ZMQ runtime objects (initialised in init_sockets)
        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_tokenizer: Optional[zmq.Socket] = None
        self._send_to_model_runner: Optional[zmq.Socket] = None
        self._recv_from_model_runner: Optional[zmq.Socket] = None
        self._send_to_detokenizer: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

        # Request management
        self._waiting_queue: Deque[Dict[str, Any]] = deque()
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
        logger.info("SchedulerProcess event loop started")
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

        Uses ``zmq.Poller`` with a short timeout so the scheduler is never
        stuck waiting when there are batches to run.
        """
        while True:
            events = dict(self._poller.poll(timeout=0))  # non-blocking
            if self._recv_from_tokenizer not in events:
                break
            req = self._recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            self._waiting_queue.append(req)

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
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = SchedulerProcess(
        recv_from_tokenizer_addr,
        send_to_model_runner_addr,
        recv_from_model_runner_addr,
        send_to_detokenizer_addr,
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
