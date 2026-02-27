"""
ModelRunnerProcess -- subprocess that executes model forward passes.

Receives batches from the SchedulerProcess, runs the model forward + sampling,
and returns the results (logits, next_token_ids) back to the scheduler.
"""

import logging
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

import zmq

from pymllm.orchestrator.ipc_utils import create_zmq_socket

logger = logging.getLogger(__name__)


class ModelRunnerProcess:
    """Runs inside a subprocess.  Owns the model and performs forward passes."""

    def __init__(
        self,
        recv_from_scheduler_addr: str,
        send_to_scheduler_addr: str,
    ):
        self._recv_from_scheduler_addr = recv_from_scheduler_addr
        self._send_to_scheduler_addr = send_to_scheduler_addr

        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_scheduler: Optional[zmq.Socket] = None
        self._send_to_scheduler: Optional[zmq.Socket] = None

        # TODO: initialise model, attention backend, memory pool, etc.
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_sockets(self) -> None:
        self._zmq_ctx = zmq.Context()
        self._recv_from_scheduler = create_zmq_socket(
            self._zmq_ctx,
            zmq.PULL,
            self._recv_from_scheduler_addr,
            bind=False,
        )
        self._send_to_scheduler = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_scheduler_addr,
            bind=False,
        )

    def event_loop(self) -> None:
        """Infinite loop: recv batch -> forward -> sample -> send result."""
        logger.info("ModelRunnerProcess event loop started")
        while True:
            batch = self._recv_from_scheduler.recv_pyobj()
            result = self._forward_batch(batch)
            self._send_to_scheduler.send_pyobj(result)

    # ------------------------------------------------------------------
    # Forward pass (placeholder)
    # ------------------------------------------------------------------

    def _forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run the model forward pass and sampling for *batch*.

        *batch* is a dict produced by ``SchedulerProcess.get_next_batch_to_run``
        whose ``"requests"`` list contains
        :class:`~pymllm.engine.io_struct.TokenizedGenerateReqInput` objects.

        Returns a dict ``{"batch_id": ..., "finished": [...], "unfinished": [...]}``
        where each element of *finished* / *unfinished* is a plain output dict
        containing at least ``"rid"`` and ``"output_token_ids"``.

        TODO: implement real forward pass, logits processing, and sampling.
        """
        requests = batch.get("requests", [])
        finished: List[Dict[str, Any]] = []
        unfinished: List[Dict[str, Any]] = []

        for req in requests:
            # Support both TokenizedGenerateReqInput dataclass (normal path) and
            # legacy plain dicts (defensive).
            rid: str = req.rid if hasattr(req, "rid") else req.get("rid")
            input_ids: List[int] = (
                req.input_ids if hasattr(req, "input_ids") else req.get("input_ids", [])
            )
            mm_inputs: Optional[Dict[str, Any]] = (
                req.mm_inputs if hasattr(req, "mm_inputs") else req.get("mm_inputs")
            )

            # TODO: actual model forward; pass input_ids and mm_inputs to the model.
            next_token_ids: List[int] = []  # placeholder

            output: Dict[str, Any] = {
                "rid": rid,
                "output_token_ids": next_token_ids,
                "finished": True,
            }
            # TODO: check EOS / max_tokens to decide finished vs. unfinished.
            finished.append(output)

        return {
            "batch_id": batch.get("batch_id"),
            "finished": finished,
            "unfinished": unfinished,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._recv_from_scheduler is not None:
            self._recv_from_scheduler.close()
        if self._send_to_scheduler is not None:
            self._send_to_scheduler.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_model_runner_process(
    recv_from_scheduler_addr: str,
    send_to_scheduler_addr: str,
    pipe_writer: Connection,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = ModelRunnerProcess(recv_from_scheduler_addr, send_to_scheduler_addr)
    proc.init_sockets()

    pipe_writer.send({"status": "ready", "process": "model_runner"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
