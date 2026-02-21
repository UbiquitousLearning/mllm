"""
TokenizerProcess -- subprocess that tokenizes incoming raw requests.

Receives raw requests from RequestResponseProcess via ZMQ, tokenizes them,
and forwards the tokenized payloads to the SchedulerProcess.
"""

import logging
from multiprocessing.connection import Connection
from typing import Any, Dict, List

import zmq

from pymllm.orchestrator.ipc_utils import create_zmq_socket

logger = logging.getLogger(__name__)


class TokenizerProcess:
    """Runs inside a subprocess spawned by ``torch.multiprocessing``."""

    def __init__(
        self,
        recv_from_rr_addr: str,
        send_to_scheduler_addr: str,
    ):
        self._recv_from_rr_addr = recv_from_rr_addr
        self._send_to_scheduler_addr = send_to_scheduler_addr

        self._zmq_ctx: zmq.Context = None
        self._recv_from_rr: zmq.Socket = None
        self._send_to_scheduler: zmq.Socket = None

        # TODO: initialise the actual tokenizer (HuggingFace / custom)
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_sockets(self) -> None:
        self._zmq_ctx = zmq.Context()
        self._recv_from_rr = create_zmq_socket(
            self._zmq_ctx,
            zmq.PULL,
            self._recv_from_rr_addr,
            bind=False,
        )
        self._send_to_scheduler = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_scheduler_addr,
            bind=True,
        )

    def event_loop(self) -> None:
        """Infinite loop: recv raw request -> tokenize -> send to scheduler."""
        logger.info("TokenizerProcess event loop started")
        while True:
            raw_request: Dict[str, Any] = self._recv_from_rr.recv_pyobj()
            tokenized = self._tokenize(raw_request)
            self._send_to_scheduler.send_pyobj(tokenized)

    # ------------------------------------------------------------------
    # Tokenization (placeholder)
    # ------------------------------------------------------------------

    def _tokenize(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single raw request and return the tokenized payload.

        TODO: replace with real tokenizer call.
        """
        text = raw_request.get("text", "")
        # placeholder: produce fake token ids
        input_ids: List[int] = []  # TODO: self._tokenizer.encode(text)
        return {
            **raw_request,
            "input_ids": input_ids,
        }

    def shutdown(self) -> None:
        if self._recv_from_rr is not None:
            self._recv_from_rr.close()
        if self._send_to_scheduler is not None:
            self._send_to_scheduler.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_tokenizer_process(
    recv_from_rr_addr: str,
    send_to_scheduler_addr: str,
    pipe_writer: Connection,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = TokenizerProcess(recv_from_rr_addr, send_to_scheduler_addr)
    proc.init_sockets()

    # Signal readiness to the parent process
    pipe_writer.send({"status": "ready", "process": "tokenizer"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
