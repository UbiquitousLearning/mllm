"""
DetokenizerProcess -- subprocess that converts token IDs back to text.

Receives ``BatchTokenIDOut``-style dicts from the SchedulerProcess,
detokenizes them, and forwards the decoded strings to the
RequestResponseProcess.
"""

import logging
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

import zmq

from pymllm.orchestrator.ipc_utils import create_zmq_socket

logger = logging.getLogger(__name__)


class DetokenizerProcess:
    """Runs inside a subprocess.  Detokenizes finished outputs."""

    def __init__(
        self,
        recv_from_scheduler_addr: str,
        send_to_rr_addr: str,
    ):
        self._recv_from_scheduler_addr = recv_from_scheduler_addr
        self._send_to_rr_addr = send_to_rr_addr

        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_scheduler: Optional[zmq.Socket] = None
        self._send_to_rr: Optional[zmq.Socket] = None

        # TODO: initialise the tokenizer (needed for decode)
        self._tokenizer = None

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
        self._send_to_rr = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_rr_addr,
            bind=False,
        )

    def event_loop(self) -> None:
        """Infinite loop: recv token IDs -> detokenize -> send text to RR."""
        logger.info("DetokenizerProcess event loop started")
        while True:
            token_id_out = self._recv_from_scheduler.recv_pyobj()
            str_out = self._detokenize(token_id_out)
            self._send_to_rr.send_pyobj(str_out)

    # ------------------------------------------------------------------
    # Detokenization (placeholder)
    # ------------------------------------------------------------------

    def _detokenize(self, token_id_out: Dict[str, Any]) -> Dict[str, Any]:
        """Convert token IDs to text.

        TODO: replace with real tokenizer.decode() call and incremental
        detokenization logic.
        """
        output_ids: List[int] = token_id_out.get("output_token_ids", [])
        # placeholder: join ids as string
        text = ""  # TODO: self._tokenizer.decode(output_ids)
        return {
            "rid": token_id_out.get("rid"),
            "text": text,
            "output_token_ids": output_ids,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._recv_from_scheduler is not None:
            self._recv_from_scheduler.close()
        if self._send_to_rr is not None:
            self._send_to_rr.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_detokenizer_process(
    recv_from_scheduler_addr: str,
    send_to_rr_addr: str,
    pipe_writer: Connection,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = DetokenizerProcess(recv_from_scheduler_addr, send_to_rr_addr)
    proc.init_sockets()

    pipe_writer.send({"status": "ready", "process": "detokenizer"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
