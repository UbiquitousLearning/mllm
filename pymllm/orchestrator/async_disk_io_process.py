"""
AsyncDiskIoProcess -- optional subprocess for asynchronous disk I/O.

Handles weight loading, checkpoint saving, or other heavy disk operations
without blocking the scheduler or model runner.
"""

import logging
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional

import zmq

from pymllm.orchestrator.ipc_utils import create_zmq_socket

logger = logging.getLogger(__name__)


class AsyncDiskIoProcess:
    """Runs inside a subprocess.  Performs disk I/O on behalf of the scheduler."""

    def __init__(self, recv_addr: str):
        self._recv_addr = recv_addr

        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_sock: Optional[zmq.Socket] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_sockets(self) -> None:
        self._zmq_ctx = zmq.Context()
        self._recv_sock = create_zmq_socket(
            self._zmq_ctx, zmq.PULL, self._recv_addr, bind=True,
        )

    def event_loop(self) -> None:
        """Infinite loop: recv I/O request -> execute -> (optionally reply)."""
        logger.info("AsyncDiskIoProcess event loop started")
        while True:
            io_request: Dict[str, Any] = self._recv_sock.recv_pyobj()
            self._handle(io_request)

    # ------------------------------------------------------------------
    # I/O handling (placeholder)
    # ------------------------------------------------------------------

    def _handle(self, io_request: Dict[str, Any]) -> None:
        """Dispatch an I/O request.

        TODO: implement weight loading, checkpoint save, etc.
        """
        kind = io_request.get("kind")
        logger.debug("AsyncDiskIoProcess received request kind=%s", kind)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._recv_sock is not None:
            self._recv_sock.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_async_disk_io_process(
    recv_addr: str,
    pipe_writer: Connection,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = AsyncDiskIoProcess(recv_addr)
    proc.init_sockets()

    pipe_writer.send({"status": "ready", "process": "async_disk_io"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
