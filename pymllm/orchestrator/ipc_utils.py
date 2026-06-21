"""ZMQ IPC utilities for inter-process communication.

Provides helpers to generate unique IPC addresses and create pre-configured
ZMQ sockets so that every process uses the same conventions.
"""

import logging
import os
import tempfile
from typing import Optional

import zmq


_IPC_DIR = os.path.join(tempfile.gettempdir(), "pymllm_ipc")


def _ensure_ipc_dir() -> None:
    os.makedirs(_IPC_DIR, exist_ok=True)


def make_ipc_address(name: str, unique_id: Optional[str] = None) -> str:
    """Return an ``ipc://`` address for *name*, optionally scoped by *unique_id*.

    Parameters
    ----------
    name
        Logical channel name, e.g. ``"rr_to_tokenizer"``.
    unique_id
        Per-engine identifier (typically ``str(os.getpid())``) to avoid
        collisions when multiple engines run on the same host.
    """
    _ensure_ipc_dir()
    suffix = f"_{unique_id}" if unique_id else ""
    return f"ipc://{_IPC_DIR}/pymllm_{name}{suffix}"


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: int,
    address: str,
    bind: bool,
) -> zmq.Socket:
    """Create a ZMQ socket, bind or connect it, and return it.

    Parameters
    ----------
    ctx
        A ``zmq.Context`` shared within the process.
    socket_type
        One of ``zmq.PUSH``, ``zmq.PULL``, ``zmq.PAIR``, etc.
    address
        The ``ipc://`` address string.
    bind
        If ``True`` the socket calls ``bind``; otherwise ``connect``.
    """
    sock = ctx.socket(socket_type)
    sock.setsockopt(zmq.LINGER, 0)
    if bind:
        sock.bind(address)
    else:
        sock.connect(address)
    return sock


def close_zmq_socket(sock: zmq.Socket) -> None:
    """Close a ZMQ socket, ignoring errors."""
    try:
        sock.close()
    except zmq.ZMQError:
        pass


def cleanup_ipc_files(unique_id: Optional[str] = None) -> None:
    """Remove IPC socket files for the given engine (or all if no id given)."""
    import glob as _glob

    suffix = f"_{unique_id}" if unique_id else ""
    pattern = os.path.join(_IPC_DIR, f"pymllm_*{suffix}")
    for f in _glob.glob(pattern):
        try:
            os.unlink(f)
        except OSError:
            pass


def setup_subprocess_logging(log_level: str = "info") -> None:
    """Configure logging for a spawned subprocess.

    When Python spawns a subprocess (``mp.set_start_method('spawn')``), the
    child starts with a blank logging configuration.  Call this function at the
    very beginning of every subprocess entry point so that log records are
    emitted at the correct level.

    Parameters
    ----------
    log_level
        Case-insensitive level name, e.g. ``"debug"``, ``"info"``, ``"warning"``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("pymllm").setLevel(level)
