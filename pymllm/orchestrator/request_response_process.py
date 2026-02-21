"""
RequestResponseProcess -- the main-process entry point for user requests.

This process is **not** a subprocess; it lives in the engine's main process.
Incoming requests are placed into an ``asyncio.Queue`` and forwarded to the
TokenizerProcess via ZMQ.  Decoded results arrive back from the
DetokenizerProcess and are dispatched to the waiting callers.

The request-tracking model uses ``ReqState`` pattern: each request
gets an ``asyncio.Event`` + output list so that streaming (multiple incremental
chunks) and one-shot responses are both supported.
"""

import asyncio
import dataclasses
import logging
from typing import Any, Dict, List, Optional

import zmq
import zmq.asyncio

from pymllm.engine.io_struct import GenerateReqInput
from pymllm.orchestrator.ipc_utils import create_zmq_socket, close_zmq_socket

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Per-request state that supports both streaming and one-shot responses.

    ``ReqState`` (Event + out_list).

    The recv loop appends results to *out_list* and signals *event*;
    callers ``await event.wait()`` in a loop, consuming results until
    *finished* is ``True``.
    """

    out_list: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    finished: bool = False
    event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)


class RequestResponseProcess:
    """Sits in the main process; bridges user-facing API and subprocess pipeline."""

    def __init__(
        self,
        send_to_tokenizer_addr: str,
        recv_from_detokenizer_addr: str,
    ):
        self._send_to_tokenizer_addr: str = send_to_tokenizer_addr
        self._recv_from_detokenizer_addr: str = recv_from_detokenizer_addr

        # asyncio queue that buffers incoming user requests
        self._request_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # rid -> ReqState (replaces the old rid -> Future dict)
        self._rid_to_state: Dict[str, ReqState] = {}

        # ZMQ (async context, sockets created lazily in the event loop)
        self._zmq_ctx: Optional[zmq.asyncio.Context] = None
        self._send_to_tokenizer: Optional[zmq.asyncio.Socket] = None
        self._recv_from_detokenizer: Optional[zmq.asyncio.Socket] = None

        self._loop_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Kick off the background send/recv tasks on *loop*."""
        self._zmq_ctx = zmq.asyncio.Context()
        self._send_to_tokenizer = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_tokenizer_addr,
            bind=True,
        )
        self._recv_from_detokenizer = create_zmq_socket(
            self._zmq_ctx,
            zmq.PULL,
            self._recv_from_detokenizer_addr,
            bind=True,
        )
        self._loop_task = loop.create_task(self._run())

    async def add_request(self, request: GenerateReqInput) -> ReqState:
        """Enqueue a request and return its :class:`ReqState`.

        Callers should ``await state.event.wait()`` in a loop, consuming
        ``state.out_list`` entries until ``state.finished`` is ``True``.
        """
        if not isinstance(request.rid, str):
            raise ValueError("RequestResponseProcess currently accepts single requests only.")
        rid = request.rid
        state = ReqState()
        self._rid_to_state[rid] = state
        await self._request_queue.put(request.to_request_dict())
        return state

    def remove_state(self, rid: str) -> None:
        """Remove the ``ReqState`` for *rid* (called by the caller once done)."""
        self._rid_to_state.pop(rid, None)

    async def abort_request(self, rid: str) -> None:
        """Cancel a pending request and notify downstream processes."""
        state = self._rid_to_state.pop(rid, None)
        if state is not None and not state.finished:
            state.finished = True
            state.out_list.append({"rid": rid, "error": "aborted", "finished": True})
            state.event.set()
        await self._send_to_tokenizer.send_pyobj({"rid": rid, "abort": True})

    async def shutdown(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
        if self._send_to_tokenizer is not None:
            close_zmq_socket(self._send_to_tokenizer)
        if self._recv_from_detokenizer is not None:
            close_zmq_socket(self._recv_from_detokenizer)
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main loop: forward requests to tokenizer, receive results from detokenizer."""
        send_task = asyncio.create_task(self._send_loop())
        recv_task = asyncio.create_task(self._recv_loop())
        await asyncio.gather(send_task, recv_task)

    async def _send_loop(self) -> None:
        """Drain the asyncio queue and push requests to the TokenizerProcess."""
        while True:
            request = await self._request_queue.get()
            await self._send_to_tokenizer.send_pyobj(request)

    async def _recv_loop(self) -> None:
        """Receive decoded results from DetokenizerProcess and dispatch to ReqStates."""
        while True:
            result = await self._recv_from_detokenizer.recv_pyobj()
            rid = result.get("rid")
            state = self._rid_to_state.get(rid)
            if state is None:
                logger.warning("Received result for unknown rid=%s", rid)
                continue
            state.out_list.append(result)
            if result.get("finished", False):
                state.finished = True
            state.event.set()
