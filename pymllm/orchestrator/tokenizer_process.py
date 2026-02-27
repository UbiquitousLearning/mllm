"""
TokenizerProcess -- subprocess that tokenizes incoming raw requests.

Receives raw requests from RequestResponseProcess via ZMQ, tokenizes them,
and forwards the tokenized payloads to the SchedulerProcess.

Supports two modes:
    1. Legacy ZMQ path: Send TokenizedGenerateReqInput via ZMQ send_pyobj
    2. Shared queue fast path: Write metadata to shared memory and put rid in shared queue
"""

import logging
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Union

import zmq
from transformers import AutoProcessor, AutoTokenizer

from pymllm.engine.io_struct import TokenizedGenerateReqInput
from pymllm.orchestrator.ipc_utils import create_zmq_socket
from pymllm.orchestrator.shared_memory_queue import SharedMemoryManager, TensorQueue

logger = logging.getLogger(__name__)


class TokenizerProcess:
    """Runs inside a subprocess spawned by ``torch.multiprocessing``."""

    def __init__(
        self,
        recv_from_rr_addr: str,
        send_to_scheduler_addr: str,
        tokenizer_cfg: Dict[str, Any],
        shared_queue: Optional[TensorQueue] = None,
    ):
        """
        Parameters
        ----------
        tokenizer_cfg:
            Serialisable dict built by the parent process (``Engine``) before
            spawning.  Required keys:

            * ``tokenizer_path``    – str, path to the tokenizer directory.
            * ``tokenizer_mode``    – ``"auto" | "slow" | "fast"``.
            * ``trust_remote_code`` – bool.
            * ``context_length``    – Optional[int], explicit cap; inferred from
              ``hf_config`` when ``None``.
            * ``hf_config``         – Optional HuggingFace PretrainedConfig
              (pickled by multiprocessing); used only to infer ``context_length``.
            * ``enable_shared_queue`` – bool, whether to use shared memory fast path.
        shared_queue:
            Optional TensorQueue for shared memory fast path communication.
        """
        self._recv_from_rr_addr = recv_from_rr_addr
        self._send_to_scheduler_addr = send_to_scheduler_addr
        self._tokenizer_cfg = tokenizer_cfg
        self._enable_shared_queue = tokenizer_cfg.get("enable_shared_queue", False)
        self._shared_queue = shared_queue

        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_rr: Optional[zmq.Socket] = None
        self._send_to_scheduler: Optional[zmq.Socket] = None

        self._tokenizer = None
        self._mm_processor = None
        self._context_length: Optional[int] = None

        self._init_tokenizers()

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
        logger.info(
            "TokenizerProcess event loop started (shared_queue=%s)",
            self._enable_shared_queue,
        )
        while True:
            raw_request: Dict[str, Any] = self._recv_from_rr.recv_pyobj()
            tokenized = self._tokenize(raw_request)

            if self._enable_shared_queue and self._shared_queue is not None:
                # Shared queue fast path
                self._send_via_shared_queue(tokenized)
            else:
                # Legacy ZMQ path
                self._send_to_scheduler.send_pyobj(tokenized)

    def _send_via_shared_queue(
        self, tokenized: Union[TokenizedGenerateReqInput, Dict[str, Any]]
    ) -> None:
        """Send tokenized request via shared memory + shared queue fast path.

        Args:
            tokenized: Either TokenizedGenerateReqInput dataclass or abort dict
        """
        # Handle abort sentinel
        if isinstance(tokenized, dict) and tokenized.get("abort"):
            # Fallback to ZMQ for abort messages
            self._send_to_scheduler.send_pyobj(tokenized)
            return

        assert isinstance(tokenized, TokenizedGenerateReqInput), (
            f"Expected TokenizedGenerateReqInput, got {type(tokenized)}"
        )

        rid = tokenized.rid
        mm_inputs = tokenized.mm_inputs

        # Create a lightweight metadata object (without mm_inputs)
        metadata = TokenizedGenerateReqInput(
            rid=tokenized.rid,
            input_text=tokenized.input_text,
            input_ids=tokenized.input_ids,
            mm_inputs=None,  # Will be passed separately via shared queue
            sampling_params=tokenized.sampling_params,
            stream=tokenized.stream,
            return_logprob=tokenized.return_logprob,
            logprob_start_len=tokenized.logprob_start_len,
            top_logprobs_num=tokenized.top_logprobs_num,
            lora_path=tokenized.lora_path,
            session_params=tokenized.session_params,
        )

        # Write metadata to shared memory
        shm_name = SharedMemoryManager.write_metadata(rid, metadata)

        # Put (rid, shm_name, mm_inputs) into shared queue
        self._shared_queue.put(rid, shm_name, mm_inputs)

        logger.debug(f"Sent request {rid} via shared queue (shm={shm_name})")

    # ------------------------------------------------------------------
    # Tokenization and multimodal preprocessing
    # ------------------------------------------------------------------

    def _init_tokenizers(self) -> None:
        """Initialise text tokenizer and (optionally) multimodal processor.

        All configuration is read from ``self._tokenizer_cfg`` which was
        serialised by the parent process before ``spawn``.  No global config
        access happens inside the subprocess.
        """
        cfg = self._tokenizer_cfg
        tokenizer_path: str = cfg["tokenizer_path"]
        tokenizer_mode: str = cfg.get("tokenizer_mode", "auto")
        trust_remote_code: bool = bool(cfg.get("trust_remote_code", False))

        tokenizer_kwargs: Dict[str, Any] = {
            "use_fast": tokenizer_mode != "slow",
            "trust_remote_code": trust_remote_code,
        }

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )

        # Default to left padding for generation.
        try:
            self._tokenizer.padding_side = "left"
        except Exception:
            pass

        # Context length: explicit config value takes priority; fall back to
        # common HF config field names.
        context_len: Optional[int] = cfg.get("context_length")
        if context_len is None:
            hf_cfg = cfg.get("hf_config")
            for name in ("max_position_embeddings", "max_sequence_length", "seq_len"):
                if hf_cfg is not None and hasattr(hf_cfg, name):
                    context_len = int(getattr(hf_cfg, name))
                    break
        self._context_length = context_len

        # Try to load multimodal processor (optional).
        try:
            self._mm_processor = AutoProcessor.from_pretrained(
                tokenizer_path,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            # Text-only models don't provide a processor; that's fine.
            self._mm_processor = None

    def _tokenize(
        self, raw_request: Dict[str, Any]
    ) -> Union[TokenizedGenerateReqInput, Dict[str, Any]]:
        """Tokenize one raw request dict and return a typed object.

        * **Abort** messages (``{"rid": ..., "abort": True}``) are returned as
          plain dicts so the scheduler can intercept them without importing the
          io_struct.
        * Normal requests are returned as a :class:`TokenizedGenerateReqInput`
          dataclass instance that carries ``input_ids``, ``mm_inputs``, and all
          sampling meta-data in typed fields.

        Each message arriving here corresponds to exactly one sub-request
        because batch splitting happens upstream in ``RequestResponseProcess``.
        """
        # Abort: propagate as a plain sentinel dict.
        if raw_request.get("abort"):
            return {"rid": raw_request.get("rid"), "abort": True}

        # ------------------------------------------------------------------ #
        # 1. Text tokenization
        # ------------------------------------------------------------------ #
        if raw_request.get("input_ids") is not None:
            # Caller already tokenized – skip text processing.
            input_ids: List[int] = list(raw_request["input_ids"])
            raw_text = raw_request.get("text")
            input_text: str = (
                str(raw_text[0]) if isinstance(raw_text, list) else str(raw_text or "")
            )
        else:
            text = raw_request.get("text")
            if text is None:
                raise ValueError(
                    "TokenizerProcess expects either `text` or `input_ids`."
                )
            # Accept a list for robustness; take the first element.
            input_text = str(text[0]) if isinstance(text, list) else str(text)

            encode_kwargs: Dict[str, Any] = {
                "add_special_tokens": True,
                "return_attention_mask": False,
            }
            if self._context_length is not None:
                encode_kwargs.update(
                    {"truncation": True, "max_length": self._context_length}
                )

            encoding = self._tokenizer(input_text, **encode_kwargs)
            input_ids = encoding["input_ids"]

        # ------------------------------------------------------------------ #
        # 2. Multimodal pre-processing
        # ------------------------------------------------------------------ #
        mm_inputs = self._collect_mm_inputs(raw_request, text=input_text)

        # ------------------------------------------------------------------ #
        # 3. Pack into the typed dataclass
        # ------------------------------------------------------------------ #
        return TokenizedGenerateReqInput(
            rid=raw_request.get("rid"),
            input_text=input_text,
            input_ids=input_ids,
            mm_inputs=mm_inputs,
            sampling_params=raw_request.get("sampling_params") or {},
            stream=bool(raw_request.get("stream", False)),
            return_logprob=bool(raw_request.get("return_logprob", False)),
            logprob_start_len=int(raw_request.get("logprob_start_len", -1)),
            top_logprobs_num=int(raw_request.get("top_logprobs_num", 0)),
            lora_path=raw_request.get("lora_path"),
            session_params=raw_request.get("session_params"),
        )

    def _normalize_image_input(self, image_data: Any) -> List[Any]:
        """Normalise ``image_data`` into a list of image-like objects.

        Supported input forms:
        - single PIL.Image / numpy array / torch.Tensor
        - path string or bytes
        - list/tuple of the above
        """

        def _to_image(obj: Any) -> Any:
            # Lazily import Pillow to avoid hard dependency for text-only models.
            try:
                from PIL import Image  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Pillow is required for image preprocessing in TokenizerProcess"
                ) from exc

            if obj is None:
                return None
            if isinstance(obj, Image.Image):
                return obj
            if isinstance(obj, (str, bytes)):
                return Image.open(obj)
            return obj

        if isinstance(image_data, (list, tuple)):
            return [
                img for img in (_to_image(x) for x in image_data) if img is not None
            ]
        return [img for img in (_to_image(image_data),) if img is not None]

    def _collect_mm_inputs(
        self, raw_request: Dict[str, Any], text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Pre-process multimodal data and return a consolidated ``mm_inputs`` dict.

        Returns ``None`` for text-only requests.  Otherwise returns a flat dict
        whose keys are ready to be unpacked by the model runner:

        * ``image_inputs``  – output of ``AutoProcessor`` (contains
          ``pixel_values``, etc.) when a processor is available.
        * ``image_data``    – raw image objects when no processor is available.
        * ``audio_data``    – forwarded verbatim (no processor yet).
        * ``video_data``    – forwarded verbatim (no processor yet).
        """
        image_data = raw_request.get("image_data")
        video_data = raw_request.get("video_data")
        audio_data = raw_request.get("audio_data")

        if not any(x is not None for x in (image_data, video_data, audio_data)):
            return None  # text-only request

        mm: Dict[str, Any] = {}

        # Image: prefer AutoProcessor output; fall back to raw data.
        if image_data is not None:
            if self._mm_processor is not None:
                images = self._normalize_image_input(image_data)
                try:
                    processor_inputs = self._mm_processor(
                        images=images,
                        text=text if text is not None else raw_request.get("text"),
                        return_tensors="pt",
                    )
                    mm["image_inputs"] = processor_inputs
                except Exception:
                    mm["image_data"] = image_data
            else:
                mm["image_data"] = image_data

        # Audio / video forwarded verbatim for now.
        if audio_data is not None:
            mm["audio_data"] = audio_data
        if video_data is not None:
            mm["video_data"] = video_data

        return mm

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
    tokenizer_cfg: Dict[str, Any],
    shared_queue: Optional[TensorQueue] = None,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    proc = TokenizerProcess(
        recv_from_rr_addr, send_to_scheduler_addr, tokenizer_cfg, shared_queue
    )
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
