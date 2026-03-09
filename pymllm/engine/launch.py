import asyncio
import atexit
import logging
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig
from huggingface_hub import snapshot_download

try:
    from pyfiglet import figlet_format
    from termcolor import colored

    HAS_BANNER_LIBS = True
except ImportError:
    HAS_BANNER_LIBS = False

from pymllm.configs import get_global_config
from pymllm.engine.io_struct import GenerateReqInput
from pymllm.orchestrator.ipc_utils import make_ipc_address
from pymllm.orchestrator.request_response_process import (
    ReqState,
    RequestResponseProcess,
)
from pymllm.orchestrator.tokenizer_process import run_tokenizer_process
from pymllm.orchestrator.scheduler_process import run_scheduler_process
from pymllm.orchestrator.detokenizer_process import run_detokenizer_process

logger = logging.getLogger(__name__)

# Standard HuggingFace config fields that indicate max output tokens,
# checked in priority order.
_MAX_NEW_TOKENS_FIELDS = (
    "max_new_tokens",
    "max_tokens",
    "max_completion_tokens",
)


def _normalize_eos_raw(raw) -> List[int]:
    """Normalize a raw eos_token_id value (int, list, or None) to a list."""
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [x for x in raw if isinstance(x, int)]
    return []


def _get_eos_token_ids(hf_config, model_path=None) -> List[int]:
    """Extract EOS token ID(s) from a HuggingFace model config.

    Searches in priority order:
    1. ``hf_config.eos_token_id`` (top-level, standard models)
    2. ``hf_config.text_config.eos_token_id`` (VL / multimodal models)
    3. ``generation_config.json`` (many models store EOS here)
    4. ``tokenizer_config.json`` via AutoTokenizer (last resort)
    """
    if hf_config is None:
        return []

    # 1. Top-level config
    ids = _normalize_eos_raw(getattr(hf_config, "eos_token_id", None))
    if ids:
        return ids

    # 2. Nested text_config (VL / multimodal models like Qwen3-VL)
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        ids = _normalize_eos_raw(getattr(text_config, "eos_token_id", None))
        if ids:
            return ids

    # 3. generation_config.json (lightweight, just reads a JSON file)
    if model_path is not None:
        try:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig.from_pretrained(str(model_path))
            ids = _normalize_eos_raw(getattr(gen_cfg, "eos_token_id", None))
            if ids:
                logger.info("EOS token IDs from generation_config.json: %s", ids)
                return ids
        except Exception:
            pass

    # 4. Tokenizer (last resort)
    if model_path is not None:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            if tok.eos_token_id is not None:
                ids = [tok.eos_token_id]
                logger.info("EOS token ID from tokenizer: %s", ids)
                return ids
        except Exception:
            pass

    return []


def _get_model_default_max_new_tokens(hf_config) -> Optional[int]:
    """Extract max output token limit from a HuggingFace model config.

    Checks standard fields in priority order.  Returns ``None`` when the
    config does not specify any recognised output-length field.
    """
    if hf_config is None:
        return None
    for field_name in _MAX_NEW_TOKENS_FIELDS:
        value = getattr(hf_config, field_name, None)
        if value is not None and isinstance(value, int) and value > 0:
            logger.info(
                "Using model config %s=%d as default max_new_tokens",
                field_name,
                value,
            )
            return value
    return None


class Engine:
    def __init__(self):
        self._subprocesses: List[mp.Process] = []
        self._rr_process: Optional[RequestResponseProcess] = None
        self._config_logging()
        self._set_default_torch_dtype()
        self._check_model_and_tokenizer()

    def launch(self) -> None:
        self._launch_processes()
        atexit.register(self.shutdown)

    def _launch_processes(self) -> None:
        """Spawn all subprocess workers and wire up ZMQ IPC channels."""
        mp.set_start_method("spawn", force=True)
        uid = str(os.getpid())

        # IPC addresses for ZMQ communication between processes
        addr_request_response_to_tokenizer: str = make_ipc_address(
            "request_response_to_tokenizer", uid
        )
        addr_tokenizer_to_scheduler: str = make_ipc_address(
            "tokenizer_to_scheduler", uid
        )
        addr_scheduler_to_detokenizer: str = make_ipc_address(
            "scheduler_to_detokenizer", uid
        )
        addr_detokenizer_to_request_response: str = make_ipc_address(
            "detokenizer_to_request_response", uid
        )
        # Record all subprocesses
        procs_and_readers: List[tuple] = []

        # Config dict for the tokenizer subprocess (must be picklable).
        cfg = get_global_config()
        enable_shared_queue = cfg.server.enable_shared_queue
        transport_mode: str = (
            cfg.server.tensor_transport_mode
        )  # "default" | "cuda_ipc" | "cuda_ipc_pool"

        # Create shared queue if enabled.
        # Note: the MmItemMemoryPool (for "cuda_ipc_pool") is created *inside*
        # the tokenizer subprocess after CUDA is initialised.  The queue here
        # is constructed without a pool; TokenizerProcess._ensure_pool() will
        # swap in a pool-aware queue at runtime.
        shared_queue = None
        if enable_shared_queue:
            from pymllm.orchestrator.shared_memory_queue import TensorQueue as _TQ

            # Construct with the configured transport mode.  The pool is not
            # supplied here; it will be lazily initialised inside the subprocess.
            shared_queue = _TQ(
                maxsize=1000,
                transport_mode=transport_mode,
                pool=None,  # pool initialised lazily inside TokenizerProcess
            )
            logger.info(
                "Shared memory queue enabled for fast IPC (transport_mode=%s)",
                transport_mode,
            )

        tokenizer_cfg: Dict[str, Any] = {
            "tokenizer_path": str(cfg.server.tokenizer_path),
            "tokenizer_mode": cfg.server.tokenizer_mode,
            "trust_remote_code": cfg.server.trust_remote_code,
            "context_length": cfg.server.context_length,
            "hf_config": cfg.model.hf_config,
            "enable_shared_queue": enable_shared_queue,
            "tensor_transport_mode": transport_mode,
            "cuda_ipc_pool_size_mb": cfg.server.cuda_ipc_pool_size_mb,
            "cuda_ipc_recycle_interval": cfg.server.cuda_ipc_recycle_interval,
            "log_level": cfg.server.log_level,
        }

        # Tokenizer
        tokenizer_reader, tokenizer_writer = mp.Pipe(duplex=False)
        tokenizer_proc = mp.Process(
            target=run_tokenizer_process,
            args=(
                addr_request_response_to_tokenizer,
                addr_tokenizer_to_scheduler,
                tokenizer_writer,
                tokenizer_cfg,
                shared_queue,  # Pass shared queue
            ),
            daemon=True,
        )
        procs_and_readers.append((tokenizer_proc, tokenizer_reader, "tokenizer"))

        # Determine default max_new_tokens from model config (if available)
        model_max_new_tokens = _get_model_default_max_new_tokens(
            cfg.model.hf_config
        )
        scheduler_kwargs = {}
        if model_max_new_tokens is not None:
            scheduler_kwargs["default_max_new_tokens"] = model_max_new_tokens

        # Extract EOS token ID(s) from model config
        eos_token_ids = _get_eos_token_ids(cfg.model.hf_config, model_path=cfg.server.model_path)
        if eos_token_ids:
            scheduler_kwargs["eos_token_ids"] = eos_token_ids
            logger.info("EOS token IDs for scheduler: %s", eos_token_ids)

        # Model runner config — passed to the scheduler process which now
        # owns the model runner in-process (sglang-style architecture).
        scheduler_kwargs["server_config"] = cfg.server
        scheduler_kwargs["model_config"] = cfg.model
        scheduler_kwargs["gpu_id"] = cfg.server.base_gpu_id

        # Scheduler (+ in-process model runner)
        scheduler_reader, scheduler_writer = mp.Pipe(duplex=False)
        scheduler_proc = mp.Process(
            target=run_scheduler_process,
            args=(
                addr_tokenizer_to_scheduler,
                addr_scheduler_to_detokenizer,
                scheduler_writer,
                shared_queue,  # Pass shared queue
                enable_shared_queue,  # Pass flag
                transport_mode,  # Pass tensor transport mode
                cfg.server.log_level,  # Pass log level
            ),
            kwargs=scheduler_kwargs,
            daemon=True,
        )
        procs_and_readers.append((scheduler_proc, scheduler_reader, "scheduler"))

        # Detokenizer
        detokenizer_reader, detokenizer_writer = mp.Pipe(duplex=False)
        detokenizer_proc = mp.Process(
            target=run_detokenizer_process,
            args=(
                addr_scheduler_to_detokenizer,
                addr_detokenizer_to_request_response,
                detokenizer_writer,
                tokenizer_cfg,
            ),
            daemon=True,
        )
        procs_and_readers.append((detokenizer_proc, detokenizer_reader, "detokenizer"))

        # Start all subprocesses
        for proc, _, name in procs_and_readers:
            proc.start()
            self._subprocesses.append(proc)
            logger.info("Started %s process (pid=%s)", name, proc.pid)

        # Wait for readiness signals
        for _, reader, name in procs_and_readers:
            try:
                msg = reader.recv()
            except EOFError:
                raise RuntimeError(f"{name} process died before signalling readiness")
            if msg.get("status") != "ready":
                raise RuntimeError(f"{name} process failed to initialise: {msg}")
            logger.info("%s process ready", name)

        # RR Process is current main process — only bind ZMQ sockets here.
        # Background tasks are started lazily by listen() on the first
        # add_request(), so they always run on the correct event loop.
        self._rr_process = RequestResponseProcess(
            send_to_tokenizer_addr=addr_request_response_to_tokenizer,
            recv_from_detokenizer_addr=addr_detokenizer_to_request_response,
        )
        self._rr_process.start()
        logger.info("RequestResponseProcess sockets bound")

        # Print colorful gradient ASCII art banner
        if HAS_BANNER_LIBS:
            try:
                text = figlet_format("pymllm", font="slant")
                fired_up = figlet_format("FIRED UP!", font="slant")

                # Apply blue-purple gradient
                lines = text.strip().split("\n")
                colors_cycle = ["blue", "cyan", "blue", "magenta", "magenta"]
                for i, line in enumerate(lines):
                    color = colors_cycle[i % len(colors_cycle)]
                    print(colored(line, color, attrs=["bold"]))

                # Print "FIRED UP!" in bright magenta
                for line in fired_up.strip().split("\n"):
                    print(colored(line, "magenta", attrs=["bold"]))
                print()
            except Exception as e:
                logger.debug(f"Failed to print banner: {e}")
                print("🚀 pymllm FIRED UP! 🚀\n")
        else:
            print("🚀 pymllm FIRED UP! 🚀\n")

    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Any] = None,
        audio_data: Optional[Any] = None,
        video_data: Optional[Any] = None,
        return_logprob: Optional[Union[List[bool], bool]] = None,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[Union[List[Optional[str]], str]] = None,
        session_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        stream: bool = False,
        rid: Optional[Union[List[str], str]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Synchronous, non-streaming generation entry point.

        Accepts a single prompt (``str``) or a batch (``List[str]``).  Returns a
        single result dict for single inputs and a list of result dicts for batch
        inputs, preserving the input order.
        """
        rid = self._make_rids(rid, prompt, input_ids)
        request = GenerateReqInput(
            rid=rid,
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            stream=stream,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            lora_path=lora_path,
            session_params=session_params,
            extra_options=kwargs,
        )
        request.normalize_batch_and_arguments()

        async def _run() -> Union[Dict[str, Any], List[Dict[str, Any]]]:
            result = await self._rr_process.add_request(request)
            if request.is_single:
                single_rid = rid if isinstance(rid, str) else rid[0]
                return await self._wait_for_final_result(single_rid, result)  # type: ignore[arg-type]
            # Batch: wait for every sub-request concurrently.
            rids_list: List[str] = rid if isinstance(rid, list) else [rid]  # type: ignore[assignment]
            states: List[ReqState] = result  # type: ignore[assignment]
            outputs = await asyncio.gather(
                *(self._wait_for_final_result(r, s) for r, s in zip(rids_list, states))
            )
            return list(outputs)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run())

    async def generate_async(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Any] = None,
        audio_data: Optional[Any] = None,
        video_data: Optional[Any] = None,
        return_logprob: Optional[Union[List[bool], bool]] = None,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[Union[List[Optional[str]], str]] = None,
        session_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        stream: bool = False,
        rid: Optional[Union[List[str], str]] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronous generation entry point.

        For a **single** request and ``stream=False`` yields one final result
        dict; with ``stream=True`` yields incremental chunks.

        For a **batch** request the iterator yields the final result for each
        sub-request as it completes (order not guaranteed); streaming mode yields
        incremental chunks from all sub-requests interleaved.
        """
        rid = self._make_rids(rid, prompt, input_ids)
        request = GenerateReqInput(
            rid=rid,
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            stream=stream,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            lora_path=lora_path,
            session_params=session_params,
            extra_options=kwargs,
        )
        request.normalize_batch_and_arguments()
        result = await self._rr_process.add_request(request)

        if request.is_single:
            single_rid = rid if isinstance(rid, str) else rid[0]  # type: ignore[index]
            state: ReqState = result  # type: ignore[assignment]
            try:
                if stream:
                    async for chunk in self._stream_results(single_rid, state):
                        yield chunk
                else:
                    yield await self._wait_for_final_result(single_rid, state)
            finally:
                if not state.finished:
                    logger.info("Aborting request %s (client disconnected)", single_rid)
                    await self._rr_process.abort_request(single_rid)
                else:
                    self._rr_process.remove_state(single_rid)
        else:
            rids_list: List[str] = rid if isinstance(rid, list) else [rid]  # type: ignore[assignment]
            states: List[ReqState] = result  # type: ignore[assignment]
            _bg_tasks: List[asyncio.Task] = []
            try:
                if stream:
                    # Merge streams from all sub-requests using an asyncio queue.
                    queue: asyncio.Queue = asyncio.Queue()

                    async def _forward(r: str, s: ReqState) -> None:
                        async for chunk in self._stream_results(r, s):
                            await queue.put(chunk)
                        await queue.put(None)  # sentinel

                    _bg_tasks = [
                        asyncio.create_task(_forward(r, s))
                        for r, s in zip(rids_list, states)
                    ]
                    done_count = 0
                    while done_count < len(_bg_tasks):
                        item = await queue.get()
                        if item is None:
                            done_count += 1
                        else:
                            yield item
                    await asyncio.gather(*_bg_tasks)
                else:
                    for coro in asyncio.as_completed(
                        [
                            self._wait_for_final_result(r, s)
                            for r, s in zip(rids_list, states)
                        ]
                    ):
                        yield await coro
            finally:
                for t in _bg_tasks:
                    t.cancel()
                for r, s in zip(rids_list, states):
                    if not s.finished:
                        logger.info("Aborting request %s (client disconnected)", r)
                        await self._rr_process.abort_request(r)
                    else:
                        self._rr_process.remove_state(r)

    @staticmethod
    async def _wait_for_final_result(rid: str, state: ReqState) -> Dict[str, Any]:
        """Block until the request is finished and return the last output."""
        while True:
            await state.event.wait()
            if state.finished:
                return state.out_list[-1]
            state.event.clear()

    @staticmethod
    async def _stream_results(
        rid: str, state: ReqState
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield incremental chunks as they arrive, until finished."""
        while True:
            await state.event.wait()
            for item in state.out_list:
                yield item
            state.out_list.clear()
            if state.finished:
                return
            state.event.clear()

    @staticmethod
    def _make_rids(
        rid: Optional[Union[str, List[str]]],
        prompt: Optional[Union[str, List[str]]],
        input_ids: Optional[Union[List[int], List[List[int]]]],
    ) -> Union[str, List[str]]:
        """Return rids, auto-generating UUIDs when *rid* is ``None``.

        The helper infers whether the call is a batch from *prompt* / *input_ids*
        so callers don't have to handle this case themselves.
        """
        if rid is not None:
            return rid
        # Determine batch size from the text/input_ids argument.
        is_batch = isinstance(prompt, list) or (
            isinstance(input_ids, list)
            and len(input_ids) > 0
            and isinstance(input_ids[0], list)
        )
        if is_batch:
            n = len(prompt) if prompt is not None else len(input_ids)  # type: ignore[arg-type]
            return [uuid.uuid4().hex for _ in range(n)]
        return uuid.uuid4().hex

    def shutdown(self) -> None:
        """Terminate all subprocesses."""
        if self._rr_process is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._rr_process.shutdown())
                else:
                    loop.run_until_complete(self._rr_process.shutdown())
            except Exception:
                pass
        for proc in self._subprocesses:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
        self._subprocesses.clear()
        logger.info("All subprocesses shut down")

    def _set_default_torch_dtype(self):
        """Set the default torch dtype based on the server configuration."""
        dtype = get_global_config().server.dtype
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_available() else "float32"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype for torch default dtype: {dtype!r}")
        torch.set_default_dtype(torch_dtype)

    def _config_logging(self):
        """Configure logging level from server configuration."""
        level_name = get_global_config().server.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            root_logger.setLevel(level)
        logging.getLogger("pymllm").setLevel(level)

    def _check_model_and_tokenizer(self):
        cfg = get_global_config()
        if cfg.server.model_path is None or cfg.server.tokenizer_path is None:
            logger.error("Model path or tokenizer path is not set")
            raise ValueError("Model path or tokenizer path is not set")
        model_path = cfg.server.model_path
        tokenizer_path = cfg.server.tokenizer_path
        download_dir = cfg.server.download_dir
        trust_remote_code = cfg.server.trust_remote_code

        shared_path = model_path == tokenizer_path

        model_path = self._maybe_download(model_path, download_dir)
        cfg.server.model_path = model_path

        if shared_path:
            cfg.server.tokenizer_path = model_path
        else:
            cfg.server.tokenizer_path = self._maybe_download(
                tokenizer_path, download_dir
            )

        cfg.model.hf_config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        logger.info("Loaded model config: %s", cfg.model.hf_config.__class__.__name__)

    @staticmethod
    def _maybe_download(path: Path, download_dir: Optional[Path] = None) -> Path:
        if path.is_dir():
            return path
        repo_id = str(path)
        logger.info("Downloading '%s' ...", repo_id)
        kwargs = {}
        if download_dir is not None:
            kwargs["local_dir"] = str(download_dir / path.name)
        downloaded = snapshot_download(repo_id=repo_id, **kwargs)
        logger.info("Downloaded '%s' to '%s'", repo_id, downloaded)
        return Path(downloaded)
