"""pymllm HTTP server -- RESTful API entry point.

This module implements a FastAPI-based HTTP server that wraps the pymllm
:class:`Engine` and exposes OpenAI-compatible and native REST endpoints.

Endpoints
---------
* ``GET  /health``            -- liveness probe
* ``GET  /v1/models``         -- list served models (OpenAI-compatible)
* ``POST /generate``          -- native generate (streaming via SSE)
* ``POST /v1/completions``    -- OpenAI-compatible completions
* ``POST /v1/chat/completions`` -- OpenAI-compatible chat completions
* ``GET  /model_info``        -- model metadata
* ``GET  /server_info``       -- runtime config dump
* ``POST /flush_cache``       -- flush internal caches
* ``POST /abort_request``     -- cancel a running request
"""

import asyncio
import contextlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import orjson
import uvicorn
import uvloop
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from pymllm.configs.global_config import get_global_config, make_args, read_args
from pymllm.engine.launch import Engine

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# ---------------------------------------------------------------------------
# Disconnect-aware async generator wrapper
# ---------------------------------------------------------------------------

_DISCONNECT_CHECK_INTERVAL = 1.0  # seconds


async def _iter_with_disconnect_check(
    agen: AsyncIterator,
    request: Request,
    interval: float = _DISCONNECT_CHECK_INTERVAL,
) -> AsyncIterator:
    """Wrap an async generator, periodically checking for client disconnect.

    The standard ``async for chunk in agen`` pattern only checks between
    items.  If the generator blocks waiting for the next item (e.g. waiting
    for a decode step), a client disconnect goes unnoticed.

    This wrapper uses ``asyncio.wait`` with a timeout so that
    ``request.is_disconnected()`` is polled every *interval* seconds even
    while waiting for the next item.

    When a disconnect is detected, the underlying generator is closed via
    ``aclose()`` which triggers its ``finally`` cleanup (abort logic).
    """
    aiter = agen.__aiter__()
    while True:
        # Start fetching the next item without blocking indefinitely.
        next_task = asyncio.ensure_future(aiter.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({next_task}, timeout=interval)
                if done:
                    break
                # Timeout: check if client is still connected.
                if await request.is_disconnected():
                    next_task.cancel()
                    with contextlib.suppress(
                        asyncio.CancelledError, StopAsyncIteration
                    ):
                        await next_task
                    # Close the generator to trigger its finally block.
                    await agen.aclose()
                    return
        except Exception:
            next_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await next_task
            raise

        try:
            yield next_task.result()
        except StopAsyncIteration:
            return

# ---------------------------------------------------------------------------
# Global handles (populated at startup)
# ---------------------------------------------------------------------------
_engine: Optional[Engine] = None
_tokenizer: Optional[Any] = None


def _get_engine() -> Engine:
    """Return the running engine or raise."""
    if _engine is None:
        raise RuntimeError("Engine not initialised")
    return _engine


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Body for ``POST /generate``."""

    text: Optional[Union[List[str], str]] = None
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    image_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    video_data: Optional[Any] = None
    return_logprob: Optional[Union[List[bool], bool]] = None
    logprob_start_len: Optional[Union[List[int], int]] = None
    top_logprobs_num: Optional[Union[List[int], int]] = None
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    session_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    stream: bool = False
    rid: Optional[Union[List[str], str]] = None

    model_config = {"extra": "allow"}  # forward unknown keys as extra_options


# -- OpenAI-compatible models -----------------------------------------------


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False
    continuous_usage_stats: Optional[bool] = False


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """OpenAI ``POST /v1/chat/completions`` body."""

    model: str = ""
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    # Tool calling
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Reasoning control
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    # Pass-through to tokenizer.apply_chat_template (e.g. enable_thinking)
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    """OpenAI ``POST /v1/completions`` body."""

    model: str = ""
    prompt: Union[str, List[str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    echo: bool = False
    logprobs: Optional[int] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


class AbortRequest(BaseModel):
    rid: Optional[str] = None


# ---------------------------------------------------------------------------
# FastAPI application & lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks for the FastAPI app."""
    global _engine, _tokenizer
    _engine = app.state.engine  # type: ignore[attr-defined]

    # Load tokenizer in server process for apply_chat_template
    cfg = get_global_config()
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.server.tokenizer_path),
            trust_remote_code=cfg.server.trust_remote_code,
        )
        logger.info(
            "Loaded tokenizer for chat template: %s", cfg.server.tokenizer_path
        )
    except Exception as e:
        logger.warning("Failed to load tokenizer for chat template: %s", e)

    logger.info(
        "HTTP server ready at http://%s:%s",
        cfg.server.host,
        cfg.server.port,
    )
    yield
    # Shutdown
    if _engine is not None:
        _engine.shutdown()
        _engine = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return ORJSONResponse(
        content={"error": {"message": exc.detail, "code": exc.status_code}},
        status_code=exc.status_code,
    )


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
@app.get("/health_generate")
async def health():
    """Liveness probe."""
    return Response(status_code=200)


@app.get("/model_info")
async def model_info():
    """Return basic model metadata."""
    cfg = get_global_config()
    hf_cfg = cfg.model.hf_config
    return {
        "model_path": str(cfg.server.model_path),
        "tokenizer_path": str(cfg.server.tokenizer_path),
        "served_model_name": cfg.server.served_model_name,
        "model_type": getattr(hf_cfg, "model_type", None) if hf_cfg else None,
        "architectures": getattr(hf_cfg, "architectures", None) if hf_cfg else None,
    }


@app.get("/server_info")
async def server_info():
    """Dump runtime server configuration."""
    import dataclasses as _dc

    cfg = get_global_config()
    return _dc.asdict(cfg.server)


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    cfg = get_global_config()
    model_name = cfg.server.served_model_name or str(cfg.server.model_path)
    return {
        "object": "list",
        "data": [_model_card(model_name)],
    }


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str):
    """OpenAI-compatible single model retrieval."""
    cfg = get_global_config()
    model_name = cfg.server.served_model_name or str(cfg.server.model_path)
    if model_id != model_name:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: '{model_name}'",
        )
    return _model_card(model_name)


def _model_card(model_name: str) -> Dict[str, Any]:
    """Build an OpenAI-compatible Model object."""
    return {
        "id": model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "pymllm",
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Map internal finish reasons to OpenAI-standard values.
_FINISH_REASON_MAP = {
    "eos": "stop",
    "stop": "stop",
    "length": "length",
    "abort": "stop",
}


def _normalize_finish_reason(reason: Optional[str]) -> Optional[str]:
    """Convert internal finish reason to OpenAI-compatible value."""
    if reason is None:
        return None
    return _FINISH_REASON_MAP.get(reason, reason)


def _build_sampling_params(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a sampling_params dict from OpenAI-style fields."""
    params: Dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params["top_k"] = top_k
    if max_tokens is not None:
        params["max_new_tokens"] = max_tokens
    if stop is not None:
        params["stop"] = stop if isinstance(stop, list) else [stop]
    if frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        params["repetition_penalty"] = repetition_penalty
    if seed is not None:
        params["seed"] = seed
    params.update(extra)
    return params


def _messages_to_prompt(
    messages: List[ChatMessage],
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Render chat messages into a prompt string via the model's chat template.

    Uses ``tokenizer.apply_chat_template()`` when available (handles Llama,
    Qwen, Mistral, etc. automatically).  Falls back to ChatML format.

    Parameters
    ----------
    chat_template_kwargs
        Extra keyword arguments forwarded to ``apply_chat_template``
        (e.g. ``enable_thinking=True`` for Qwen3).
    """
    # Flatten each message into a plain dict for the tokenizer.
    msg_dicts: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            # Multimodal: extract only text parts for the prompt string.
            text_parts = [p.text for p in content if p.type == "text" and p.text]
            content = "\n".join(text_parts) if text_parts else ""
        elif content is None:
            content = ""
        d: Dict[str, Any] = {"role": msg.role, "content": content}
        if msg.name is not None:
            d["name"] = msg.name
        msg_dicts.append(d)

    tokenizer = _tokenizer
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            extra = dict(chat_template_kwargs) if chat_template_kwargs else {}
            return tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
                **extra,
            )
        except Exception as e:
            logger.warning("apply_chat_template failed, using fallback: %s", e)

    # Fallback: ChatML format (Qwen-style)
    parts: List[str] = []
    for m in msg_dicts:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _extract_image_data(messages: List[ChatMessage]) -> Optional[List[str]]:
    """Extract image URLs / base64 strings from multimodal content parts."""
    images: List[str] = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if part.type == "image_url" and part.image_url is not None:
                images.append(part.image_url.url)
    return images if images else None


def _make_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _make_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Native generate endpoint
# ---------------------------------------------------------------------------


@app.api_route("/generate", methods=["POST", "PUT"])
async def generate(obj: GenerateRequest, request: Request):
    """Native generation endpoint.  Supports SSE streaming."""
    engine = _get_engine()

    # Collect extra fields as extra_options
    known = set(GenerateRequest.model_fields.keys())
    extra_options = {k: v for k, v in obj.model_dump().items() if k not in known}

    kwargs: Dict[str, Any] = {
        "prompt": obj.text,
        "input_ids": obj.input_ids,
        "sampling_params": obj.sampling_params,
        "image_data": obj.image_data,
        "audio_data": obj.audio_data,
        "video_data": obj.video_data,
        "return_logprob": obj.return_logprob,
        "logprob_start_len": obj.logprob_start_len,
        "top_logprobs_num": obj.top_logprobs_num,
        "lora_path": obj.lora_path,
        "session_params": obj.session_params,
        "stream": obj.stream,
        "rid": obj.rid,
        **extra_options,
    }
    # Strip None values so Engine defaults are used
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if obj.stream:

        async def _stream() -> AsyncIterator[bytes]:
            gen = engine.generate_async(**kwargs)
            try:
                async for chunk in _iter_with_disconnect_check(gen, request):
                    # Skip empty intermediate chunks (e.g. special tokens
                    # stripped by the detokenizer)
                    if not chunk.get("delta") and not chunk.get("finished"):
                        continue
                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
            except Exception as e:
                err = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(err) + b"\n\n"
            finally:
                await gen.aclose()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    gen = engine.generate_async(**kwargs)
    try:
        results = []
        async for item in _iter_with_disconnect_check(gen, request):
            results.append(item)
        result = results[0] if len(results) == 1 else results
        return ORJSONResponse(result)
    except Exception as e:
        logger.error("[generate] Error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        await gen.aclose()


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/completions
# ---------------------------------------------------------------------------


@app.post("/v1/completions")
async def openai_completions(obj: CompletionRequest, request: Request):
    """OpenAI-compatible text completion endpoint."""
    engine = _get_engine()
    sp = _build_sampling_params(
        temperature=obj.temperature,
        top_p=obj.top_p,
        top_k=obj.top_k,
        max_tokens=obj.max_tokens,
        stop=obj.stop,
        frequency_penalty=obj.frequency_penalty,
        presence_penalty=obj.presence_penalty,
        repetition_penalty=obj.repetition_penalty,
        seed=obj.seed,
    )
    cfg = get_global_config()
    model_name = obj.model or cfg.server.served_model_name or str(cfg.server.model_path)
    include_usage = (
        obj.stream_options is not None and obj.stream_options.include_usage
    )

    if obj.stream:

        async def _stream() -> AsyncIterator[bytes]:
            comp_id = _make_completion_id()
            prompt_tokens = 0
            completion_tokens = 0
            gen = engine.generate_async(
                prompt=obj.prompt, sampling_params=sp, stream=True
            )
            try:
                async for chunk in _iter_with_disconnect_check(gen, request):
                    prompt_tokens = chunk.get("prompt_tokens", prompt_tokens)
                    completion_tokens = chunk.get("completion_tokens", completion_tokens)
                    delta_text = chunk.get("delta", "")
                    finish_reason = _normalize_finish_reason(
                        chunk.get("finished_reason")
                    )
                    # Skip empty intermediate chunks
                    if not delta_text and finish_reason is None:
                        continue
                    sse: Dict[str, Any] = {
                        "id": comp_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": delta_text,
                                "logprobs": None,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield b"data: " + orjson.dumps(sse) + b"\n\n"
            except Exception as e:
                err = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(err) + b"\n\n"
            finally:
                await gen.aclose()
            # Final usage-only chunk (OpenAI stream_options.include_usage)
            if include_usage:
                usage_chunk: Dict[str, Any] = {
                    "id": comp_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield b"data: " + orjson.dumps(usage_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    gen = engine.generate_async(
        prompt=obj.prompt, sampling_params=sp
    )
    try:
        results = []
        async for item in _iter_with_disconnect_check(gen, request):
            results.append(item)
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for i, r in enumerate(results):
            choices.append(
                {
                    "index": i,
                    "text": r.get("text", ""),
                    "logprobs": None,
                    "finish_reason": _normalize_finish_reason(
                        r.get("finished_reason", "stop")
                    ),
                }
            )
            prompt_tokens += r.get("prompt_tokens", 0)
            completion_tokens += r.get("completion_tokens", 0)

        return ORJSONResponse(
            {
                "id": _make_completion_id(),
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
    except Exception as e:
        logger.error("[v1/completions] Error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        await gen.aclose()


# ---------------------------------------------------------------------------
# OpenAI-compatible /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def openai_chat_completions(obj: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completion endpoint with reasoning & tool-call parsing."""
    engine = _get_engine()
    cfg = get_global_config()
    # Auto-enable thinking when reasoning_parser is configured and the
    # client didn't explicitly set enable_thinking.
    chat_kwargs = dict(obj.chat_template_kwargs) if obj.chat_template_kwargs else {}
    if cfg.server.reasoning_parser and "enable_thinking" not in chat_kwargs:
        chat_kwargs["enable_thinking"] = True
    prompt = _messages_to_prompt(obj.messages, chat_template_kwargs=chat_kwargs or None)
    image_data = _extract_image_data(obj.messages)

    # max_completion_tokens takes precedence over max_tokens (OpenAI convention)
    max_tokens = obj.max_completion_tokens if obj.max_completion_tokens is not None else obj.max_tokens

    sp = _build_sampling_params(
        temperature=obj.temperature,
        top_p=obj.top_p,
        top_k=obj.top_k,
        max_tokens=max_tokens,
        stop=obj.stop,
        frequency_penalty=obj.frequency_penalty,
        presence_penalty=obj.presence_penalty,
        repetition_penalty=obj.repetition_penalty,
        seed=obj.seed,
    )
    cfg = get_global_config()
    model_name = obj.model or cfg.server.served_model_name or str(cfg.server.model_path)
    include_usage = (
        obj.stream_options is not None and obj.stream_options.include_usage
    )

    # Resolve parsers from server config
    reasoning_type = cfg.server.reasoning_parser
    tool_call_type = cfg.server.tool_call_parser

    gen_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "sampling_params": sp,
    }
    if image_data is not None:
        gen_kwargs["image_data"] = image_data

    if obj.stream:

        async def _stream() -> AsyncIterator[bytes]:
            from pymllm.parsers import ReasoningParser, ToolCallParser

            comp_id = _make_chat_completion_id()
            created = int(time.time())
            first = True
            prompt_tokens = 0
            completion_tokens = 0
            has_tool_calls = False  # track across entire stream

            # Instantiate streaming parsers
            r_parser = (
                ReasoningParser(reasoning_type, stream_reasoning=obj.stream_reasoning)
                if reasoning_type and obj.separate_reasoning
                else None
            )
            tc_parser = (
                ToolCallParser(tool_call_type, tools=obj.tools)
                if tool_call_type and obj.tools
                else None
            )

            def _make_sse(delta: Dict[str, Any], finish: Optional[str] = None) -> bytes:
                sse: Dict[str, Any] = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "logprobs": None,
                            "finish_reason": finish,
                        }
                    ],
                }
                return b"data: " + orjson.dumps(sse) + b"\n\n"

            gen = engine.generate_async(**gen_kwargs, stream=True)
            try:
                async for chunk in _iter_with_disconnect_check(gen, request):
                    prompt_tokens = chunk.get("prompt_tokens", prompt_tokens)
                    completion_tokens = chunk.get("completion_tokens", completion_tokens)

                    raw_delta = chunk.get("delta", "")
                    finish_reason = _normalize_finish_reason(
                        chunk.get("finished_reason")
                    )

                    # --- Phase 1: reasoning parser ---
                    reasoning_delta = ""
                    content_delta = raw_delta
                    if r_parser and raw_delta:
                        reasoning_delta, content_delta = r_parser.parse_stream_chunk(
                            raw_delta
                        )

                    # --- Phase 2: tool-call parser ---
                    tool_items: list = []
                    if tc_parser and content_delta:
                        content_delta, tool_items = tc_parser.parse_stream_chunk(
                            content_delta
                        )

                    # --- Emit chunks ---
                    # Role chunk (first)
                    if first:
                        yield _make_sse({"role": "assistant"})
                        first = False

                    # Reasoning content
                    if reasoning_delta:
                        yield _make_sse({"reasoning_content": reasoning_delta})

                    # Tool call deltas
                    if tool_items:
                        has_tool_calls = True
                    for tc in tool_items:
                        yield _make_sse({"tool_calls": [tc.to_openai_dict()]})

                    # Normal content
                    if content_delta:
                        yield _make_sse({"content": content_delta})

                    # Finish
                    if finish_reason is not None:
                        # Flush remaining tool call data
                        if tc_parser:
                            remaining = tc_parser.flush()
                            for tc in remaining:
                                has_tool_calls = True
                                yield _make_sse({"tool_calls": [tc.to_openai_dict()]})
                            if has_tool_calls:
                                finish_reason = "tool_calls"
                        yield _make_sse({}, finish=finish_reason)

            except Exception as e:
                err = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(err) + b"\n\n"
            finally:
                await gen.aclose()
            # Final usage-only chunk
            if include_usage:
                usage_chunk: Dict[str, Any] = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield b"data: " + orjson.dumps(usage_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # -- Non-streaming --
    gen = engine.generate_async(**gen_kwargs)
    try:
        from pymllm.parsers import ReasoningParser, ToolCallParser

        r = {}
        async for item in _iter_with_disconnect_check(gen, request):
            r = item
        prompt_tokens = r.get("prompt_tokens", 0)
        completion_tokens = r.get("completion_tokens", 0)
        text = r.get("text", "")
        finish_reason = _normalize_finish_reason(r.get("finished_reason", "stop"))

        # Parse reasoning
        reasoning_content = None
        if reasoning_type and obj.separate_reasoning:
            rp = ReasoningParser(reasoning_type)
            reasoning_content, text = rp.parse_non_stream(text)

        # Parse tool calls
        tool_calls_list = None
        if tool_call_type and obj.tools:
            tp = ToolCallParser(tool_call_type, tools=obj.tools)
            if tp.has_tool_call(text):
                text, parsed_calls = tp.parse_non_stream(text)
                if parsed_calls:
                    tool_calls_list = [tc.to_openai_dict(streaming=False) for tc in parsed_calls]
                    finish_reason = "tool_calls"

        message: Dict[str, Any] = {"role": "assistant", "content": text or None}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls_list:
            message["tool_calls"] = tool_calls_list

        return ORJSONResponse(
            {
                "id": _make_chat_completion_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
    except Exception as e:
        logger.error("[v1/chat/completions] Error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        await gen.aclose()


# ---------------------------------------------------------------------------
# Administrative endpoints
# ---------------------------------------------------------------------------


@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """Placeholder cache flush."""
    return Response(content="Cache flushed.\n", status_code=200)


@app.post("/abort_request")
async def abort_request(obj: AbortRequest):
    """Abort a running request by rid."""
    engine = _get_engine()
    if obj.rid and engine._rr_process is not None:
        await engine._rr_process.abort_request(obj.rid)
        return Response(status_code=200)
    raise HTTPException(status_code=400, detail="Missing or invalid rid")


# ---------------------------------------------------------------------------
# Prepare args helper
# ---------------------------------------------------------------------------


def _prepare_args():
    """Parse CLI arguments into the global config singleton."""
    parser = make_args()
    read_args(parser=parser)


# ---------------------------------------------------------------------------
# Server launcher
# ---------------------------------------------------------------------------


def launch_server():
    """Launch the pymllm Engine then start the uvicorn HTTP server.

    It first boots all engine subprocesses (tokenizer, scheduler, model-runner, detokenizer) 
    and then hands off to uvicorn to serve HTTP traffic.
    """
    _prepare_args()
    cfg = get_global_config()

    engine = Engine()
    engine.launch()

    # Attach engine to app.state so the lifespan hook can pick it up.
    app.state.engine = engine  # type: ignore[attr-defined]

    logger.info(
        "Starting HTTP server on %s:%s (root_path=%r)",
        cfg.server.host,
        cfg.server.port,
        cfg.server.fastapi_root_path,
    )

    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        root_path=cfg.server.fastapi_root_path,
        log_level=cfg.server.log_level,
        timeout_keep_alive=5,
        loop="uvloop",
    )


def main():
    """CLI entry point."""
    launch_server()


if __name__ == "__main__":
    main()
