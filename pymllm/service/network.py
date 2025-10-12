# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import json
import uuid
import typing
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionChunk

from .models_hub import MODEL_HUB_LOOKUP_TABLE
from .rr_process import (
    start_service,
    stop_service,
    send_request,
    get_response,
)

MODEL_SESSION_CREATED = set()

_SENTINEL = object()


def _get_next_chunk(blocking_generator):
    try:
        return next(blocking_generator)
    except StopIteration:
        return _SENTINEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts the background service on application startup and stops it on shutdown."""
    print("Starting background services...")
    start_service(1)
    yield
    print("Stopping background services...")
    stop_service()


app = FastAPI(lifespan=lifespan)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = True
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    enable_thinking: bool = Field(default=True)


@app.post("/v1/chat/completions")
async def predict(req: ChatCompletionRequest) -> StreamingResponse:
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    """
    model_name = req.model

    if model_name not in MODEL_HUB_LOOKUP_TABLE:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    send_request(req.model_dump_json())

    async def correctly_handled_stream() -> typing.AsyncIterator[str]:
        loop = asyncio.get_running_loop()

        try:
            blocking_generator = get_response(req.id)
            while True:
                raw_json = await loop.run_in_executor(
                    None, _get_next_chunk, blocking_generator
                )
                if raw_json is _SENTINEL:
                    print("Stream finished.")
                    break
                chunk = ChatCompletionChunk.model_validate(json.loads(raw_json))
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            # Handle any errors that might occur during the process.
            print(f"An error occurred during streaming: {e}")
            error_payload = {
                "error": {
                    "message": "An internal error occurred during streaming.",
                    "type": "server_error",
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

        # Send the SSE (Server-Sent Events) termination signal.
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        correctly_handled_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
