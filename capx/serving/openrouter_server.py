import itertools
import json
import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "openrouter/google/gemini-2.5-pro-preview"
    messages: list[Message]
    temperature: float | None = 0.2
    max_tokens: int | None = 256
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


def _load_api_keys(key_file: str) -> list[str]:
    """Load API keys from a file, one key per line. Ignores blank lines and comments."""
    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {key_file}")
    keys = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    if not keys:
        raise ValueError(f"No API keys found in {key_file}")
    return keys


def create_app(api_key: str, base_url: str, async_client: bool = True) -> FastAPI:
    default_headers = {
        "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
        "X-Title": "CaP-X",
    }

    if async_client:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
    else:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    app = FastAPI(title="OpenRouter Proxy", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if async_client:

        @app.post("/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                client_kwargs = request.model_dump(exclude_none=True)

                # Strip the "openrouter/" prefix if present so OpenRouter sees the
                # native model identifier (e.g. "google/gemini-2.5-pro-preview").
                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/"):]

                if request.stream:
                    client_kwargs["stream"] = True
                    response = await client.chat.completions.create(**client_kwargs)

                    async def event_stream():
                        async for chunk in response:
                            data = chunk.model_dump_json()
                            yield f"data: {data}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(event_stream(), media_type="text/event-stream")

                client_kwargs["stream"] = False
                response = await client.chat.completions.create(**client_kwargs)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    else:

        @app.post("/chat/completions", response_model=ChatCompletionResponse)
        def chat_completions(request: ChatCompletionRequest):
            try:
                client_kwargs = request.model_dump(exclude_none=True)

                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/"):]

                client_kwargs["stream"] = False

                response = client.chat.completions.create(**client_kwargs)

                choices = [
                    ChatCompletionResponseChoice(
                        index=c.index,
                        message=Message(role=c.message.role, content=c.message.content),
                        finish_reason=c.finish_reason,
                    )
                    for c in response.choices
                ]

                return ChatCompletionResponse(
                    id=response.id, created=response.created, model=response.model, choices=choices
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main(
    key_file: str = ".openrouterkey",
    api_key: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8111,
    base_url: str = "https://openrouter.ai/api/v1/",
    async_client: bool = True,
):
    """
    Start the OpenRouter Proxy Server.

    Reads an API key from --api-key or from the key file (one key per line).
    """
    if api_key is None:
        keys = _load_api_keys(key_file)
        api_key = keys[0]
        logger.info(f"Loaded API key from {key_file}")

    app = create_app(api_key=api_key, base_url=base_url, async_client=async_client)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
