"""OpenRouter backend: text, image and vision through one chat-completions
endpoint, so any model OpenRouter serves can run the storyboard."""
from __future__ import annotations

import base64
import binascii
import logging
import time
from pathlib import Path

import httpx
from PIL import Image
from pydantic import ValidationError

from .base import ModelNames, T

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULTS = ModelNames(
    text="google/gemini-3.8-flash",
    region="google/gemini-3.1-flash-lite",
    image="google/gemini-3.1-flash-image",
)

logger = logging.getLogger(__name__)


def data_url(path: Path) -> str:
    with Image.open(path) as image:
        mime_type = Image.MIME.get(image.format, "image/png")
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def decode_data_url(url: str) -> bytes:
    header, sep, payload = url.partition(",")
    if not sep or not header.startswith("data:") or not header.endswith(";base64"):
        raise RuntimeError(f"image is not a base64 data URL: {url[:80]}")
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise RuntimeError("image data URL has invalid base64") from exc


class OpenRouterLLM:
    defaults = DEFAULTS

    def __init__(
        self,
        api_key: str,
        *,
        http: httpx.Client | None = None,
        max_retries: int = 5,
        retry_delay: float = 8.0,
    ):
        self.api_key = api_key
        self.http = http or httpx.Client(timeout=300)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def structured(self, model: str, prompt: str, schema: type[T], image: Path | None = None) -> T:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": data_url(image)}})
        messages: list[dict] = [{"role": "user", "content": content}]
        text = self.complete(self.structured_request(model, messages, schema)).get("content") or ""
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            logger.warning("%s reply did not match %s; asking it to correct", model, schema.__name__)
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That reply does not match the JSON schema:\n{exc}\n"
                                            "Reply again with only the corrected JSON."},
            ]
        text = self.complete(self.structured_request(model, messages, schema)).get("content") or ""
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise RuntimeError(f"{model} returned JSON that does not match {schema.__name__}: {exc}") from exc

    @staticmethod
    def structured_request(model: str, messages: list[dict], schema: type[T]) -> dict:
        return {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
        }

    def image(self, model: str, prompt: str) -> bytes:
        reply = self.complete({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": "16:9"},
        })
        images = reply.get("images") or []
        if not images:
            raise RuntimeError(f"{model} returned no image. text: {(reply.get('content') or '')[:500]}")
        return decode_data_url(images[0]["image_url"]["url"])

    def complete(self, payload: dict) -> dict:
        """POST one chat completion and return its first message, backing off
        on rate limits, server errors and dropped connections."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.http.post(
                    API_URL, json=payload, headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return self.first_message(response)
                failure = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code != 429 and response.status_code < 500:
                    raise RuntimeError(self.redact(f"OpenRouter request failed, {failure}"))
            if attempt == self.max_retries:
                raise RuntimeError(self.redact(f"OpenRouter request failed after {attempt} attempts, {failure}"))
            logger.warning("OpenRouter call failed (%s); retry %s/%s", self.redact(failure), attempt, self.max_retries)
            time.sleep(self.retry_delay * attempt)

    def first_message(self, response: httpx.Response) -> dict:
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(self.redact(f"OpenRouter returned no choices: {response.text[:500]}"))
        return choices[0]["message"]

    def redact(self, text: str) -> str:
        return text.replace(self.api_key, "[redacted]")
