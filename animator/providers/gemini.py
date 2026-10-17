"""Gemini backend: text, image and vision through GeminiLLM, speech through
GeminiNarrator. Both retry rate limits and server errors."""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import errors, types
from PIL import Image

from ..storyboard.narration import pcm_to_wav
from .base import ModelNames, Narration, T
from .beat_timing import PCM_RATE, PCM_SAMPLE_WIDTH, locate_beats

DEFAULT_TEXT_MODEL = "gemini-3.8-flash"
DEFAULT_REGION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"

MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
RETRY_DELAY_SEC = float(os.getenv("GEMINI_RETRY_DELAY_SEC", "8"))

logger = logging.getLogger(__name__)


def default_models() -> ModelNames:
    """$GEMINI_MODEL, when set, replaces the text and region defaults."""
    override = os.environ.get("GEMINI_MODEL")
    return ModelNames(
        text=override or DEFAULT_TEXT_MODEL,
        region=override or DEFAULT_REGION_MODEL,
        image=DEFAULT_IMAGE_MODEL,
    )


def is_retryable(exc: Exception) -> bool:
    """Rate limits (429) and server errors (5xx) are worth retrying."""
    return isinstance(exc, errors.APIError) and (exc.code == 429 or (exc.code or 0) >= 500)


def with_retry(call):
    """Run a blocking Gemini call, backing off on retryable errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == MAX_RETRIES or not is_retryable(exc):
                raise
            logger.warning("Gemini call failed (%s); retry %s/%s", exc, attempt, MAX_RETRIES)
            time.sleep(RETRY_DELAY_SEC * attempt)


def first_inline_data(response, what: str) -> bytes:
    """Bytes of the first inline part (image or audio) in a response."""
    if response.candidates:
        for part in response.candidates[0].content.parts or []:
            if part.inline_data is not None:
                return part.inline_data.data
    raise RuntimeError(f"{what} returned no data. raw: {(response.text or '')[:500]}")


class GeminiLLM:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.defaults = default_models()

    def structured(self, model: str, prompt: str, schema: type[T], image: Path | None = None) -> T:
        contents: list = [prompt]
        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
        if image is not None:
            with Image.open(image) as opened:
                mime_type = Image.MIME.get(opened.format, "image/png")
            contents.insert(0, types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_type))
            # vision calls are region detection, which runs without thinking for speed
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        response = with_retry(lambda: self.client.models.generate_content(
            model=model, contents=contents, config=config,
        ))
        parsed = response.parsed if response else None
        if not isinstance(parsed, schema):
            raise RuntimeError(f"{schema.__name__} came back empty. raw: {(response.text or '')[:500]}")
        return parsed

    def image(self, model: str, prompt: str) -> bytes:
        response = with_retry(lambda: self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
            ),
        ))
        return first_inline_data(response, "Image generation")


class GeminiNarrator:
    """Speaks each scene as one take, which keeps the voice natural and stays
    inside TTS rate limits; beat windows are recovered from the pauses."""

    def __init__(self, api_key: str, model: str = DEFAULT_TTS_MODEL, voice: str = DEFAULT_VOICE):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.voice = voice

    def narrate(self, scenes: list[list[str]]) -> list[Narration]:
        with ThreadPoolExecutor(max_workers=len(scenes) or 1) as pool:
            return list(pool.map(self.narrate_scene, scenes))

    def narrate_scene(self, beats: list[str]) -> Narration:
        pcm = self.synthesize_pcm(" ".join(text.strip() for text in beats))
        return Narration(
            wav=pcm_to_wav(pcm, channels=1, sample_width=PCM_SAMPLE_WIDTH, rate=PCM_RATE),
            windows=locate_beats(pcm, beats),
        )

    def synthesize_pcm(self, text: str) -> bytes:
        """Raw speech: mono 16-bit PCM at 24kHz."""
        response = with_retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                    )
                ),
            ),
        ))
        return first_inline_data(response, "Narration synthesis")
