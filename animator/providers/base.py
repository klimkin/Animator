"""What the storyboard pipeline needs from a backend: an LLM for text, images
and vision, and a Narrator for speech."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ModelNames:
    text: str
    region: str
    image: str


@dataclass(frozen=True)
class Narration:
    wav: bytes
    """The whole scene take as PCM WAV."""
    windows: list[list[float]]
    """[start, end] seconds of each beat inside `wav`."""


class LLM(Protocol):
    """Text, image and vision calls one storyboard needs. Blocking; the
    pipeline runs them with asyncio.to_thread."""

    defaults: ModelNames

    def structured(self, model: str, prompt: str, schema: type[T], image: Path | None = None) -> T: ...

    def image(self, model: str, prompt: str) -> bytes:
        """One 16:9 image, as encoded bytes."""
        ...


class Narrator(Protocol):
    def narrate(self, scenes: list[list[str]]) -> list[Narration]:
        """One call per video: every scene's beat sentences in, one Narration
        per scene out, so local engines load their model once."""
        ...
