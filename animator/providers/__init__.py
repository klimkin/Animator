"""Backends for the storyboard pipeline, picked by name from the CLI.

Each builder turns parsed CLI args into a ready backend, reading its API key
from the environment, and raises ValueError before any network call when a
key or a required flag is missing. Backend modules import lazily so --help
and unused backends cost nothing.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from .base import LLM, Narrator


def env_key(*names: str) -> str:
    for name in names:
        if value := os.environ.get(name):
            return value
    alternatives = "".join(f" (or {name})" for name in names[1:])
    raise ValueError(f"needs {names[0]}{alternatives} in the environment")


def required_file(args: argparse.Namespace, attr: str) -> Path:
    flag = "--" + attr.replace("_", "-")
    if not (value := getattr(args, attr)):
        raise ValueError(f"--tts {args.tts} needs {flag}")
    if not Path(value).is_file():
        raise ValueError(f"{flag} file not found: {value}")
    return Path(value)


def read_transcript(path: Path) -> str:
    """A voice reference transcript as one line. UTF-8, with a cp1252 fallback
    for files saved by Windows tools."""
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    if not (text := " ".join(text.split())):
        raise ValueError(f"voice reference transcript is empty: {path}")
    return text


def gemini_key() -> str:
    return env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


def gemini_llm(args: argparse.Namespace) -> LLM:
    from .gemini import GeminiLLM

    return GeminiLLM(gemini_key())


def openrouter_llm(args: argparse.Namespace) -> LLM:
    from .openrouter import OpenRouterLLM

    return OpenRouterLLM(env_key("OPENROUTER_API_KEY"))


def gemini_narrator(args: argparse.Namespace) -> Narrator:
    from .gemini import DEFAULT_TTS_MODEL, GeminiNarrator

    return GeminiNarrator(gemini_key(), args.tts_model or DEFAULT_TTS_MODEL, args.voice)


def audiocpp_narrator(args: argparse.Namespace) -> Narrator:
    from .audiocpp import AudioCppNarrator

    return AudioCppNarrator(
        cli=required_file(args, "audiocpp_cli"),
        model=required_file(args, "tts_model"),
        voice_ref=required_file(args, "voice_ref"),
        reference_text=read_transcript(required_file(args, "voice_ref_text")),
        backend=args.tts_backend,
    )


LLMS: dict[str, Callable[[argparse.Namespace], LLM]] = {
    "gemini": gemini_llm,
    "openrouter": openrouter_llm,
}

NARRATORS: dict[str, Callable[[argparse.Namespace], Narrator]] = {
    "gemini": gemini_narrator,
    "audiocpp": audiocpp_narrator,
}
