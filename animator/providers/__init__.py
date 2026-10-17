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

from .base import LLM, Narrator


def env_key(*names: str) -> str:
    for name in names:
        if value := os.environ.get(name):
            return value
    alternatives = "".join(f" (or {name})" for name in names[1:])
    raise ValueError(f"needs {names[0]}{alternatives} in the environment")


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


LLMS: dict[str, Callable[[argparse.Namespace], LLM]] = {
    "gemini": gemini_llm,
    "openrouter": openrouter_llm,
}

NARRATORS: dict[str, Callable[[argparse.Namespace], Narrator]] = {
    "gemini": gemini_narrator,
}
