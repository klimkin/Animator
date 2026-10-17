"""audio.cpp narrator: a local Qwen3-TTS GGUF model, voice-cloned from a
reference clip, run as one CLI batch for the whole video."""
from __future__ import annotations

import subprocess
import tempfile
from itertools import islice
from pathlib import Path

from ..storyboard.narration import join_clips
from .base import Narration

LOG_TAIL_CHARS = 2000


class AudioCppNarrator:
    def __init__(
        self,
        cli: Path,
        model: Path,
        voice_ref: Path,
        reference_text: str,
        backend: str = "cuda",
        language: str = "english",
    ):
        self.cli = cli
        self.model = model
        self.voice_ref = voice_ref
        self.reference_text = reference_text
        self.backend = backend
        self.language = language

    def narrate(self, scenes: list[list[str]]) -> list[Narration]:
        # one sentence per line: audio.cpp writes line N of the batch file to line_N.wav
        sentences = [" ".join(sentence.split()) for beats in scenes for sentence in beats]
        if not all(sentences):
            raise ValueError("every beat needs narration text")
        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = Path(temp_dir) / "sentences.txt"
            text_file.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            out_dir = Path(temp_dir) / "out"
            out_dir.mkdir()
            log = self.run(text_file, out_dir)
            clips = []
            for line in range(1, len(sentences) + 1):
                wav = out_dir / f"line_{line}.wav"
                if not wav.is_file():
                    raise RuntimeError(f"audio.cpp wrote no {wav.name}:\n{log[-LOG_TAIL_CHARS:]}")
                clips.append(wav.read_bytes())

        remaining = iter(clips)
        return [join_clips(list(islice(remaining, len(beats)))) for beats in scenes]

    def run(self, text_file: Path, out_dir: Path) -> str:
        """Run the batch and return its combined stdout and stderr."""
        command = [
            str(self.cli),
            "--task", "tts",
            "--family", "qwen3_tts",
            "--model", str(self.model),
            "--backend", self.backend,
            "--language", self.language,
            "--batch-text-file", str(text_file),
            "--batch-merge-audio", "none",
            "--out-dir", str(out_dir),
            "--reference-text", self.reference_text,
            "--voice-ref", str(self.voice_ref),
        ]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"audio.cpp exited with code {result.returncode}:\n{result.stdout[-LOG_TAIL_CHARS:]}")
        return result.stdout
