"""VoxCPM narrator: a local Python TTS model, optionally voice-cloned from a
reference clip and its transcript."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..storyboard.narration import join_clips, pcm_to_wav
from .base import Narration

DEFAULT_MODEL = "openbmb/VoxCPM2"


class VoxCPMNarrator:
    def __init__(self, model_id: str = DEFAULT_MODEL, prompt_wav: Path | None = None, prompt_text: str | None = None):
        if (prompt_wav is None) != (prompt_text is None):
            raise ValueError("VoxCPM voice cloning needs both --voice-ref and --voice-ref-text")
        self.model_id = model_id
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text

    def narrate(self, scenes: list[list[str]]) -> list[Narration]:
        try:
            from voxcpm import VoxCPM
        except ImportError as exc:
            raise RuntimeError("--tts voxcpm needs the voxcpm package: pip install voxcpm") from exc
        model = VoxCPM.from_pretrained(self.model_id)
        rate = model.tts_model.sample_rate
        prompt_wav = str(self.prompt_wav) if self.prompt_wav else None

        def speak(sentence: str) -> bytes:
            audio = model.generate(text=sentence.strip(), prompt_wav_path=prompt_wav, prompt_text=self.prompt_text)
            pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
            return pcm_to_wav(pcm.tobytes(), channels=1, sample_width=2, rate=rate)

        return [join_clips([speak(sentence) for sentence in beats]) for beats in scenes]
