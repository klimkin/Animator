import argparse
import io
import sys
import types
import wave

import numpy as np
import pytest

from animator.providers import NARRATORS
from animator.providers.voxcpm import VoxCPMNarrator

RATE = 16000


class FakeModel:
    loads = []

    def __init__(self, model_id):
        self.model_id = model_id
        self.tts_model = types.SimpleNamespace(sample_rate=RATE)
        self.generated = []

    @classmethod
    def from_pretrained(cls, model_id):
        model = cls(model_id)
        cls.loads.append(model)
        return model

    def generate(self, text, prompt_wav_path=None, prompt_text=None):
        self.generated.append((text, prompt_wav_path, prompt_text))
        return np.full(RATE * len(text.split()) // 10, 1.5, dtype=np.float32)  # 0.1 s per word, clipped


@pytest.fixture
def fake_voxcpm(monkeypatch):
    FakeModel.loads = []
    monkeypatch.setitem(sys.modules, "voxcpm", types.SimpleNamespace(VoxCPM=FakeModel))
    return FakeModel


def test_loads_once_and_joins_each_scene(fake_voxcpm, tmp_path):
    voice = tmp_path / "voice.wav"
    narrator = VoxCPMNarrator("some/model", voice, "Reference words.")

    narrations = narrator.narrate([["One two.", "Three."], ["Four five six."]])

    (model,) = fake_voxcpm.loads
    assert model.model_id == "some/model"
    assert model.generated == [
        ("One two.", str(voice), "Reference words."),
        ("Three.", str(voice), "Reference words."),
        ("Four five six.", str(voice), "Reference words."),
    ]
    assert [n.windows for n in narrations] == [[[0.0, 0.2], [0.4, 0.5]], [[0.0, 0.3]]]
    with wave.open(io.BytesIO(narrations[1].wav)) as wav_file:
        assert (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()) == (1, 2, RATE)
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    assert samples.max() == 32767


def test_without_voice_reference(fake_voxcpm):
    VoxCPMNarrator().narrate([["Hi."]])
    (model,) = fake_voxcpm.loads
    assert model.model_id == "openbmb/VoxCPM2"
    assert model.generated == [("Hi.", None, None)]


def test_reference_needs_both_parts(tmp_path):
    with pytest.raises(ValueError, match="both"):
        VoxCPMNarrator(prompt_wav=tmp_path / "voice.wav")
    transcript = tmp_path / "voice.txt"
    transcript.write_text("hi")
    args = argparse.Namespace(tts="voxcpm", tts_model=None, voice_ref=None, voice_ref_text=str(transcript))
    with pytest.raises(ValueError, match="both"):
        NARRATORS["voxcpm"](args)


def test_missing_package_explains_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "voxcpm", None)
    with pytest.raises(RuntimeError, match="pip install voxcpm"):
        VoxCPMNarrator().narrate([["Hi."]])
