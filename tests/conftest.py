import io
import wave

import numpy as np
import pytest


def tone_wav(seconds: float, *, rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """A 440 Hz tone as PCM WAV."""
    frames = round(seconds * rate)
    signal = 0.5 * np.sin(2 * np.pi * 440 * np.arange(frames) / rate)
    if sample_width == 1:
        samples = (signal * 127 + 128).astype(np.uint8)
    else:
        samples = (signal * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(np.repeat(samples, channels).tobytes())
    return buf.getvalue()


@pytest.fixture
def tone():
    return tone_wav
