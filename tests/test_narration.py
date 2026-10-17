import io
import wave

import pytest

from animator.storyboard.narration import join_clips


def read(wav: bytes) -> tuple[wave._wave_params, bytes]:
    with wave.open(io.BytesIO(wav)) as wav_file:
        return wav_file.getparams(), wav_file.readframes(wav_file.getnframes())


def test_windows_follow_sample_counts_and_gaps(tone):
    narration = join_clips([tone(1.0, rate=16000), tone(0.5, rate=16000), tone(0.25, rate=16000)], gap=0.2)

    assert narration.windows == [[0.0, 1.0], [1.2, 1.7], [1.9, 2.15]]
    params, _ = read(narration.wav)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, 16000)
    assert params.nframes == round(2.15 * 16000)


def test_gap_is_silence(tone):
    _, frames = read(join_clips([tone(0.1), tone(0.1)], gap=0.5).wav)
    clip_bytes = round(0.1 * 24000) * 2
    assert set(frames[clip_bytes:clip_bytes + round(0.5 * 24000) * 2]) == {0}


def test_8bit_silence_is_midpoint(tone):
    _, frames = read(join_clips([tone(0.1, sample_width=1), tone(0.1, sample_width=1)], gap=0.1).wav)
    clip_bytes = round(0.1 * 24000)
    assert set(frames[clip_bytes:clip_bytes + 2400]) == {0x80}


def test_stereo_windows(tone):
    narration = join_clips([tone(0.5, channels=2), tone(0.5, channels=2)], gap=0.0)
    assert narration.windows == [[0.0, 0.5], [0.5, 1.0]]


@pytest.mark.parametrize("other", [
    {"rate": 16000},
    {"channels": 2},
    {"sample_width": 1},
])
def test_rejects_mismatched_formats(tone, other):
    with pytest.raises(ValueError, match="clip 2 has"):
        join_clips([tone(0.1), tone(0.1, **other)])


def test_rejects_empty_and_non_pcm(tone):
    with pytest.raises(ValueError, match="no clips"):
        join_clips([])
    with pytest.raises(ValueError, match="clip 2 is empty"):
        join_clips([tone(0.1), tone(0.0)])
    float_wav = bytearray(tone(0.1))
    float_wav[20:22] = (3).to_bytes(2, "little")  # format tag 3: IEEE float
    with pytest.raises(ValueError, match="clip 1 is not a PCM WAV"):
        join_clips([bytes(float_wav)])
