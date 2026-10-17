"""WAV helpers for narrators that speak one sentence per clip."""
from __future__ import annotations

import io
import wave

from ..providers.base import Narration


def pcm_to_wav(pcm: bytes, *, channels: int, sample_width: int, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)
    return buf.getvalue()


def join_clips(clips: list[bytes], gap: float = 0.2) -> Narration:
    """Concatenate per-sentence PCM WAVs with `gap` seconds of silence between
    them. Each window comes from the clips' real sample counts. Compressed or
    float WAVs fail to open, since the wave module reads integer PCM only."""
    if not clips:
        raise ValueError("no clips to join")
    layout: tuple[int, int, int] | None = None
    frames: list[bytes] = []
    for number, clip in enumerate(clips, start=1):
        try:
            with wave.open(io.BytesIO(clip)) as wav_file:
                params = wav_file.getparams()
                frames.append(wav_file.readframes(params.nframes))
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"clip {number} is not a PCM WAV: {exc}") from exc
        if params.nframes == 0:
            raise ValueError(f"clip {number} is empty")
        clip_layout = (params.nchannels, params.sampwidth, params.framerate)
        if layout is None:
            layout = clip_layout
        elif clip_layout != layout:
            raise ValueError(f"clip {number} has (channels, width, rate) {clip_layout}, clip 1 has {layout}")

    channels, width, rate = layout
    frame_size = channels * width
    # 8-bit WAV is unsigned, so its silence is 0x80
    silence = (b"\x80" if width == 1 else b"\x00") * (round(gap * rate) * frame_size)
    windows = []
    position = 0
    for clip_frames in frames:
        start = position
        position += len(clip_frames) // frame_size
        windows.append([round(start / rate, 3), round(position / rate, 3)])
        position += len(silence) // frame_size
    pcm = silence.join(frames)
    return Narration(wav=pcm_to_wav(pcm, channels=channels, sample_width=width, rate=rate), windows=windows)
