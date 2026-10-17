import argparse
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from animator.providers import NARRATORS, read_transcript
from animator.providers.audiocpp import AudioCppNarrator

FAKE_CLI = textwrap.dedent('''\
    import argparse, json, os, sys, wave
    from pathlib import Path

    parser = argparse.ArgumentParser()
    for flag in ["--task", "--family", "--model", "--backend", "--language", "--batch-text-file",
                 "--batch-merge-audio", "--out-dir", "--reference-text", "--voice-ref"]:
        parser.add_argument(flag, required=True)
    args = parser.parse_args()
    calls = Path(__file__).with_name("calls.jsonl")
    with calls.open("a") as log:
        log.write(json.dumps({"argv": sys.argv[1:], "text": Path(args.batch_text_file).read_text("utf-8")}) + "\\n")
    print("loading model", args.model)
    if os.environ.get("FAKE_AUDIOCPP_FAIL"):
        print("ggml_cuda_init: out of memory", file=sys.stderr)
        sys.exit(3)
    skip = os.environ.get("FAKE_AUDIOCPP_SKIP")
    lines = Path(args.batch_text_file).read_text("utf-8").split("\\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip() or str(number) == skip:
            continue
        with wave.open(str(Path(args.out_dir) / f"line_{number}.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\\x01\\x00" * (1600 * len(line.split())))  # 0.1 s per word
''')


@pytest.fixture
def fake_cli(tmp_path) -> Path:
    script = tmp_path / "fake_audiocpp.py"
    script.write_text(FAKE_CLI)
    if os.name == "nt":
        cli = tmp_path / "audiocpp_cli.bat"
        cli.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        cli = tmp_path / "audiocpp_cli"
        cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    return cli


def calls(cli: Path) -> list[dict]:
    log = cli.with_name("calls.jsonl")
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def narrator(cli: Path, tmp_path: Path) -> AudioCppNarrator:
    return AudioCppNarrator(cli, tmp_path / "qwen3.gguf", tmp_path / "voice.wav", "Hello there.", backend="vulkan")


def test_one_process_for_every_scene(fake_cli, tmp_path):
    scenes = [["One two.", "Three four five."], ["Six\n  seven   eight nine."]]

    narrations = narrator(fake_cli, tmp_path).narrate(scenes)

    (call,) = calls(fake_cli)
    assert call["text"] == "One two.\nThree four five.\nSix seven eight nine.\n"
    argv = call["argv"]
    assert argv[argv.index("--task") + 1] == "tts"
    assert argv[argv.index("--family") + 1] == "qwen3_tts"
    assert argv[argv.index("--backend") + 1] == "vulkan"
    assert argv[argv.index("--batch-merge-audio") + 1] == "none"
    assert argv[argv.index("--reference-text") + 1] == "Hello there."
    assert argv[argv.index("--voice-ref") + 1] == str(tmp_path / "voice.wav")
    assert [n.windows for n in narrations] == [[[0.0, 0.2], [0.4, 0.7]], [[0.0, 0.4]]]


def test_failure_surfaces_log(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AUDIOCPP_FAIL", "1")
    with pytest.raises(RuntimeError, match="code 3") as caught:
        narrator(fake_cli, tmp_path).narrate([["Hi."]])
    assert "loading model" in str(caught.value)
    assert "out of memory" in str(caught.value)


def test_missing_output_is_an_error(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AUDIOCPP_SKIP", "2")
    with pytest.raises(RuntimeError, match="line_2.wav"):
        narrator(fake_cli, tmp_path).narrate([["Hi.", "There."]])


def test_blank_beat_is_rejected_before_running(fake_cli, tmp_path):
    with pytest.raises(ValueError):
        narrator(fake_cli, tmp_path).narrate([["Hi.", "  "]])
    assert calls(fake_cli) == []


def cli_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    files = {}
    (tmp_path / "flags").mkdir(exist_ok=True)
    for name in ("audiocpp_cli", "tts_model", "voice_ref", "voice_ref_text"):
        files[name] = tmp_path / "flags" / name
        files[name].write_text("x")
    return argparse.Namespace(tts="audiocpp", tts_backend="cuda", **{**files, **overrides})


@pytest.mark.parametrize(("missing", "flag"), [
    ("audiocpp_cli", "--audiocpp-cli"),
    ("tts_model", "--tts-model"),
    ("voice_ref", "--voice-ref"),
    ("voice_ref_text", "--voice-ref-text"),
])
def test_builder_requires_each_flag(tmp_path, missing, flag):
    with pytest.raises(ValueError, match=f"--tts audiocpp needs {flag}$"):
        NARRATORS["audiocpp"](cli_args(tmp_path, **{missing: None}))
    with pytest.raises(ValueError, match=f"{flag} file not found"):
        NARRATORS["audiocpp"](cli_args(tmp_path, **{missing: str(tmp_path / "nope")}))


def test_transcript_collapses_whitespace_and_falls_back_to_cp1252(tmp_path):
    utf8 = tmp_path / "utf8.txt"
    utf8.write_text("﻿Café  says\r\nhello.\n", encoding="utf-8")
    cp1252 = tmp_path / "cp1252.txt"
    cp1252.write_bytes("It’s fine.\n".encode("cp1252"))
    assert read_transcript(utf8) == "Café says hello."
    assert read_transcript(cp1252) == "It’s fine."


def test_cli_renders_with_audiocpp(fake_cli, tmp_path, monkeypatch):
    from animator.cli import main
    from animator.providers import LLMS
    from tests.test_pipeline import FakeLLM, PLAN, probe

    monkeypatch.setitem(LLMS, "fake", lambda args: FakeLLM())
    paths = cli_args(tmp_path)
    output = tmp_path / "out.mp4"

    code = main([
        "--prompt", "the sky", "-o", str(output), "--quality", "low", "--llm", "fake",
        "--tts", "audiocpp", "--audiocpp-cli", str(fake_cli), "--tts-model", str(paths.tts_model),
        "--voice-ref", str(paths.voice_ref), "--voice-ref-text", str(paths.voice_ref_text),
    ])

    assert code == 0
    assert len(calls(fake_cli)) == 1
    words = [len(beat.narration.split()) for scene in PLAN.scenes for beat in scene.beats]
    # 0.1 s per word from the fake CLI, plus one 0.2 s gap between a scene's two beats
    assert abs(probe(output) - (0.1 * sum(words) + 0.2 * len(PLAN.scenes))) < 0.3
