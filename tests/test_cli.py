import subprocess
import sys

import pytest

from animator.cli import main


def test_help_runs():
    result = subprocess.run([sys.executable, "-m", "animator", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--llm" in result.stdout


def test_api_app_imports():
    subprocess.run([sys.executable, "-c", "import animator.api.app"], check=True)


@pytest.mark.parametrize(("flags", "env_var"), [
    ([], "GEMINI_API_KEY"),
    (["--llm", "openrouter"], "OPENROUTER_API_KEY"),
])
def test_missing_key_fails_before_any_call(monkeypatch, tmp_path, capsys, flags, env_var):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("animator.cli.load_env_file", lambda: None)

    assert main(["--prompt", "idea", "-o", str(tmp_path / "x.mp4"), *flags]) == 1
    assert env_var in capsys.readouterr().err
