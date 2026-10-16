from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "laptop_output_check.py"
    spec = importlib.util.spec_from_file_location("laptop_output_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_check_uses_only_output_api(monkeypatch):
    module = _module()
    calls = []
    fake = SimpleNamespace(
        check_output_settings=lambda **kwargs: calls.append(("check", kwargs)),
        play=lambda samples, **kwargs: calls.append(("play", samples.size, kwargs)),
        wait=lambda: calls.append(("wait",)),
        stop=lambda: calls.append(("stop",)),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    module.play_output_check(seconds=0.1, output_device="Earphones", sample_rate=1_000)

    assert [call[0] for call in calls] == ["check", "play", "wait", "stop"]
    assert calls[1][1] == 100


def test_output_check_fails_closed_when_no_output_is_available(monkeypatch):
    module = _module()
    stopped = []
    fake = SimpleNamespace(
        check_output_settings=lambda **kwargs: (_ for _ in ()).throw(OSError("no output")),
        play=lambda *args, **kwargs: pytest.fail("play must not run"),
        wait=lambda: pytest.fail("wait must not run"),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    with pytest.raises(RuntimeError, match="output_device_unavailable"):
        module.play_output_check(seconds=0.1, sample_rate=1_000)
    assert stopped == [True]


def test_output_cli_requires_explicit_speaker_approval():
    module = _module()

    with pytest.raises(SystemExit):
        module.main(["--seconds", "3"])
