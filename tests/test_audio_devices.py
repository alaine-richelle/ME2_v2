import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from me2.app import _capture
from me2.audio_devices import (
    AudioDeviceError,
    list_alsa_devices,
    resolve_alsa_device,
    select_capture_input,
    validate_alsa_device,
)


def _listing(*lines):
    return subprocess.CompletedProcess(["arecord", "-l"], 0, stdout="\n".join(lines), stderr="")


def test_alsa_auto_selection_uses_stable_card_identifier(monkeypatch):
    import me2.audio_devices as devices

    monkeypatch.setattr(devices.shutil, "which", lambda _: "/usr/bin/arecord")
    runner = lambda *args, **kwargs: _listing(
        "card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]"
    )
    candidates = list_alsa_devices("capture", runner=runner)
    assert candidates[0].stable_pcm == "plughw:CARD=Device,DEV=0"
    assert resolve_alsa_device("capture", "auto", runner=runner) == (
        "plughw:CARD=Device,DEV=0", "named_alsa"
    )


def test_alsa_name_selector_survives_card_renumbering(monkeypatch):
    import me2.audio_devices as devices

    monkeypatch.setattr(devices.shutil, "which", lambda _: "/usr/bin/arecord")
    runner = lambda *args, **kwargs: _listing(
        "card 7: USBMic [C-Media USB Audio], device 0: USB Audio [USB Audio]"
    )
    assert resolve_alsa_device("capture", "name:c-media", runner=runner)[0] == (
        "plughw:CARD=USBMic,DEV=0"
    )


def test_alsa_auto_refuses_ambiguous_identical_capture_devices(monkeypatch):
    import me2.audio_devices as devices

    monkeypatch.setattr(devices.shutil, "which", lambda _: "/usr/bin/arecord")
    runner = lambda *args, **kwargs: _listing(
        "card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]",
        "card 2: Device_1 [USB Audio Device], device 0: USB Audio [USB Audio]",
    )
    with pytest.raises(AudioDeviceError, match="ambiguous_capture_device"):
        resolve_alsa_device("capture", "auto", runner=runner)


def test_auto_prefers_shared_sounddevice_default(monkeypatch):
    calls = []
    fake = SimpleNamespace(check_input_settings=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    selected = select_capture_input(
        "auto", input_device=None, alsa_device="auto", sample_rate=16_000
    )
    assert selected.backend == "sounddevice"
    assert selected.selection_source == "shared_default"
    assert calls == [{"device": None, "channels": 1, "dtype": "float32", "samplerate": 16_000}]


def test_direct_alsa_validation_reports_busy_without_stderr_detail(monkeypatch):
    import me2.audio_devices as devices

    monkeypatch.setattr(devices.shutil, "which", lambda _: "/usr/bin/arecord")
    runner = lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, stdout="", stderr="arecord: audio open error: Device or resource busy"
    )
    with pytest.raises(AudioDeviceError, match="capture_device_busy"):
        validate_alsa_device("capture", "plughw:CARD=Device,DEV=0", runner=runner)


def test_direct_alsa_capture_uses_named_pcm_and_trims_integer_duration(monkeypatch):
    from me2 import app

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=np.zeros(16_000, dtype="<i2").tobytes())

    monkeypatch.setattr(app.subprocess, "run", run)
    samples = _capture(
        0.5, 16_000, audio_backend="arecord", alsa_device="plughw:CARD=Device,DEV=0"
    )
    assert samples.shape == (8_000,)
    assert calls[0][0][2:4] == ["-D", "plughw:CARD=Device,DEV=0"]
    assert calls[0][0][calls[0][0].index("-d") + 1] == "1"


def test_health_auto_falls_back_to_named_alsa_without_card_number(monkeypatch):
    from me2 import health

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(
            query_devices=lambda: (_ for _ in ()).throw(RuntimeError("shared unavailable"))
        ),
    )
    monkeypatch.setattr(
        health, "resolve_alsa_device", lambda *_args, **_kwargs: ("plughw:CARD=USBMic,DEV=0", "named_alsa")
    )
    monkeypatch.setattr(
        health.shutil, "which", lambda _: "/usr/bin/arecord"
    )
    monkeypatch.setattr(
        health.subprocess, "run",
        lambda *args, **kwargs: _listing(
            "card 6: USBMic [USB Microphone], device 0: USB Audio [USB Audio]"
        ),
    )
    report = health.inspect_input_devices(None, 16_000, audio_backend="auto", alsa_device="auto")
    assert report["alsa_device"] == "plughw:CARD=USBMic,DEV=0"
    assert report["selection_source"] == "named_alsa"
