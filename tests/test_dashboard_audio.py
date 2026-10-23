import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from me2.dashboard_audio import DashboardAudioPlayer, WakeChimePlayer, feedback_tone


def test_generated_sample_and_ringtone_are_finite_and_bounded():
    player = DashboardAudioPlayer(sample_rate=8_000)
    assert player._waveform.dtype == np.float32
    assert player._waveform.ndim == 1
    assert np.isfinite(player._waveform).all()
    assert np.max(np.abs(player._waveform)) <= 0.12
    ringtone = player._render_ringtone()
    assert len(ringtone) > 0
    assert np.isfinite(ringtone).all()
    assert np.max(np.abs(ringtone)) <= 0.12


def test_player_callback_silences_paused_stream_and_applies_volume():
    player = DashboardAudioPlayer(sample_rate=8_000)
    output = np.ones((256, 1), dtype=np.float32)
    player._callback(output, 256, None, None)
    assert np.count_nonzero(output) == 0

    with player._lock:
        player._paused = False
        player._volume = 0.25
    player._callback(output, 256, None, None)
    assert np.any(output)
    assert np.max(np.abs(output)) <= 0.03


def test_track_change_preserves_finite_sample():
    player = DashboardAudioPlayer(sample_rate=8_000)
    initial = player._waveform.copy()
    player.next_track()
    assert player._waveform.shape == initial.shape
    assert not np.array_equal(player._waveform, initial)
    assert player._cursor == 0


def test_audio_stream_is_lazy_and_closed_on_stop(monkeypatch):
    streams = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False
            streams.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    player = DashboardAudioPlayer(sample_rate=8_000)
    assert streams == []
    player.play()
    assert len(streams) == 1 and streams[0].started
    assert streams[0].kwargs["channels"] == 1
    player.pause()
    player.stop()
    assert streams[0].stopped and streams[0].closed


def test_wake_chime_uses_generated_pcm_with_bounded_alsa_process(monkeypatch):
    calls, completed = [], []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("me2.dashboard_audio.shutil.which", lambda name: None)
    player = WakeChimePlayer(
        enabled=True,
        alsa_device="plughw:CARD=USB,DEV=0",
        duration_ms=120,
        frequency_hz=880,
        volume=0.12,
        on_complete=lambda status, error_type: completed.append((status, error_type)),
    )
    player._play()
    command, kwargs = calls[0]
    assert command[:5] == ["aplay", "-q", "-D", "plughw:CARD=USB,DEV=0", "-t"]
    assert command[-8:] == ["raw", "-f", "S16_LE", "-r", "48000", "-c", "1", "-"]
    assert isinstance(kwargs["input"], bytes) and kwargs["input"]
    assert kwargs["timeout"] <= 2.0
    assert kwargs["check"] is True
    assert completed == [("played", None)]


def test_wake_chime_failure_is_reported_without_raising(monkeypatch):
    completed = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["aplay"])
        ),
    )
    player = WakeChimePlayer(
        enabled=True, alsa_device="default", duration_ms=20, frequency_hz=200,
        volume=0.01, on_complete=lambda status, error_type: completed.append((status, error_type)),
    )
    player._play()
    assert completed == [("failed", "CalledProcessError")]


def test_feedback_tones_are_distinct_and_error_is_safe():
    assert feedback_tone("wake_accepted") == 880
    assert feedback_tone("command_accepted") == 1046
    assert feedback_tone("rejection_or_error") == 440
    with pytest.raises(ValueError):
        feedback_tone("unknown")


def test_requested_feedback_frequency_is_rendered_per_tone():
    player = WakeChimePlayer(
        enabled=True, alsa_device="default", duration_ms=40, frequency_hz=880, volume=.1,
    )
    assert player._render_pcm(880) != player._render_pcm(440)
