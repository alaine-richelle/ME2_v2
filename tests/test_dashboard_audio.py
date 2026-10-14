import sys
from types import SimpleNamespace

import numpy as np

from me2.dashboard_audio import DashboardAudioPlayer


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
