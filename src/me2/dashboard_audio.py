"""Optional, small synthesized sample player for local dashboard demonstrations."""

from __future__ import annotations

import threading

import numpy as np


class DashboardAudioPlayer:
    """Lazy PortAudio output; the stream is opened only after an accepted action."""

    _MELODY = (392.0, 440.0, 523.25, 440.0, 392.0, 329.63, 349.23, 392.0)

    def __init__(self, *, sample_rate: int = 22_050, output_device=None):
        self.sample_rate = sample_rate
        self.output_device = output_device
        self._lock = threading.Lock()
        self._stream = None
        self._cursor = 0
        self._track = 1
        self._volume = 0.5
        self._paused = True
        self._waveform = self._render(self._track)

    def _render(self, track: int) -> np.ndarray:
        rate = self.sample_rate
        note_seconds = 0.28
        pieces = []
        melody = self._MELODY[track % len(self._MELODY):] + self._MELODY[:track % len(self._MELODY)]
        for frequency in melody:
            count = round(note_seconds * rate)
            times = np.arange(count, dtype=np.float32) / rate
            envelope = np.minimum(1.0, times / 0.018) * np.minimum(
                1.0, (note_seconds - times) / 0.035
            )
            note = 0.12 * np.sin(2 * np.pi * frequency * times) * envelope
            pieces.append(note.astype(np.float32))
        return np.concatenate(pieces)

    def _render_ringtone(self) -> np.ndarray:
        rate = self.sample_rate
        pieces = []
        for frequency in (880.0, 660.0, 880.0, 660.0):
            count = round(0.18 * rate)
            times = np.arange(count, dtype=np.float32) / rate
            envelope = np.minimum(1.0, times / 0.008) * np.minimum(
                1.0, (0.18 - times) / 0.02
            )
            pieces.append(
                (0.12 * np.sin(2 * np.pi * frequency * times) * envelope).astype(np.float32)
            )
            pieces.append(np.zeros(round(0.08 * rate), dtype=np.float32))
        return np.concatenate(pieces)

    def _callback(self, output, frames, _time_info, _status):
        with self._lock:
            if self._paused:
                output.fill(0)
                return
            waveform = self._waveform
            positions = (np.arange(frames) + self._cursor) % len(waveform)
            output[:, 0] = waveform[positions] * self._volume
            self._cursor = int((self._cursor + frames) % len(waveform))

    def play(self) -> None:
        import sounddevice as sd

        with self._lock:
            if self._stream is None:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.output_device,
                    callback=self._callback,
                )
                try:
                    self._stream.start()
                except Exception:
                    self._stream.close()
                    self._stream = None
                    raise
            self._paused = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            self._paused = True
            self._cursor = 0
        if stream is not None:
            stream.stop()
            stream.close()

    def next_track(self) -> None:
        track = self._track + 1
        waveform = self._render(track)
        with self._lock:
            self._track = track
            self._waveform = waveform
            self._cursor = 0

    def play_ringtone(self) -> None:
        waveform = self._render_ringtone()
        with self._lock:
            self._waveform = waveform
            self._cursor = 0
        self.play()

    def set_volume(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("volume must be between 0 and 1")
        with self._lock:
            self._volume = float(value)
