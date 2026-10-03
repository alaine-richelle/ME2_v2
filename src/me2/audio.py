"""Small, dependency-light audio loading and normalization helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 4.0
    silence_threshold_db: float = -48.0
    normalize_peak: bool = True

    @property
    def sample_count(self) -> int:
        return round(self.sample_rate * self.duration_seconds)


def _mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono or multichannel audio, got shape {samples.shape}")
    return np.asarray(samples, dtype=np.float32)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = _mono(samples)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate:
        return samples
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


def prepare_audio(samples: np.ndarray, source_rate: int, config: AudioConfig) -> np.ndarray:
    """Convert audio to fixed-duration mono float32 and trim surrounding silence."""
    waveform = resample_audio(samples, source_rate, config.sample_rate)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio is empty or contains non-finite samples")

    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        threshold = peak * (10.0 ** (config.silence_threshold_db / 20.0))
        active = np.flatnonzero(np.abs(waveform) >= threshold)
        if active.size:
            waveform = waveform[active[0] : active[-1] + 1]
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        if config.normalize_peak and rms >= 1e-3:
            waveform = waveform / max(float(np.max(np.abs(waveform))), 1e-8)

    target = config.sample_count
    if waveform.size > target:
        start = (waveform.size - target) // 2
        waveform = waveform[start : start + target]
    elif waveform.size < target:
        left = (target - waveform.size) // 2
        waveform = np.pad(waveform, (left, target - waveform.size - left))
    return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)


def load_wav(path: str | Path, config: AudioConfig) -> np.ndarray:
    path = Path(path)
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return prepare_audio(samples, sample_rate, config)
    except Exception:
        LOGGER.exception("Failed to read audio file %s", path)
        raise


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _mono(samples), sample_rate, subtype="PCM_16")
