"""Small, dependency-light audio loading and normalization helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from me2.contracts import finite, integer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 4.0
    silence_threshold_db: float = -48.0
    normalize_peak: bool = True

    def __post_init__(self) -> None:
        integer("sample_rate", self.sample_rate, 1, 192_000)
        finite("duration_seconds", self.duration_seconds, 0.01, 60.0)
        finite("silence_threshold_db", self.silence_threshold_db, -160.0, 0.0)
        if not isinstance(self.normalize_peak, bool):
            raise TypeError("normalize_peak must be boolean")

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
    integer("source_rate", source_rate, 1, 192_000)
    integer("target_rate", target_rate, 1, 192_000)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    if source_rate == target_rate:
        return samples
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


def prepare_audio(samples: np.ndarray, source_rate: int, config: AudioConfig) -> np.ndarray:
    """Convert audio to fixed-duration mono float32 and trim surrounding silence."""
    waveform = _trim_and_normalize(samples, source_rate, config)
    target = config.sample_count
    if waveform.size > target:
        start = (waveform.size - target) // 2
        waveform = waveform[start : start + target]
    elif waveform.size < target:
        left = (target - waveform.size) // 2
        waveform = np.pad(waveform, (left, target - waveform.size - left))
    return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)


def prepare_audio_sequence(
    samples: np.ndarray,
    source_rate: int,
    config: AudioConfig,
) -> tuple[np.ndarray, int]:
    """Prepare right-padded CTC audio and return its usable sample length.

    Unlike ``prepare_audio``, this function never silently crops a long command.
    """

    waveform = _trim_and_normalize(samples, source_rate, config)
    target = config.sample_count
    if waveform.size > target:
        raise ValueError(
            f"Audio has {waveform.size} samples after trimming, exceeding the "
            f"configured maximum of {target}"
        )
    usable_samples = int(waveform.size)
    waveform = np.pad(waveform, (0, target - waveform.size))
    return (
        np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False),
        usable_samples,
    )


def _trim_and_normalize(
    samples: np.ndarray,
    source_rate: int,
    config: AudioConfig,
) -> np.ndarray:
    if config.sample_rate <= 0 or config.duration_seconds <= 0:
        raise ValueError("Audio sample rate and duration must be positive")
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
    return waveform.astype(np.float32, copy=False)


def load_wav(path: str | Path, config: AudioConfig) -> np.ndarray:
    path = Path(path)
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return prepare_audio(samples, sample_rate, config)
    except Exception:
        LOGGER.exception("Failed to read audio file %s", path)
        raise


def load_wav_sequence(path: str | Path, config: AudioConfig) -> tuple[np.ndarray, int]:
    """Load a CTC utterance without silently cropping a command ending."""

    path = Path(path)
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return prepare_audio_sequence(samples, sample_rate, config)
    except Exception:
        LOGGER.exception("Failed to read CTC audio file %s", path)
        raise


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _mono(samples), sample_rate, subtype="PCM_16")
