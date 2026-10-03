"""Waveform augmentation; callers must apply this only on training examples."""

from __future__ import annotations

import numpy as np


def augment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    gain_db: float = 8.0,
    noise_snr_db_min: float = 8.0,
    noise_snr_db_max: float = 30.0,
    max_shift_ms: float = 100.0,
) -> np.ndarray:
    augmented = np.asarray(waveform, dtype=np.float32).copy()
    gain = 10.0 ** (rng.uniform(-gain_db, gain_db) / 20.0)
    augmented *= gain
    shift_limit = round(sample_rate * max_shift_ms / 1000)
    shift = int(rng.integers(-shift_limit, shift_limit + 1)) if shift_limit else 0
    if shift:
        augmented = np.roll(augmented, shift)
        if shift > 0:
            augmented[:shift] = 0
        else:
            augmented[shift:] = 0
    signal_rms = float(np.sqrt(np.mean(np.square(augmented))))
    if signal_rms > 1e-7 and rng.random() < 0.5:
        snr_db = rng.uniform(noise_snr_db_min, noise_snr_db_max)
        noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        augmented += rng.normal(0.0, noise_rms, augmented.shape).astype(np.float32)
    return np.clip(augmented, -1.0, 1.0)
