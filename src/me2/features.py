"""Compact log-Mel feature extraction implemented with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from me2.contracts import finite, integer


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 40
    f_min: float = 40.0
    f_max: float = 7_600.0

    def __post_init__(self) -> None:
        integer("feature sample_rate", self.sample_rate, 1, 192_000)
        integer("n_fft", self.n_fft, 4, 8192)
        integer("hop_length", self.hop_length, 1, self.n_fft)
        integer("n_mels", self.n_mels, 1, 256)
        finite("f_min", self.f_min, 0.0, self.sample_rate / 2)
        finite("f_max", self.f_max, 0.0, self.sample_rate / 2)
        if self.f_min >= self.f_max:
            raise ValueError("Invalid Mel filterbank bounds")


def _hz_to_mel(value: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def mel_filterbank(config: FeatureConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    bins = config.n_fft // 2 + 1
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(config.f_min)),
        _hz_to_mel(torch.tensor(config.f_max)),
        config.n_mels + 2,
    )
    hz_edges = _mel_to_hz(mel_edges).to(device=device, dtype=dtype)
    frequencies = torch.linspace(0, config.sample_rate / 2, bins, device=device, dtype=dtype)
    lower = (frequencies[None, :] - hz_edges[:-2, None]) / (
        hz_edges[1:-1, None] - hz_edges[:-2, None]
    ).clamp_min(1e-8)
    upper = (hz_edges[2:, None] - frequencies[None, :]) / (
        hz_edges[2:, None] - hz_edges[1:-1, None]
    ).clamp_min(1e-8)
    return torch.minimum(lower, upper).clamp_min(0.0)


@lru_cache(maxsize=16)
def _cached_frontend_tensors(
    config: FeatureConfig,
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    device = torch.device(device_type, device_index)
    window = torch.hann_window(config.n_fft, device=device, dtype=dtype)
    filters = mel_filterbank(config, device, dtype)
    return window, filters


class LogMel:
    def __init__(self, config: FeatureConfig) -> None:
        if config.sample_rate <= 0 or config.n_fft <= 0 or config.hop_length <= 0:
            raise ValueError("Feature sample rate, FFT size, and hop length must be positive")
        if config.n_mels <= 0 or not 0 <= config.f_min < config.f_max <= config.sample_rate / 2:
            raise ValueError("Invalid Mel filterbank bounds")
        self.config = config

    def frame_lengths(self, sample_lengths: Tensor) -> Tensor:
        """Return STFT frame counts for center=True."""

        return torch.div(sample_lengths, self.config.hop_length, rounding_mode="floor") + 1

    def __call__(self, waveform: np.ndarray | Tensor) -> Tensor:
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got {tuple(tensor.shape)}")
        if tensor.shape[0] < 1 or tensor.shape[1] <= self.config.n_fft // 2:
            raise ValueError("Audio must exceed half the FFT size for reflected STFT padding")
        if not torch.isfinite(tensor).all():
            raise ValueError("Audio contains non-finite samples")
        window, filters = _cached_frontend_tensors(
            self.config,
            tensor.device.type,
            tensor.device.index,
            tensor.dtype,
        )
        spectrum = torch.stft(
            tensor,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs().pow(2)
        mel_power = torch.matmul(filters.unsqueeze(0), spectrum)
        return torch.log(mel_power.clamp_min(1e-8)).unsqueeze(1)
