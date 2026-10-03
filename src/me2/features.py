"""Compact log-Mel feature extraction implemented with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 40
    f_min: float = 40.0
    f_max: float = 7_600.0


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


class LogMel:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def __call__(self, waveform: np.ndarray | Tensor) -> Tensor:
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got {tuple(tensor.shape)}")
        window = torch.hann_window(self.config.n_fft, device=tensor.device)
        spectrum = torch.stft(
            tensor,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs().pow(2)
        filters = mel_filterbank(self.config, tensor.device, tensor.dtype)
        mel_power = torch.matmul(filters.unsqueeze(0), spectrum)
        return torch.log(mel_power.clamp_min(1e-8)).unsqueeze(1)
