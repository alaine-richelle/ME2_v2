"""Versioned runtime-only settings: defaults < JSON file < explicit CLI flags."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from me2.contracts import finite, integer

_NUMERIC_ALSA_SELECTOR = re.compile(r"(?:plughw|hw):\d+,\d+", re.IGNORECASE)


@dataclass(frozen=True)
class RuntimeSettings:
    threads: int = 2
    audio_backend: str = "auto"
    alsa_device: str = "auto"
    stream_chunk_ms: float = 100.0
    wake_hop_ms: float = 250.0
    pre_roll_ms: float = 250.0
    onset_timeout_ms: float = 1000.0
    minimum_speech_ms: float = 250.0
    endpoint_silence_ms: float = 500.0
    maximum_command_ms: float = 0.0  # zero inherits the artifact limit
    cooldown_ms: float = 1000.0
    speech_rms_threshold: float = 0.008
    noise_floor_calibration_ms: float = 1000.0
    noise_floor_multiplier: float = 2.5
    speech_hysteresis_ratio: float = 0.7
    clipping_warning_fraction: float = 0.02
    clipping_reject_fraction: float = 1.0
    post_wake_guard_ms: float = 150.0
    queue_chunks: int = 32

    def __post_init__(self) -> None:
        if self.audio_backend not in {"auto", "sounddevice", "arecord"}:
            raise ValueError("audio_backend must be auto, sounddevice, or arecord")
        if not isinstance(self.alsa_device, str) or not self.alsa_device.strip():
            raise ValueError("alsa_device must be a non-empty ALSA device name")
        if _NUMERIC_ALSA_SELECTOR.search(self.alsa_device):
            raise ValueError("alsa_device must not persist a numeric ALSA card/device selector")
        integer("threads", self.threads, 1, 64)
        integer("queue_chunks", self.queue_chunks, 1, 128)
        for name, value in asdict(self).items():
            if name.endswith("_ms"):
                finite(name, value, 0 if name in {"pre_roll_ms", "maximum_command_ms"} else 1, 60_000)
        finite("speech_rms_threshold", self.speech_rms_threshold, 1e-6, 1.0)
        finite("noise_floor_multiplier", self.noise_floor_multiplier, 1.0, 100.0)
        finite("speech_hysteresis_ratio", self.speech_hysteresis_ratio, 0.01, 1.0)
        finite("clipping_warning_fraction", self.clipping_warning_fraction, 0.0, 1.0)
        finite("clipping_reject_fraction", self.clipping_reject_fraction, 0.0, 1.0)
        if self.clipping_warning_fraction > self.clipping_reject_fraction:
            raise ValueError("clipping warning fraction must not exceed reject fraction")
        if self.maximum_command_ms and self.minimum_speech_ms >= self.maximum_command_ms:
            raise ValueError("minimum speech must be shorter than the maximum command")

    @classmethod
    def from_namespace(cls, args):
        return cls(**{key: getattr(args, key) for key in cls.__dataclass_fields__})


def load_runtime_defaults(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Runtime config requires schema_version=1")
    integer("runtime schema_version", value.get("schema_version"), 1, 1)
    if set(value) - {"schema_version", "runtime", "paths"}:
        raise ValueError("Unknown runtime configuration section")
    runtime = asdict(RuntimeSettings(**value.get("runtime", {})))
    paths = value.get("paths", {})
    if not isinstance(paths, dict) or set(paths) - {
        "wake_model", "vcm_model", "log_file",
    }:
        raise ValueError("Unknown runtime path setting")
    for key, raw_path in paths.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Runtime paths must be non-empty strings")
        candidate = Path(raw_path)
        runtime[key] = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
    return runtime
