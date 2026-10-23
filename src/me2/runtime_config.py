"""Versioned runtime-only settings: defaults < JSON file < explicit CLI flags."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from me2.contracts import finite, integer


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
    queue_chunks: int = 32

    def __post_init__(self) -> None:
        if self.audio_backend not in {"auto", "sounddevice", "arecord"}:
            raise ValueError("audio_backend must be auto, sounddevice, or arecord")
        if not isinstance(self.alsa_device, str) or not self.alsa_device.strip():
            raise ValueError("alsa_device must be a non-empty ALSA device name")
        integer("threads", self.threads, 1, 64)
        integer("queue_chunks", self.queue_chunks, 1, 128)
        for name, value in asdict(self).items():
            if name.endswith("_ms"):
                finite(name, value, 0 if name in {"pre_roll_ms", "maximum_command_ms"} else 1, 60_000)
        finite("speech_rms_threshold", self.speech_rms_threshold, 1e-6, 1.0)
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
