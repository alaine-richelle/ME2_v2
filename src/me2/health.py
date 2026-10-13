"""Offline artifact/operator/environment check. Never records or dispatches audio."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from me2.bundle import load_bundle, predict_command, predict_waveform
from me2.contracts import integer
from me2.io_utils import atomic_json, file_sha256
from me2.logging_utils import config_hash, runtime_revision
from me2.runtime_config import load_runtime_defaults
from me2.vcm.ctc import DECODER_RUNTIME_VERSION


def _read_optional(path: str):
    try:
        return Path(path).read_text().strip("\x00\n ")
    except OSError:
        return None


def environment_report() -> dict:
    model = _read_optional("/proc/device-tree/model")
    telemetry = {}
    if model and "Raspberry Pi" in model and shutil.which("vcgencmd"):
        for command in ("measure_temp", "get_throttled"):
            try:
                telemetry[command] = subprocess.check_output(
                    ["vcgencmd", command], text=True, timeout=2,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                telemetry[command] = None
    packages = {}
    for name in ("me2-v2", "torch", "numpy", "scipy", "soundfile", "psutil", "sounddevice"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
        "board_model": model, "measured_pi5": bool(model and "Raspberry Pi 5" in model),
        "os_release": platform.freedesktop_os_release() if sys.platform == "linux" else None,
        "packages": packages, "logical_cpus": psutil.cpu_count(),
        "available_memory_bytes": psutil.virtual_memory().available,
        "available_storage_bytes": shutil.disk_usage(Path.cwd()).free,
        "cpu_governor": _read_optional("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "temperature_millicelsius": _read_optional("/sys/class/thermal/thermal_zone0/temp"),
        "firmware_telemetry": telemetry,
        "quantized_backends": list(torch.backends.quantized.supported_engines),
        "quantization_verified": False,
    }


def inspect_artifact(path: Path, *, expected_sha256=None, require_approved_wake=False) -> dict:
    started = time.perf_counter()
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("release_checksum_mismatch")
    model, bundle = load_bundle(path)
    load_ms = (time.perf_counter() - started) * 1000
    if (require_approved_wake and bundle["task"] == "wake_word"
            and bundle.get("deployment_ready") is not True):
        raise ValueError("wake_deployment_approval_missing")
    samples = np.zeros(round(
        bundle["audio"]["sample_rate"] * bundle["audio"]["duration_seconds"]
    ), dtype=np.float32)
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        if bundle["task"] == "wake_word":
            predict_waveform(model, bundle, samples)
        else:
            predict_command(model, bundle, samples)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "artifact_sha256": digest, "weights_sha256": bundle["weights_sha256"],
        "metadata_sha256": config_hash({k: v for k, v in bundle.items() if k != "state_dict"}),
        "task": bundle["task"], "architecture": bundle.get("architecture", "TinyAudioCNN"),
        "model_version": bundle.get("model_version"),
        "grammar_version": bundle.get("grammar_version"),
        "dataset_revision": bundle.get("dataset_revision"),
        "deployment_ready": bundle.get("deployment_ready"),
        "parameters": sum(p.numel() for p in model.parameters()), "artifact_bytes": path.stat().st_size,
        "load_ms": load_ms, "first_forward_ms": timings[0],
        "warm_forward_ms": timings[1:], "synthetic_operator_smoke_only": True,
        "warnings": [name for name, missing in (
            ("legacy_grammar_without_slot_metadata", bundle.get("grammar_version") == "legacy-training-transcripts"),
            ("training_dataset_provenance_missing", not bundle.get("dataset_revision")),
            ("wake_not_approved", bundle["task"] == "wake_word" and bundle.get("deployment_ready") is not True),
        ) if missing],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--sha256", help="Expected release SHA-256; requires exactly one model")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--require-pi5", action="store_true")
    parser.add_argument("--require-approved-wake", action="store_true")
    parser.add_argument("--devices", action="store_true", help="Inspect input devices; no capture")
    parser.add_argument("--input-device")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "status": "failed", "git_commit": runtime_revision(),
              "decoder_runtime_version": DECODER_RUNTIME_VERSION, "artifacts": [],
              "raw_audio_recorded": False, "network_access_required": False}
    try:
        integer("threads", args.threads, 1, 64)
        torch.set_num_threads(args.threads)
        report["environment"] = environment_report()
        if args.require_pi5 and (
            not report["environment"]["measured_pi5"]
            or report["environment"]["pointer_bits"] != 64
            or platform.machine() != "aarch64"
        ):
            raise ValueError("requires_actual_64bit_raspberry_pi5")
        if args.config:
            defaults = load_runtime_defaults(args.config)
            report["runtime_config_sha256"] = config_hash(defaults)
            args.model.extend(defaults[k] for k in ("wake_model", "vcm_model") if k in defaults)
        if args.sha256 and len(args.model) != 1:
            raise ValueError("checksum_requires_exactly_one_artifact")
        for path in args.model:
            report["artifacts"].append(inspect_artifact(
                path, expected_sha256=args.sha256,
                require_approved_wake=args.require_approved_wake,
            ))
        if args.devices:
            import sounddevice as sd

            report["input_devices"] = [
                {"index": i, "channels": d["max_input_channels"],
                 "default_sample_rate": d["default_samplerate"]}
                for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
            ]
            if not report["input_devices"]:
                raise RuntimeError("no_input_devices")
            device = args.input_device
            if device and device.isdecimal():
                device = int(device)
            sd.check_input_settings(device=device, channels=1, dtype="float32",
                                    samplerate=args.sample_rate)
            report["device_format_check"] = "passed_no_capture"
        report["rss_bytes"] = psutil.Process().memory_info().rss
        report["status"] = "passed_operator_health_not_deployment_approval"
    except Exception as error:  # noqa: BLE001 -- CLI boundary records failure and exits nonzero.
        report["error_type"] = type(error).__name__
        safe_reasons = {
            "no_input_devices", "release_checksum_mismatch",
            "wake_deployment_approval_missing", "requires_actual_64bit_raspberry_pi5",
            "checksum_requires_exactly_one_artifact",
        }
        report["error_code"] = str(error) if str(error) in safe_reasons else "validation_failed"
        # Do not persist exception messages, which can contain private filenames/device names.
        report["next_action"] = "check_schema_hash_permissions_runtime_and_target"
        if args.devices:
            report["next_action"] = "connect_input_device_and_check_os_microphone_permissions"
    if args.output:
        try:
            atomic_json(args.output, report)
        except OSError:
            print("Health output is unwritable; no report saved.", file=sys.stderr)
            raise SystemExit(1) from None
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
