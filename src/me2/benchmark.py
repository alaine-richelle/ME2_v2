"""Benchmark one offline ME2 artifact on a representative WAV file."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import soundfile as sf
import torch

from me2.audio import AudioConfig, prepare_audio, prepare_audio_sequence
from me2.bundle import load_bundle, predict_command
from me2.features import FeatureConfig, LogMel

ROOT = Path(__file__).resolve().parents[2]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[position])


def benchmark(
    model_path: Path,
    audio_path: Path,
    *,
    iterations: int,
    warmup: int,
    threads: int,
) -> dict[str, Any]:
    if iterations <= 0 or warmup < 0 or threads <= 0:
        raise ValueError("iterations and threads must be positive; warmup must be non-negative")
    torch.set_num_threads(threads)
    model, bundle = load_bundle(model_path)
    task = str(bundle.get("task", "vcm"))
    audio_config = AudioConfig(**bundle["audio"])
    samples, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    architecture = str(bundle.get("architecture", "TinyAudioCNN"))
    if architecture in {"CharacterAudioCNN", "CompactCTCModel"}:
        waveform, usable_samples = prepare_audio_sequence(
            samples,
            source_rate,
            audio_config,
        )
    else:
        waveform = prepare_audio(samples, source_rate, audio_config)
        usable_samples = waveform.size

    wake_frontend = (
        LogMel(FeatureConfig(**bundle["features"]))
        if task == "wake_word"
        else None
    )

    def run_once():
        if task != "wake_word":
            result = predict_command(
                model,
                bundle,
                waveform,
                sample_length=usable_samples,
            )
            return result, dict(result.stage_timings_ms)
        frontend_started = time.perf_counter()
        features = wake_frontend(waveform)
        frontend_ms = (time.perf_counter() - frontend_started) * 1000.0
        model_started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model(features), dim=1)[0]
        model_ms = (time.perf_counter() - model_started) * 1000.0
        index = int(probabilities.argmax())
        label = str(bundle["labels"][index])
        wake_probability = float(
            probabilities[bundle["labels"].index("ateko")]
        )
        detected = (
            label == "ateko"
            and wake_probability >= float(bundle.get("threshold", 0.7))
        )
        return {
            "accepted": detected,
            "decoded_text": label,
            "intent": "ateko" if detected else None,
            "rejection_reason": None if detected else "wake_not_detected",
        }, {
            "frontend": frontend_ms,
            "acoustic_model": model_ms,
        }

    for _ in range(warmup):
        run_once()

    process = psutil.Process()
    cpu_before = process.cpu_times()
    wall_before = time.perf_counter()
    timings: dict[str, list[float]] = {}
    total_ms: list[float] = []
    last_result = None
    for _ in range(iterations):
        started = time.perf_counter()
        last_result, stage_values = run_once()
        total_ms.append((time.perf_counter() - started) * 1000.0)
        for stage, value in stage_values.items():
            timings.setdefault(stage, []).append(float(value))
    wall_seconds = max(time.perf_counter() - wall_before, 1e-9)
    cpu_after = process.cpu_times()
    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean_ms": float(statistics.fmean(values)),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "maximum_ms": float(max(values)),
        }

    memory = process.memory_info()
    return {
        "benchmark_schema_version": 1,
        "model": {
            "path": str(model_path.resolve()),
            "task": task,
            "architecture": architecture,
            "model_version": str(bundle.get("model_version", "")),
            "grammar_version": str(bundle.get("grammar_version", "")),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "artifact_bytes": model_path.stat().st_size,
        },
        "audio": {
            "path": str(audio_path.resolve()),
            "source_sample_rate": int(source_rate),
            "usable_samples": int(usable_samples),
            "usable_seconds": float(usable_samples / audio_config.sample_rate),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "logical_cpus": os.cpu_count(),
        },
        "run": {
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "process_cpu_percent_of_one_core": 100.0 * cpu_seconds / wall_seconds,
            "rss_bytes_after_run": memory.rss,
            "total": summarize(total_ms),
            "stages": {
                stage: summarize(values)
                for stage, values in sorted(timings.items())
            },
        },
        "last_result": {
            "accepted": bool(
                last_result
                and (
                    last_result["accepted"]
                    if isinstance(last_result, dict)
                    else last_result.accepted
                )
            ),
            "decoded_text": (
                ""
                if last_result is None
                else last_result["decoded_text"]
                if isinstance(last_result, dict)
                else last_result.decoded_text
            ),
            "intent": (
                None
                if last_result is None
                else last_result["intent"]
                if isinstance(last_result, dict)
                else last_result.actionable_intent
            ),
            "rejection_reason": (
                None
                if last_result is None
                else last_result["rejection_reason"]
                if isinstance(last_result, dict)
                else last_result.rejection_reason
            ),
        },
        "limitations": [
            "RSS is process-wide, not model-only.",
            "CPU percentage is measured over the active inference loop.",
            "Temperature, throttling, idle CPU, and sustained-run power require target-hardware tools.",
            "File decoding and endpoint wait time are outside the measured inference loop.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "vcm/models/vcm_ctc_model.me2",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.model,
        args.audio,
        iterations=args.iterations,
        warmup=args.warmup,
        threads=args.threads,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
