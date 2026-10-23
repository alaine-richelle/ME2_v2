"""Private, five-sample Pi wake calibration; never stores audio in the repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from me2.audio import AudioConfig, prepare_audio
from me2.bundle import load_bundle, predict_waveform

PRIVATE_ROOT = Path("/home/rpiadmin/me2-validation-evidence/wake-calibration").resolve()


def _private_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != PRIVATE_ROOT and PRIVATE_ROOT not in resolved.parents:
        raise ValueError("private calibration inputs and output must remain under the approved external root")
    return resolved


def _validate_sample(path: Path, audio: np.ndarray, rate: int) -> str | None:
    if rate <= 0 or audio.size == 0 or not np.isfinite(audio).all():
        return "invalid_format"
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    if mono.ndim != 1 or mono.size / rate < 0.25 or mono.size / rate > 8.0:
        return "invalid_duration"
    peak = float(np.max(np.abs(mono)))
    if peak >= 0.999:
        return "clipping"
    if float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) < 0.001:
        return "silence"
    return None


def score_samples(model, bundle: dict, samples: list[Path]) -> dict:
    if len(samples) != 5:
        raise ValueError("exactly five private wake samples are required")
    results = []
    for index, path in enumerate(samples, start=1):
        path = _private_path(path)
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        invalid = _validate_sample(path, np.asarray(audio), int(rate))
        if invalid:
            results.append({"sample": index, "valid": False, "reason": invalid, "detected": False})
            continue
        prepared = prepare_audio(np.asarray(audio), int(rate), AudioConfig(**bundle["audio"]))
        label, probability, _ = predict_waveform(model, bundle, prepared)
        detected = label == "ateko" and probability >= float(bundle.get("threshold", 0.7))
        results.append({"sample": index, "valid": True, "reason": "accepted" if detected else "wake_rejected", "detected": detected})
    valid = sum(item["valid"] for item in results)
    detected = sum(item["detected"] for item in results)
    return {"schema_version": 1, "sample_count": len(results), "valid_samples": valid,
            "full_detections": detected, "passed": valid == 5 and detected >= 4,
            "samples": results, "training_permitted": False}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = _private_path(args.output)
    model, bundle = load_bundle(args.model)
    if bundle.get("task") != "wake_word":
        raise ValueError("private calibration requires a wake_word model")
    report = score_samples(model, bundle, args.sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid_samples", "full_detections", "passed")}, sort_keys=True))


if __name__ == "__main__":
    main()
