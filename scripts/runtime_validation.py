"""Fixed official-train validation subset; never downloads or tunes anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import torch
from sklearn.metrics import accuracy_score, f1_score

from me2.audio import AudioConfig, prepare_audio, prepare_audio_sequence
from me2.bundle import load_bundle, predict_command
from me2.io_utils import file_sha256
from me2.vcm.data import DATASET_ID, DATASET_REVISION, command_from_row, decode_audio


def sha256(path):
    return file_sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-revision", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; evidence must not be overwritten")
    if args.per_label <= 0 or args.threads <= 0:
        raise ValueError("per-label and threads must be positive")
    torch.set_num_threads(args.threads)
    partitions = json.loads(args.partitions.read_text())
    if partitions["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Wrong persisted dataset revision")
    if args.cache_revision.name != DATASET_REVISION:
        raise ValueError("Expected the pinned revision cache directory")
    shards = sorted(args.cache_revision.glob("ai231-me2-voice-commands-train-*.arrow"))
    if len(shards) != 2:
        raise ValueError("Expected exactly the two official-train shards")
    table = pa.concat_tables([
        pa.ipc.open_stream(pa.memory_map(str(path))).read_all() for path in shards
    ])
    metadata = table.drop(["audio"]).to_pylist()
    indices = partitions["partitions"]["validation"]["indices"]
    by_label = {}
    for index in indices:
        by_label.setdefault(command_from_row(metadata[index]), []).append(index)
    selected = []
    for label, candidates in sorted(by_label.items()):
        selected.extend(sorted(candidates, key=lambda i: hashlib.sha256(
            f"me2-runtime-validation-v1:{i}".encode()
        ).hexdigest())[:args.per_label])
    rows = [table.slice(index, 1).to_pylist()[0] for index in selected]
    audio = [decode_audio(row["audio"]) for row in rows]
    gold = [command_from_row(row) for row in rows]
    process = psutil.Process()
    report = {
        "schema_version": 1, "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "protocol": "official_train/persisted_validation/class_balanced_regression_only",
        "indices": selected, "supports": dict(Counter(gold)),
        "speakers": len({row["speaker_id"] for row in rows}),
        "partition_sha256": sha256(args.partitions),
        "shards": {path.name: sha256(path) for path in shards},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "repository_path": subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "script_sha256": file_sha256(__file__),
        "git_tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "tests", "pyproject.toml"]
        )).hexdigest(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "threads": torch.get_num_threads(),
        },
        "models": [],
        "limitations": [
            "Small class-balanced regression sample, not a final assessment.",
            "Legacy training ancestry is incomplete; speaker independence from its weights is unproven.",
            "No test/holdout, no tuning, no training, no microphone or Pi measurements.",
            "Timing excludes capture/endpointing; RSS includes resident dataset mappings.",
        ],
    }
    for model_path in args.model:
        started = time.perf_counter()
        model, bundle = load_bundle(model_path)
        load_ms = (time.perf_counter() - started) * 1000
        config = AudioConfig(**bundle["audio"])
        sequence = bundle.get("architecture") in {"CharacterAudioCNN", "CompactCTCModel"}
        predictions = []
        # Invalid/overlong samples remain explicit failures in the denominator.
        prepared = []
        for samples, rate in audio:
            try:
                prepared.append(prepare_audio_sequence(samples, rate, config) if sequence
                                else (prepare_audio(samples, rate, config), None))
            except ValueError as error:
                if "exceeding the configured maximum" not in str(error):
                    raise
                prepared.append((None, None))
        for _ in range(3):
            waveform, length = next(item for item in prepared if item[0] is not None)
            predict_command(model, bundle, waveform, sample_length=length)
        wall = time.perf_counter()
        cpu = process.cpu_times()
        for index, row, (waveform, length) in zip(selected, rows, prepared):
            if waveform is None:
                predictions.append({
                    "source_index": index, "gold_intent": command_from_row(row),
                    "accepted": False, "predicted_intent": None, "candidate": "",
                    "rejection_reason": "audio_exceeds_artifact_duration",
                    "score": None, "timings_ms": {}, "total_ms": None,
                })
                continue
            started = time.perf_counter()
            result = predict_command(model, bundle, waveform, sample_length=length)
            predictions.append({
                "source_index": index,
                "gold_intent": command_from_row(row),
                "accepted": result.accepted, "predicted_intent": result.actionable_intent,
                "candidate": result.decoded_text, "rejection_reason": result.rejection_reason,
                "score": result.normalized_log_score if np.isfinite(result.normalized_log_score) else None,
                "timings_ms": dict(result.stage_timings_ms),
                "total_ms": (time.perf_counter() - started) * 1000,
            })
        elapsed = time.perf_counter() - wall
        cpu_end = process.cpu_times()
        predicted = [p["predicted_intent"] or "out_of_scope" for p in predictions]
        oos = sum(g == "out_of_scope" for g in gold)
        latencies = [p["total_ms"] for p in predictions if p["total_ms"] is not None]
        report["models"].append({
            "artifact": model_path.name, "sha256": sha256(model_path),
            "weights_sha256": bundle["weights_sha256"], "load_ms": load_ms,
            "model_version": bundle.get("model_version"),
            "grammar_version": bundle.get("grammar_version"),
            "acceptance_policy": bundle.get("acceptance_policy"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "artifact_bytes": model_path.stat().st_size, "labels": bundle["labels"],
            "accuracy": accuracy_score(gold, predicted),
            "macro_f1": f1_score(gold, predicted, labels=bundle["labels"], average="macro", zero_division=0),
            "oos_count": oos,
            "oos_false_accepts": sum(g == "out_of_scope" and p != g for g, p in zip(gold, predicted)),
            "in_scope_count": len(gold) - oos,
            "in_scope_rejects": sum(g != "out_of_scope" and p == "out_of_scope" for g, p in zip(gold, predicted)),
            "wrong_accepted": sum(g != "out_of_scope" and p not in {g, "out_of_scope"} for g, p in zip(gold, predicted)),
            "coverage": sum(p["accepted"] for p in predictions) / len(predictions),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "active_cpu_percent_one_core": 100 * (cpu_end.user + cpu_end.system - cpu.user - cpu.system) / elapsed,
            "rss_bytes": process.memory_info().rss,
            "predictions": predictions,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(args.output)
        finally:
            temp.unlink(missing_ok=True)
    print(json.dumps({m["artifact"]: {k: m[k] for k in (
        "accuracy", "macro_f1", "oos_false_accepts", "in_scope_rejects",
        "wrong_accepted", "coverage", "p50_ms", "p95_ms"
    )} for m in report["models"]}, indent=2))


if __name__ == "__main__":
    main()
