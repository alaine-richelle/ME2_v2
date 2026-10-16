"""Matched offline capture comparisons. No devices, downloads, tuning or audio writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


def worker(args):
    sys.path.insert(0, str(args.source_root / "src"))
    import numpy as np
    import torch

    from me2.app import build_streaming_pipeline, parse_args
    from me2.audio import AudioConfig, load_wav_sequence
    from me2.bundle import load_bundle
    from me2.streaming import replay_chunks
    from me2.vcm.ctc import CommandResult, required_ctc_frames

    logging.disable(logging.CRITICAL)
    settings = parse_args(["--interactive", "--config", str(args.config)])
    torch.set_num_threads(settings.threads)
    wm, wb = load_bundle(settings.wake_model)
    cm, cb = load_bundle(settings.vcm_model)
    rate = wb["audio"]["sample_rate"]
    fixture_audio, usable = load_wav_sequence(args.audio, AudioConfig(**cb["audio"]))
    scenarios = [
        ("immediate_short", 0, [(600, .1), (1000, 0)], False),
        ("pause_1500_short", 1500, [(600, .1), (1000, 0)], False),
        ("pause_2000_short", 2000, [(600, .1), (1000, 0)], False),
        ("speech_4300", 0, [(4300, .1), (1000, 0)], False),
        ("pause_2000_speech_4600", 2000, [(4600, .1), (1000, 0)], False),
        ("internal_pause_600", 0, [(600, .1), (600, 0), (600, .1), (1000, 0)], False),
        ("too_short_noise", 0, [(100, .1), (1000, 0)], False),
        ("duration_exceeded", 0, [(5100, .1), (1000, 0)], False),
        ("no_command", 0, [(4000, 0)], False),
        ("prepared_fixture_immediate", 0, [], True),
        ("prepared_fixture_pause_1500", 1500, [], True),
    ]
    rows = []
    for name, pause, spans, real_command in scenarios:
        p = build_streaming_pipeline(settings, wm, wb, cm, cb)
        triggers = [True]
        p.wake_predictor = lambda _, triggers=triggers: triggers.pop() if triggers else False
        calls = []
        original = p.command_predictor
        def predict(audio, calls=calls, original=original, real_command=real_command):
            calls.append({"samples": len(audio), "nonzero_samples": int(np.count_nonzero(audio))})
            return original(audio) if real_command else CommandResult(
                "play music", "play_music", {}, "", 0, 0, 1, 0, True, None,
            )
        p.command_predictor = predict
        tail = (
            np.concatenate([fixture_audio[:usable], np.zeros(rate)])
            if real_command else np.concatenate([
                np.full(round(ms * rate / 1000), level, np.float32) for ms, level in spans
            ])
        )
        waveform = np.concatenate([
            np.zeros(p.config.wake_window_samples), np.zeros(round(pause * rate / 1000)), tail,
        ]).astype(np.float32)
        events = replay_chunks(p, (
            waveform[i:i + p.config.chunk_samples]
            for i in range(0, len(waveform), p.config.chunk_samples)
        ))
        rows.append({
            "scenario": name, "predictors": "mock_wake_real_command" if real_command else "mock_wake_and_command",
            "calls": calls, "events": [{
                "state": e.state.value, "reason": e.message,
                "ms_since_wake": (e.sample_index - p.config.wake_window_samples) * 1000 / rate,
                "accepted": e.result.accepted if e.result else None,
                "intent": e.result.actionable_intent if e.result else None,
                "diagnostics": dict(getattr(e.result, "diagnostics", {})) if e.result else {},
            } for e in events],
        })
        p.stop()
    frames = int(cb["audio"]["duration_seconds"] * rate) // cb["features"]["hop_length"] + 1
    output_frames = int(cm.output_lengths(torch.tensor([frames]))[0])
    report = {
        "source_root": str(args.source_root), "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "streaming_sha256": hashlib.sha256((args.source_root / "src/me2/streaming.py").read_bytes()).hexdigest(),
        "artifact_hashes": {key: hashlib.sha256(getattr(settings, key).read_bytes()).hexdigest()
                            for key in ("wake_model", "vcm_model")},
        "fixture_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "maximum_feature_frames": frames, "maximum_output_frames": output_frames,
        "infeasible_grammar_terminals": [
            {"text": text, "intent": intent, "required_frames": required_ctc_frames(text)}
            for text, intent in cb["grammar"] if required_ctc_frames(text) > output_frames
        ],
        "scenarios": rows,
    }
    if args.cache_revision:
        import pyarrow as pa
        from sklearn.metrics import accuracy_score, f1_score

        from me2.audio import resample_audio
        from me2.vcm.data import DATASET_REVISION, command_from_row, decode_audio

        selection = json.loads(args.selection.read_text())
        partitions = json.loads(args.partitions.read_text())
        if (args.cache_revision.name != DATASET_REVISION
                or selection["dataset_revision"] != DATASET_REVISION
                or not set(selection["indices"]) <= set(partitions["partitions"]["validation"]["indices"])
                or selection["partition_sha256"] != hashlib.sha256(args.partitions.read_bytes()).hexdigest()):
            raise ValueError("Require the pinned, persisted official-train validation selection")
        shards = sorted(args.cache_revision.glob("*-train-*.arrow"))
        if len(shards) != 2:
            raise ValueError("Expected exactly two official-train shards")
        table = pa.concat_tables([
            pa.ipc.open_stream(pa.memory_map(str(path))).read_all() for path in shards
        ])
        dataset_rows = [table.slice(i, 1).to_pylist()[0] for i in selection["indices"]]
        prepared = [resample_audio(*decode_audio(row["audio"]), rate) for row in dataset_rows]
        report["dataset_revision"] = DATASET_REVISION
        report["validation_indices"] = selection["indices"]
        report["validation_replay"] = {}
        for pause_ms in (0, 1500):
            predictions = []
            for index, row, audio in zip(selection["indices"], dataset_rows, prepared):
                p = build_streaming_pipeline(settings, wm, wb, cm, cb)
                triggers = [True]
                p.wake_predictor = lambda _, triggers=triggers: triggers.pop() if triggers else False
                signal_audio = np.concatenate([
                    np.zeros(p.config.wake_window_samples),
                    np.zeros(round(pause_ms * rate / 1000)), audio, np.zeros(rate * 2),
                ]).astype(np.float32)
                events = replay_chunks(p, (
                    signal_audio[i:i + p.config.chunk_samples]
                    for i in range(0, len(signal_audio), p.config.chunk_samples)
                ))
                decisions = [e.result for e in events if e.result is not None]
                decision = decisions[0] if decisions else None
                predictions.append({
                    "source_index": index, "gold": command_from_row(row),
                    "source_duration_seconds": len(audio) / rate,
                    "accepted": bool(decision and decision.actionable_intent),
                    "intent": decision.actionable_intent if decision else None,
                    "reason": decision.rejection_reason if decision else "no_response",
                    "decisions": len(decisions),
                })
                p.stop()
            gold = [r["gold"] for r in predictions]
            predicted = [r["intent"] or "out_of_scope" for r in predictions]
            report["validation_replay"][str(pause_ms)] = {
                "accuracy": accuracy_score(gold, predicted),
                "macro_f1": f1_score(gold, predicted, labels=cb["labels"], average="macro", zero_division=0),
                "oos_false_accepts": sum(r["gold"] == "out_of_scope" and r["accepted"] for r in predictions),
                "in_scope_rejects": sum(r["gold"] != "out_of_scope" and not r["accepted"] for r in predictions),
                "wrong_accepted": sum(r["accepted"] and r["gold"] != "out_of_scope" and r["gold"] != r["intent"] for r in predictions),
                "predictions": predictions,
            }
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("config/runtime.natural-pause.json"))
    parser.add_argument("--baseline-config", type=Path, default=Path("config/runtime.example.json"))
    parser.add_argument("--audio", type=Path, default=Path("vcm/data/test_sample.wav"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cache-revision", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--partitions", type=Path, default=Path("results/vcm_ctc_partitions.json"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.baseline_root is None or args.output is None:
        parser.error("--baseline-root and a new --output are required")
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    reports = {}
    for name, root, config in [
        ("before", args.baseline_root, args.baseline_config),
        ("candidate", args.source_root, args.config),
    ]:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--source-root", str(root.resolve()), "--config", str(config.resolve()),
            "--audio", str(args.audio.resolve()),
        ]
        if args.cache_revision:
            if args.selection is None:
                parser.error("--selection is required for validation replay")
            command += ["--cache-revision", str(args.cache_revision.resolve()),
                        "--selection", str(args.selection.resolve()),
                        "--partitions", str(args.partitions.resolve())]
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        reports[name] = json.loads(completed.stdout)
    reports["limitations"] = [
        "Offline development mechanics: mocked wake edge, synthetic signals and one existing file fixture.",
        "No real pronunciation review, live microphone, test/holdout tuning or Raspberry Pi measurement.",
        "Candidate profile is provisional until explicitly approved live before/after trials.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(reports, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)
    for i, row in enumerate(reports["before"]["scenarios"]):
        after = reports["candidate"]["scenarios"][i]
        decisions = lambda r: [(e["reason"], e["intent"]) for e in r["events"] if e["accepted"] is not None]
        print(row["scenario"], decisions(row), "->", decisions(after))
    for name, report in reports.items():
        if isinstance(report, dict):
            for pause, summary in report.get("validation_replay", {}).items():
                print(name, "pause", pause, {k: v for k, v in summary.items() if k != "predictions"})


if __name__ == "__main__":
    main()
