"""Explicitly calibrate wake acceptance on the frozen calibration split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from me2.bundle import save_bundle
from me2.wake_word.data import CANONICAL_WAKE_LABEL
from me2.wake_word.evaluate import score_rows

ROOT = Path(__file__).resolve().parents[3]


def calibrate_artifact(
    model_path: Path,
    manifest: Path,
    *,
    split: str = "calibration",
    target_false_accept_rate: float = 0.05,
    output_report: Path | None = None,
) -> dict[str, Any]:
    if not 0.0 <= target_false_accept_rate < 1.0:
        raise ValueError("target_false_accept_rate must be in [0, 1)")
    selected, labels, probabilities, bundle = score_rows(
        model_path,
        manifest,
        split,
    )
    class_names = list(bundle["labels"])
    positive_id = class_names.index(CANONICAL_WAKE_LABEL)
    truth = labels == positive_id
    if not truth.any() or not (~truth).any():
        raise ValueError("Wake calibration requires positive and negative examples")
    predicted_class = probabilities.argmax(axis=1)
    scores = probabilities[:, positive_id]
    candidates = np.unique(
        np.concatenate((scores, [np.nextafter(scores.max(), np.inf)]))
    )
    best: tuple[float, float, float] | None = None
    best_threshold = float("inf")
    for threshold in candidates:
        detected = (scores >= threshold) & (predicted_class == positive_id)
        false_accept_rate = float((~truth & detected).sum() / (~truth).sum())
        if false_accept_rate > target_false_accept_rate:
            continue
        recall = float((truth & detected).sum() / truth.sum())
        # When several thresholds have identical recall and false-accept rate,
        # retain the lowest one. The previous highest-threshold tie break could
        # reject otherwise recognized positives under tiny score shifts.
        candidate = (recall, -false_accept_rate, -float(threshold))
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    detected = (scores >= best_threshold) & (predicted_class == positive_id)
    false_accepts = int((~truth & detected).sum())
    true_positives = int((truth & detected).sum())
    report = {
        "calibration_schema_version": 1,
        "split": split,
        "rows": len(selected),
        "positive_rows": int(truth.sum()),
        "negative_rows": int((~truth).sum()),
        "target_false_accept_rate": target_false_accept_rate,
        "threshold": best_threshold,
        "observed_false_accepts": false_accepts,
        "observed_false_accept_rate": false_accepts / int((~truth).sum()),
        "observed_recall": true_positives / int(truth.sum()),
        "calibration_mode": "clip_proxy_pending_continuous_validation",
        "warning": (
            "This is not false activations per hour. Continuous validation audio "
            "with event boundaries is still required to calibrate smoothing and cooldown."
        ),
    }
    bundle["threshold"] = best_threshold
    bundle["wake_calibration"] = report
    save_bundle(model_path, bundle)
    if output_report is not None:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_report.with_suffix(output_report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "wake-word/models/wake_word_model_v3_five_class.me2",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "wake-word/data/v3/synthetic/manifest.csv",
    )
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/wake_v2_calibration.json",
    )
    args = parser.parse_args()
    report = calibrate_artifact(
        args.model,
        args.manifest,
        split=args.split,
        target_false_accept_rate=args.target_far,
        output_report=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
