"""Read-only wake evaluation with pronunciation and provenance reporting."""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from me2.audio import AudioConfig
from me2.bundle import load_bundle
from me2.features import FeatureConfig, LogMel
from me2.metrics import classification_metrics, profile_model, save_evaluation
from me2.wake_word.data import (
    CANONICAL_WAKE_LABEL,
    LEGACY_DATASET_VERSION,
    PRONUNCIATION_GUIDE,
    read_manifest,
)
from me2.wake_word.train import WakeDataset

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _manifest_version(rows: list[dict[str, str]]) -> str:
    versions = {row["dataset_version"] for row in rows}
    if len(versions) != 1:
        raise ValueError(f"Evaluation manifest mixes dataset versions: {sorted(versions)}")
    return versions.pop()


def _check_compatibility(
    bundle: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    allow_dataset_mismatch: bool,
) -> None:
    manifest_version = _manifest_version(rows)
    model_version = str(
        bundle.get("wake_dataset_version", LEGACY_DATASET_VERSION)
    )
    if model_version != manifest_version and not allow_dataset_mismatch:
        raise ValueError(
            f"Wake dataset/model mismatch: model={model_version!r}, "
            f"manifest={manifest_version!r}. Use --allow-dataset-mismatch only "
            "for an explicitly labeled historical baseline experiment."
        )
    guide = str(bundle.get("pronunciation_guide", PRONUNCIATION_GUIDE))
    if guide != PRONUNCIATION_GUIDE and not allow_dataset_mismatch:
        raise ValueError(f"Wake pronunciation mismatch: {guide!r}")


def score_rows(
    model_path: Path,
    manifest: Path,
    split: str,
    *,
    allow_dataset_mismatch: bool = False,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, Any]]:
    rows = read_manifest(manifest)
    model, bundle = load_bundle(model_path)
    if bundle.get("task") != "wake_word":
        raise ValueError("Wake evaluation requires a wake_word artifact")
    _check_compatibility(
        bundle,
        rows,
        allow_dataset_mismatch=allow_dataset_mismatch,
    )
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No wake rows assigned to split {split!r}")
    dataset = WakeDataset(
        selected,
        AudioConfig(**bundle["audio"]),
        tuple(bundle["labels"]),
        label_mode=str(bundle.get("label_mode", "five_class")),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    frontend = LogMel(FeatureConfig(**bundle["features"]))
    labels: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for waveforms, targets in loader:
            logits = model(frontend(waveforms))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Wake model produced non-finite logits")
            scores = torch.softmax(logits, dim=1).cpu().numpy()
            labels.extend(targets.numpy().tolist())
            probabilities.extend(scores.tolist())
    return selected, np.asarray(labels), np.asarray(probabilities), bundle


def evaluate(
    model_path: Path,
    manifest: Path,
    split: str,
    output: Path,
    *,
    allow_dataset_mismatch: bool = False,
) -> dict[str, Any]:
    selected, labels, matrix, bundle = score_rows(
        model_path,
        manifest,
        split,
        allow_dataset_mismatch=allow_dataset_mismatch,
    )
    predictions = matrix.argmax(axis=1)
    class_names = list(bundle["labels"])
    metrics = classification_metrics(labels, predictions, matrix, class_names)
    positive_id = class_names.index(CANONICAL_WAKE_LABEL)
    truth = labels == positive_id
    score = matrix[:, positive_id]
    predicted_class = predictions
    threshold = float(bundle["threshold"])
    predicted_wake = (score >= threshold) & (predicted_class == positive_id)
    false_accepts = int((~truth & predicted_wake).sum())
    false_rejects = int((truth & ~predicted_wake).sum())
    negative_count = int((~truth).sum())
    positive_count = int(truth.sum())
    true_positives = int((truth & predicted_wake).sum())
    precision = true_positives / max(true_positives + false_accepts, 1)
    recall = true_positives / max(positive_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    continuous_negative_rows = [
        row
        for row, is_positive in zip(selected, truth)
        if not is_positive and row["kind"] == "continuous_negative"
    ]
    continuous_negative_seconds = sum(
        float(row.get("duration_seconds") or 0.0)
        for row in continuous_negative_rows
    )
    continuous_false_activations = sum(
        bool(detected)
        for row, is_positive, detected in zip(selected, truth, predicted_wake)
        if not is_positive and row["kind"] == "continuous_negative"
    )
    false_activations_per_hour = (
        continuous_false_activations / (continuous_negative_seconds / 3600.0)
        if continuous_negative_seconds > 0
        else None
    )

    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positives": 0, "detected": 0, "negatives": 0, "false_accepts": 0}
    )
    for row, is_positive, detected in zip(selected, truth, predicted_wake):
        group = by_source[row["source"]]
        if is_positive:
            group["positives"] += 1
            group["detected"] += int(detected)
        else:
            group["negatives"] += 1
            group["false_accepts"] += int(detected)

    metrics.update(
        {
            "split": split,
            "wake_threshold": threshold,
            "smoothing_policy": bundle.get("streaming_policy", {}),
            "false_accepts": false_accepts,
            "false_accept_rate": (
                false_accepts / negative_count if negative_count else None
            ),
            "false_accept_rate_95pct_wilson": _wilson(
                false_accepts, negative_count
            ),
            "false_rejects": false_rejects,
            "false_reject_rate": (
                false_rejects / positive_count if positive_count else None
            ),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "wake_detection_precision": precision,
            "wake_detection_recall": recall,
            "wake_detection_accuracy": float(np.mean(truth == predicted_wake)),
            "wake_recall_95pct_wilson": _wilson(
                true_positives, positive_count
            ),
            "wake_detection_f1": f1,
            "wake_detection_roc_auc": (
                float(roc_auc_score(truth, score))
                if truth.any() and (~truth).any()
                else None
            ),
            "continuous_negative_seconds": continuous_negative_seconds,
            "continuous_false_activations": continuous_false_activations,
            "false_activations_per_hour": false_activations_per_hour,
            "false_activations_note": (
                "Unavailable without rows explicitly annotated as continuous_negative."
                if false_activations_per_hour is None
                else "Counts detector events over annotated continuous-negative audio."
            ),
            "detection_delay_ms": None,
            "detection_delay_note": (
                "Unavailable until reviewed wake boundaries and streaming event "
                "timestamps are present."
            ),
            "duplicate_trigger_rate": None,
            "pronunciation_guide": PRONUNCIATION_GUIDE,
            "dataset_version": _manifest_version(selected),
            "review_status_counts": dict(
                Counter(row["review_status"] for row in selected)
            ),
            "independent_speakers": len(
                {row["speaker_id"] for row in selected}
            ),
            "source_breakdown": dict(by_source),
            "syllable_cues_counted_as_non_wake": sum(
                row["label"] in {"a", "te", "ko"} for row in selected
            ),
            "read_only_evaluation": True,
        }
    )
    model, _ = load_bundle(model_path)
    metrics.update(profile_model(model, bundle["input_shape"], model_path))

    prediction_rows = []
    for row, gold, probabilities, detected in zip(
        selected,
        labels,
        matrix,
        predicted_wake,
    ):
        prediction_rows.append(
            {
                "input_audio": row["path"],
                "recording_id": row["recording_id"],
                "gold_command": class_names[int(gold)],
                "predicted_command": class_names[int(np.argmax(probabilities))],
                "confidence": float(np.max(probabilities)),
                "wake_probability": float(probabilities[positive_id]),
                "wake_detected": bool(detected),
                "correct": int(gold) == int(np.argmax(probabilities)),
                "speaker_id": row["speaker_id"],
                "session_id": row["session_id"],
                "source": row["source"],
                "kind": row["kind"],
                "review_status": row["review_status"],
                "pronunciation_guide": row["pronunciation_guide"],
                "is_synthetic": row["is_synthetic"],
                "device": row["device"],
                "room": row["room"],
            }
        )
    save_evaluation(output, metrics, prediction_rows)
    LOGGER.info("Wake %s metrics: %s", split, metrics)
    return metrics


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
    parser.add_argument(
        "--split",
        choices=("train", "validation", "calibration", "test"),
        default="test",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/wake_v3_test")
    parser.add_argument("--allow-dataset-mismatch", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(
        args.model,
        args.manifest,
        args.split,
        args.output,
        allow_dataset_mismatch=args.allow_dataset_mismatch,
    )


if __name__ == "__main__":
    main()
