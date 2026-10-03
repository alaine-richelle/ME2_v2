"""Evaluate wake classification and false-accept/reject behavior."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from me2.audio import AudioConfig
from me2.bundle import load_bundle
from me2.features import FeatureConfig, LogMel
from me2.metrics import classification_metrics, profile_model, save_evaluation
from me2.wake_word.data import read_manifest
from me2.wake_word.train import WakeDataset, assign_speaker_splits

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def evaluate(model_path: Path, manifest: Path, split: str, output: Path) -> dict:
    rows = read_manifest(manifest)
    model, bundle = load_bundle(model_path)
    assign_speaker_splits(rows, int(bundle.get("seed", 42)))
    selected = [row for row in rows if row["split"] == split]
    dataset = WakeDataset(selected, AudioConfig(**bundle["audio"]), tuple(bundle["labels"]))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    frontend = LogMel(FeatureConfig(**bundle["features"]))
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for waveforms, targets in loader:
            scores = torch.softmax(model(frontend(waveforms)), dim=1).numpy()
            labels.extend(targets.numpy().tolist())
            predictions.extend(scores.argmax(axis=1).tolist())
            probabilities.extend(scores.tolist())
    matrix = np.asarray(probabilities)
    metrics = classification_metrics(labels, predictions, matrix, list(bundle["labels"]))
    positive_id = bundle["labels"].index("ateko")
    truth = np.asarray(labels) == positive_id
    score = matrix[:, positive_id]
    predicted_class = np.asarray(predictions)
    best_threshold = float(bundle.get("threshold", 0.7))
    if split == "validation" and truth.any() and (~truth).any():
        max_false_accept_rate = float(bundle.get("max_false_accept_rate", 0.05))
        candidates = np.unique(np.concatenate((score, [np.nextafter(score.max(), np.inf)])))
        best_recall = -1.0
        best_false_accept_rate = float("inf")
        for candidate in candidates:
            candidate_detected = (score >= candidate) & (predicted_class == positive_id)
            candidate_false_accept_rate = float((~truth & candidate_detected).sum() / (~truth).sum())
            candidate_recall = float((truth & candidate_detected).sum() / truth.sum())
            if candidate_false_accept_rate > max_false_accept_rate:
                continue
            if (candidate_recall, -candidate_false_accept_rate, candidate) > (
                best_recall, -best_false_accept_rate, best_threshold
            ):
                best_recall = candidate_recall
                best_false_accept_rate = candidate_false_accept_rate
                best_threshold = float(candidate)
        bundle["threshold"] = best_threshold
    predicted_wake = (score >= best_threshold) & (predicted_class == positive_id)
    false_accepts = int((~truth & predicted_wake).sum())
    false_rejects = int((truth & ~predicted_wake).sum())
    negative_count = int((~truth).sum())
    positive_count = int(truth.sum())
    true_positives = int((truth & predicted_wake).sum())
    binary_precision = true_positives / max(true_positives + false_accepts, 1)
    binary_recall = true_positives / max(positive_count, 1)
    binary_f1 = (
        2.0 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall else 0.0
    )
    metrics.update({
        "split": split,
        "wake_threshold": best_threshold,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / negative_count if negative_count else None,
        "false_rejects": false_rejects,
        "false_reject_rate": false_rejects / positive_count if positive_count else None,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "wake_detection_accuracy": float(np.mean(truth == predicted_wake)),
        "wake_detection_precision": binary_precision,
        "wake_detection_recall": binary_recall,
        "wake_detection_f1": binary_f1,
        "wake_detection_roc_auc": float(roc_auc_score(truth, score))
        if truth.any() and (~truth).any() else None,
        "syllable_cues_counted_as_non_wake": sum(row["label"] in {"a", "te", "ko"} for row in selected),
    })
    metrics.update(profile_model(model, bundle["input_shape"], model_path))
    predictions_rows = []
    for row, gold, probs, is_wake in zip(selected, labels, matrix, predicted_wake):
        predicted = bundle["labels"][int(np.argmax(probs))]
        predictions_rows.append({
            "input_audio": row["path"],
            "gold_command": row["label"],
            "predicted_command": predicted,
            "confidence": float(np.max(probs)),
            "wake_probability": float(probs[positive_id]),
            "wake_detected": bool(is_wake),
            "correct": row["label"] == predicted,
            "speaker_id": row["speaker_id"],
            "kind": row["kind"],
        })
    save_evaluation(output, metrics, predictions_rows)
    LOGGER.info("Wake %s metrics: %s", split, metrics)
    if split == "validation":
        from me2.bundle import save_bundle

        save_bundle(model_path, bundle)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "wake-word/models/wake_word_model.pkl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "wake-word/data/generated/manifest.csv")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "results/wake_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(args.model, args.manifest, args.split, args.output)


if __name__ == "__main__":
    main()
