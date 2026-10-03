"""Metric and prediction artifact writers."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], probabilities: np.ndarray, names: list[str]
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    result: dict[str, Any] = {
        "class_names": names,
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "precision_macro": float(precision_score(labels_array, predictions_array, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels_array, predictions_array, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels_array, predictions_array, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels_array, predictions_array, labels=np.arange(len(names))).tolist(),
        "per_class": classification_report(
            labels_array,
            predictions_array,
            labels=np.arange(len(names)),
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        if len(names) == 2:
            result["roc_auc"] = float(roc_auc_score(labels_array, probabilities[:, 1]))
        elif len(np.unique(labels_array)) > 1:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(labels_array, probabilities, labels=np.arange(len(names)), multi_class="ovr", average="macro")
            )
    except ValueError:
        result["roc_auc"] = None
    return result


def profile_model(model: nn.Module, input_shape: Sequence[int], model_path: str | Path) -> dict[str, Any]:
    model = model.cpu().eval()
    inputs = torch.zeros((1, *input_shape), dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(3):
            model(inputs)
        timings = []
        for _ in range(25):
            started = time.perf_counter()
            model(inputs)
            timings.append((time.perf_counter() - started) * 1000.0)
    try:
        import resource
        import sys

        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss_bytes *= 1024
    except (ImportError, AttributeError):
        peak_rss_bytes = psutil.Process().memory_info().rss
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_size_bytes": Path(model_path).stat().st_size,
        "input_dimensions": list(input_shape),
        "cpu_inference_ms_mean": float(np.mean(timings)),
        "cpu_inference_ms_p95": float(np.percentile(timings, 95)),
        "process_peak_rss_bytes": peak_rss_bytes,
        "measurement_note": "CPU-only single-example inference; process RSS is process-wide, not model-only memory",
    }


def save_evaluation(
    output_dir: str | Path,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "Evaluation report",
        "=================",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro precision: {metrics['precision_macro']:.4f}",
        f"Macro recall: {metrics['recall_macro']:.4f}",
        f"Macro F1: {metrics['f1_macro']:.4f}",
    ]
    auc = metrics.get("roc_auc", metrics.get("roc_auc_ovr_macro"))
    lines.append(f"ROC-AUC: {auc:.4f}" if auc is not None else "ROC-AUC: not defined")
    lines.append("")
    lines.append("Per-class performance (observed)")
    for name in metrics.get("class_names", []):
        score = metrics["per_class"].get(name, {})
        lines.append(
            f"{name}: support={score.get('support', 0):.0f}, "
            f"precision={score.get('precision', 0):.4f}, "
            f"recall={score.get('recall', 0):.4f}, f1={score.get('f1-score', 0):.4f}"
        )
    confusion = np.asarray(metrics.get("confusion_matrix", []), dtype=np.int64)
    names = metrics.get("class_names", [])
    confusions = [
        (int(confusion[row, column]), names[row], names[column])
        for row in range(min(len(confusion), len(names)))
        for column in range(min(len(confusion), len(names)))
        if row != column and confusion[row, column] > 0
    ]
    if confusions:
        lines.extend(["", "Most frequent confusions (gold -> predicted)"])
        for count, gold, predicted in sorted(confusions, reverse=True)[:10]:
            lines.append(f"{gold} -> {predicted}: {count}")
    if "false_accepts" in metrics:
        lines.extend([
            "",
            "Binary wake detection",
            f"Threshold: {metrics['wake_threshold']:.4f}",
            f"False accepts: {metrics['false_accepts']}/{metrics['negative_count']} ({metrics['false_accept_rate']:.4f})",
            f"False rejects: {metrics['false_rejects']}/{metrics['positive_count']} ({metrics['false_reject_rate']:.4f})",
            f"Binary accuracy: {metrics['wake_detection_accuracy']:.4f}",
            f"Binary precision: {metrics['wake_detection_precision']:.4f}",
            f"Binary recall: {metrics['wake_detection_recall']:.4f}",
            f"Binary F1: {metrics['wake_detection_f1']:.4f}",
            f"Binary ROC-AUC: {metrics['wake_detection_roc_auc']:.4f}"
            if metrics['wake_detection_roc_auc'] is not None else "Binary ROC-AUC: not defined",
            f"Syllable-only negatives: {metrics['syllable_cues_counted_as_non_wake']}",
        ])
    if "parameter_count" in metrics:
        lines.extend([
            "",
            "Model footprint (measured)",
            f"Parameters: {metrics['parameter_count']}",
            f"Serialized model: {metrics['model_size_bytes']} bytes",
            f"Input dimensions: {metrics['input_dimensions']}",
            f"CPU inference mean/p95: {metrics['cpu_inference_ms_mean']:.3f}/{metrics['cpu_inference_ms_p95']:.3f} ms per example",
            f"Process peak RSS: {metrics['process_peak_rss_bytes']} bytes (process-wide, not model-only)",
        ])
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
