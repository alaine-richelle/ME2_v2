"""Held-out VCM evaluation and per-record error analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from me2.audio import AudioConfig
from me2.bundle import load_bundle
from me2.features import FeatureConfig, LogMel
from me2.metrics import classification_metrics, profile_model, save_evaluation
from me2.vcm.data import HFAudioDataset, command_from_row, load_hf_splits, valid_audio_indices

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def evaluate(bundle_path: Path, split_name: str, output_dir: Path, dataset_id: str) -> dict:
    model, bundle = load_bundle(bundle_path)
    dataset = load_hf_splits(dataset_id)
    if split_name not in dataset:
        raise ValueError(f"Unknown evaluation split {split_name!r}")
    labels = {name: index for index, name in enumerate(bundle["labels"])}
    audio_config = AudioConfig(**bundle["audio"])
    feature_config = FeatureConfig(**bundle["features"])
    indices = valid_audio_indices(dataset[split_name], audio_config, split_name)
    eval_data = HFAudioDataset(dataset[split_name], indices, labels, audio_config)
    loader = DataLoader(eval_data, batch_size=64, shuffle=False, num_workers=0)
    frontend = LogMel(feature_config)
    all_gold: list[int] = []
    all_pred: list[int] = []
    all_probabilities: list[list[float]] = []
    metadata: list[dict] = []
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for waveforms, targets in loader:
            probabilities = torch.softmax(model(frontend(waveforms)), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            all_gold.extend(targets.numpy().tolist())
            all_pred.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())
            for local_index in range(len(targets)):
                row = dataset[split_name][indices[cursor]]
                gold = command_from_row(row)
                metadata.append({
                    "input_audio": str(row.get("file") or row.get("audio", {}).get("path") or indices[cursor]),
                    "gold_command": gold,
                    "predicted_command": bundle["labels"][int(predictions[local_index])],
                    "confidence": float(probabilities[local_index].max()),
                    "correct": gold == bundle["labels"][int(predictions[local_index])],
                    "speaker_id": str(row.get("speaker_id", "")),
                    "source": str(row.get("source", "")),
                })
                cursor += 1
    metrics = classification_metrics(all_gold, all_pred, np.asarray(all_probabilities), bundle["labels"])
    oos_id = labels.get("out_of_scope")
    if oos_id is not None:
        gold = np.asarray(all_gold)
        predicted = np.asarray(all_pred)
        metrics["out_of_scope_false_accepts"] = int(((gold == oos_id) & (predicted != oos_id)).sum())
        metrics["out_of_scope_count"] = int((gold == oos_id).sum())
        metrics["out_of_scope_false_accept_rate"] = (
            metrics["out_of_scope_false_accepts"] / metrics["out_of_scope_count"]
            if metrics["out_of_scope_count"] else None
        )
        command_mask = gold != oos_id
        metrics["in_scope_false_rejects"] = int((command_mask & (predicted == oos_id)).sum())
        metrics["in_scope_count"] = int(command_mask.sum())
    metrics["split"] = split_name
    metrics["rows_evaluated"] = len(all_gold)
    metrics.update(profile_model(model, bundle["input_shape"], bundle_path))
    save_evaluation(output_dir, metrics, metadata)
    LOGGER.info("VCM %s metrics: %s", split_name, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "vcm/models/vcm_model.pkl")
    parser.add_argument("--split", choices=("test", "holdout"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "results/vcm_test")
    parser.add_argument("--dataset-id", default="airimonda/ai231-me2-voice-commands")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(args.model, args.split, args.output, args.dataset_id)


if __name__ == "__main__":
    main()
