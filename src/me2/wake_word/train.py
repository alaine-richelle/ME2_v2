"""Train the syllable-aware wake classifier using speaker-disjoint splits."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from me2.audio import AudioConfig, load_wav
from me2.bundle import save_bundle
from me2.features import FeatureConfig
from me2.model import TinyAudioCNN, parameter_count
from me2.training import fit_classifier
from me2.wake_word.data import CLASSES, read_manifest

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def assign_speaker_splits(records: list[dict[str, str]], seed: int) -> None:
    speakers = sorted({row["speaker_id"] for row in records})
    if len(speakers) < 3:
        raise ValueError("Wake training requires at least three speakers/voices for train/validation/test")
    random.Random(seed).shuffle(speakers)
    train_end = max(1, int(0.65 * len(speakers)))
    validation_end = max(train_end + 1, int(0.82 * len(speakers)))
    split_by_speaker = {
        speaker: ("train" if index < train_end else "validation" if index < validation_end else "test")
        for index, speaker in enumerate(speakers)
    }
    for row in records:
        row["split"] = split_by_speaker[row["speaker_id"]]


class WakeDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], audio_config: AudioConfig, labels: tuple[str, ...]):
        self.rows = rows
        self.audio_config = audio_config
        self.labels = labels
        self.mapping = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        waveform = load_wav(row["path"], self.audio_config)
        return torch.from_numpy(waveform), torch.tensor(self.mapping[row["label"]], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "wake-word/data/generated/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "wake-word/models/wake_word_model.pkl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = read_manifest(args.manifest)
    assign_speaker_splits(rows, args.seed)
    classes = tuple(CLASSES)
    counts = {split: {label: sum(row["split"] == split and row["label"] == label for row in rows) for label in classes}
              for split in ("train", "validation", "test")}
    LOGGER.info("Wake dataset label counts by speaker split: %s", counts)
    config = json.loads((ROOT / "config/default.json").read_text(encoding="utf-8"))
    audio_config = AudioConfig(
        sample_rate=int(config["wake_word"]["sample_rate"]),
        duration_seconds=float(config["wake_word"]["duration_seconds"]),
        silence_threshold_db=float(config["audio"]["silence_threshold_db"]),
        normalize_peak=bool(config["audio"]["normalize_peak"]),
    )
    feature_config = FeatureConfig(**{
        key: config["audio"][key] for key in FeatureConfig.__dataclass_fields__
    })
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("Generated corpus must have non-empty train and validation splits")
    model = TinyAudioCNN(num_classes=len(classes), width=12)
    train_counts = np.bincount(
        [classes.index(row["label"]) for row in train_rows], minlength=len(classes)
    )
    class_weights = (len(train_rows) / (len(classes) * np.maximum(train_counts, 1))).tolist()
    LOGGER.info("Wake loss weights by class: %s", dict(zip(classes, class_weights)))
    fit = fit_classifier(
        model,
        WakeDataset(train_rows, audio_config, classes),
        WakeDataset(validation_rows, audio_config, classes),
        feature_config,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=int(config["training"]["patience"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        num_workers=0,
        log_path=ROOT / "logs/wake_training.jsonl",
        augmentation=config["augmentation"],
        class_weights=class_weights,
    )
    bundle = {
        "task": "wake_word",
        "architecture": "TinyAudioCNN",
        "model_width": 12,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "labels": list(classes),
        "audio": audio_config.__dict__,
        "features": feature_config.__dict__,
        "input_shape": [1, feature_config.n_mels, round(audio_config.sample_count / feature_config.hop_length) + 1],
        "seed": args.seed,
        "threshold": float(config["wake_word"]["positive_threshold"]),
        "max_false_accept_rate": float(config["wake_word"]["max_false_accept_rate"]),
        "dataset_counts": counts,
        "best_validation_loss": fit["best_validation_loss"],
        "parameter_count": parameter_count(model),
        "training_source": sorted({row["source"] for row in rows}),
    }
    save_bundle(args.output, bundle)
    LOGGER.info("Saved canonical wake model %s (%d bytes)", args.output, args.output.stat().st_size)
    from me2.wake_word.evaluate import evaluate

    evaluate(args.output, args.manifest, "validation", ROOT / "results/wake_validation")
    evaluate(args.output, args.manifest, "test", ROOT / "results/wake_test")


if __name__ == "__main__":
    main()
