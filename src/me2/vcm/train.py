"""Train the compact intent classifier from the required HF dataset."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from me2.audio import AudioConfig
from me2.bundle import save_bundle
from me2.features import FeatureConfig
from me2.model import TinyAudioCNN, parameter_count
from me2.training import fit_classifier
from me2.vcm.data import (
    HFAudioDataset,
    command_from_row,
    label_mapping,
    load_hf_splits,
    split_train_validation,
    valid_audio_indices,
    validate_official_speakers,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="airimonda/ai231-me2-voice-commands")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    parser.add_argument("--output", type=Path, default=ROOT / "vcm/models/vcm_model.pkl")
    parser.add_argument("--smoke-limit", type=int, default=0, help="Limit per training/validation subset for pipeline smoke tests")
    parser.add_argument("--skip-test-evaluation", action="store_true", help="Skip final held-out test scoring for smoke runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audio_config = AudioConfig(**{key: config["audio"][key] for key in AudioConfig.__dataclass_fields__})
    feature_config = FeatureConfig(**{key: config["audio"][key] for key in FeatureConfig.__dataclass_fields__})
    dataset = load_hf_splits(args.dataset_id)
    validate_official_speakers(dataset)
    label_to_id = label_mapping(dataset["train"])
    train_indices, validation_indices = split_train_validation(
        dataset["train"], args.validation_fraction, args.seed
    )
    valid_indices = set(valid_audio_indices(dataset["train"], audio_config, "train"))
    train_indices = [index for index in train_indices if index in valid_indices]
    validation_indices = [index for index in validation_indices if index in valid_indices]
    if args.smoke_limit:
        rng = random.Random(args.seed)

        def balanced_subset(indices: list[int], limit: int) -> list[int]:
            groups: dict[str, list[int]] = {}
            for index in indices:
                name = command_from_row(dataset["train"][index])
                groups.setdefault(name, []).append(index)
            for group in groups.values():
                rng.shuffle(group)
            selected: list[int] = []
            while len(selected) < limit and any(groups.values()):
                for name in sorted(groups):
                    if groups[name] and len(selected) < limit:
                        selected.append(groups[name].pop())
            return selected

        train_indices = balanced_subset(train_indices, args.smoke_limit)
        validation_indices = balanced_subset(validation_indices, max(20, args.smoke_limit // 4))
    train_data = HFAudioDataset(dataset["train"], train_indices, label_to_id, audio_config)
    validation_data = HFAudioDataset(dataset["train"], validation_indices, label_to_id, audio_config)
    train_counts = Counter(
        command_from_row(dataset["train"][index]) for index in train_indices
    )
    class_weights = [
        len(train_indices) / (len(label_to_id) * max(train_counts.get(name, 0), 1))
        for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])
    ]
    LOGGER.info("VCM training counts=%s class_weights=%s", dict(train_counts), class_weights)
    model = TinyAudioCNN(num_classes=len(label_to_id))
    LOGGER.info("VCM model parameters=%d", parameter_count(model))
    fit = fit_classifier(
        model,
        train_data,
        validation_data,
        feature_config,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=int(config["training"]["patience"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        num_workers=int(config["training"]["num_workers"]),
        log_path=ROOT / "logs/vcm_training.jsonl",
        augmentation=config["augmentation"],
        class_weights=class_weights,
    )
    bundle = {
        "task": "vcm",
        "dataset_id": args.dataset_id,
        "architecture": "TinyAudioCNN",
        "model_width": 16,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "labels": [name for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
        "audio": audio_config.__dict__,
        "features": feature_config.__dict__,
        "input_shape": [1, feature_config.n_mels, round(audio_config.sample_count / feature_config.hop_length) + 1],
        "seed": args.seed,
        "best_validation_loss": fit["best_validation_loss"],
        "parameter_count": parameter_count(model),
    }
    save_bundle(args.output, bundle)
    LOGGER.info("Saved canonical VCM model %s (%d bytes)", args.output, args.output.stat().st_size)
    from me2.vcm.evaluate import evaluate

    if not args.skip_test_evaluation:
        evaluate(args.output, "test", ROOT / "results/vcm_test", args.dataset_id)


if __name__ == "__main__":
    main()
