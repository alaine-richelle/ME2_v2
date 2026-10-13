"""Train the compact intent classifier from the required HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shlex
import sys
from collections import Counter
from pathlib import Path

import torch

from me2.audio import AudioConfig
from me2.bundle import save_bundle
from me2.features import FeatureConfig
from me2.model import TinyAudioCNN, parameter_count
from me2.training import fit_classifier, seed_everything
from me2.vcm.ctc import (
    BLANK_ID,
    CHARSET,
    DECODER_VERSION,
    GRAMMAR_SCHEMA_VERSION,
    TEXT_NORMALIZER_VERSION,
    CompactCTCModel,
)
from me2.vcm.ctc_training import (
    HFCtcAudioDataset,
    calibrate_acceptance_policy,
    fit_ctc_classifier,
    training_grammar,
    valid_ctc_text_indices,
)
from me2.vcm.data import (
    DATASET_REVISION,
    HFAudioDataset,
    audit_audio_indices,
    command_from_row,
    label_mapping,
    load_hf_splits,
    save_partition_manifest,
    split_train_validation,
    split_train_validation_calibration,
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
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument(
        "--architecture",
        choices=("TinyAudioCNN", "CompactCTCModel"),
        default="CompactCTCModel",
    )
    parser.add_argument("--beam-width", type=int, default=16, help="CTC grammar decoder beam width")
    parser.add_argument("--target-oos-far", type=float, default=0.05)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--partition-output", type=Path)
    parser.add_argument("--quarantine-output", type=Path)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument(
        "--smoke-limit",
        "--development-limit",
        dest="smoke_limit",
        type=int,
        default=0,
        help=(
            "Balanced limit for train, validation, and calibration partitions; "
            "nonzero runs are preliminary development experiments"
        ),
    )
    parser.add_argument("--skip-test-evaluation", action="store_true", help="Skip final held-out test scoring for smoke runs")
    args = parser.parse_args()
    default_name = (
        "vcm_compact_ctc_model.me2"
        if args.architecture == "CompactCTCModel"
        else "vcm_baseline_model.me2"
    )
    output_path = args.output or ROOT / "vcm/models" / default_name
    log_path = args.log_path or ROOT / "logs" / f"vcm_{args.run_name}_training.jsonl"
    partition_output = (
        args.partition_output
        or ROOT / "results" / f"vcm_{args.run_name}_partitions.json"
    )
    quarantine_output = (
        args.quarantine_output
        or ROOT / "results" / f"vcm_{args.run_name}_quarantine.json"
    )
    test_output = (
        args.test_output
        or ROOT / "results" / f"vcm_{args.run_name}_test"
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audio_config = AudioConfig(**{key: config["audio"][key] for key in AudioConfig.__dataclass_fields__})
    feature_config = FeatureConfig(**{key: config["audio"][key] for key in FeatureConfig.__dataclass_fields__})
    dataset = load_hf_splits(args.dataset_id)
    validate_official_speakers(dataset)
    label_to_id = label_mapping(dataset["train"])
    if args.architecture == "CompactCTCModel":
        partitions = split_train_validation_calibration(
            dataset["train"],
            args.validation_fraction,
            args.calibration_fraction,
            args.seed,
        )
        save_partition_manifest(
            partition_output,
            dataset["train"],
            partitions,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            calibration_fraction=args.calibration_fraction,
        )
        train_indices = partitions["train"]
        validation_indices = partitions["validation"]
        calibration_indices = partitions["calibration"]
        quarantine = [
            {
                "index": index,
                "file": str(dataset["train"][index].get("file") or ""),
                "stage": "partition",
                "reason": "missing_speaker_identity_empty_nonspeech",
            }
            for index in partitions["quarantine"]
        ]
    else:
        train_indices, validation_indices = split_train_validation(
            dataset["train"], args.validation_fraction, args.seed
        )
        calibration_indices = []
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
        if calibration_indices:
            calibration_indices = balanced_subset(
                calibration_indices, max(20, args.smoke_limit // 4)
            )

    candidates = sorted(
        set(train_indices) | set(validation_indices) | set(calibration_indices)
    )
    valid_audio, audio_issues = audit_audio_indices(
        dataset["train"],
        audio_config,
        "train",
        candidates,
        sequence_mode=args.architecture == "CompactCTCModel",
    )
    if candidates and len(audio_issues) / len(candidates) > 0.05:
        raise ValueError(
            f"Audio quarantine rate {len(audio_issues)}/{len(candidates)} exceeds 5%"
        )
    valid_indices = set(valid_audio)
    train_indices = [index for index in train_indices if index in valid_indices]
    validation_indices = [
        index for index in validation_indices if index in valid_indices
    ]
    calibration_indices = [
        index for index in calibration_indices if index in valid_indices
    ]
    if args.architecture == "CompactCTCModel":
        quarantine.extend(
            {**issue, "stage": "audio"} for issue in audio_issues
        )
        train_indices, train_quarantine = valid_ctc_text_indices(
            dataset["train"], train_indices
        )
        validation_indices, validation_quarantine = valid_ctc_text_indices(
            dataset["train"], validation_indices
        )
        quarantine.extend(
            {**issue, "stage": "target", "partition": "train"}
            for issue in train_quarantine
        )
        quarantine.extend(
            {**issue, "stage": "target", "partition": "validation"}
            for issue in validation_quarantine
        )
        if not train_indices or not validation_indices or not calibration_indices:
            raise ValueError("CTC train, validation, and calibration partitions must be non-empty")
        observed_train_labels = {
            command_from_row(dataset["train"][index]) for index in train_indices
        }
        missing_train_labels = sorted(set(label_to_id) - observed_train_labels)
        if missing_train_labels:
            raise ValueError(
                f"CTC training lost label coverage after quarantine: {missing_train_labels}"
            )
        quarantine_path = quarantine_output
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.write_text(
            json.dumps(quarantine, indent=2) + "\n", encoding="utf-8"
        )

    if args.architecture == "CompactCTCModel":
        if args.beam_width < 1:
            parser.error("--beam-width must be positive")
        train_data = HFCtcAudioDataset(dataset["train"], train_indices, label_to_id, audio_config)
        validation_data = HFCtcAudioDataset(
            dataset["train"], validation_indices, label_to_id, audio_config
        )
        grammar = training_grammar(dataset["train"], train_indices)
        seed_everything(args.seed)
        model = CompactCTCModel(n_mels=feature_config.n_mels)
        model_parameters = parameter_count(model)
        LOGGER.info(
            "VCM compact CTC parameters=%d grammar_transcripts=%d beam_width=%d",
            model_parameters,
            len(grammar),
            args.beam_width,
        )
        fit = fit_ctc_classifier(
            model,
            train_data,
            validation_data,
            feature_config,
            grammar,
            [name for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=int(config["training"]["patience"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            num_workers=int(config["training"]["num_workers"]),
            log_path=log_path,
            augmentation=config["augmentation"],
            beam_width=args.beam_width,
        )
        model.load_state_dict(fit["state_dict"])
        device = torch.device(fit["device"])
        model.to(device)
        calibration_data = HFCtcAudioDataset(
            dataset["train"],
            calibration_indices,
            label_to_id,
            audio_config,
            allow_empty_targets=True,
        )
        acceptance_policy, calibration_summary = calibrate_acceptance_policy(
            model,
            calibration_data,
            feature_config,
            grammar,
            [
                name
                for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])
            ],
            beam_width=args.beam_width,
            batch_size=args.batch_size,
            target_false_accept_rate=args.target_oos_far,
        )
        grammar_payload = json.dumps(
            grammar, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        bundle = {
            "task": "vcm",
            "dataset_id": args.dataset_id,
            "dataset_revision": DATASET_REVISION,
            "architecture": "CompactCTCModel",
            "model_config": {
                "width": 48,
                "projection_size": 128,
                "hidden_size": 96,
            },
            "model_version": "compact-ctc-v1",
            "state_dict": fit["state_dict"],
            "labels": [name for name, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
            "grammar": grammar,
            "grammar_schema_version": GRAMMAR_SCHEMA_VERSION,
            "grammar_version": hashlib.sha256(grammar_payload).hexdigest(),
            "beam_width": args.beam_width,
            "decoder_version": DECODER_VERSION,
            "charset": CHARSET,
            "blank_id": BLANK_ID,
            "text_normalizer_version": TEXT_NORMALIZER_VERSION,
            "acceptance_policy": acceptance_policy.__dict__,
            "calibration": calibration_summary,
            "audio": audio_config.__dict__,
            "features": feature_config.__dict__,
            "input_shape": [
                1,
                feature_config.n_mels,
                round(audio_config.sample_count / feature_config.hop_length) + 1,
            ],
            "seed": args.seed,
            "best_validation_macro_f1": fit["best_validation_macro_f1"],
            "training_run": {
                "run_name": args.run_name,
                "scope": (
                    "preliminary_balanced_subset"
                    if args.smoke_limit
                    else "full_permitted_training_partition"
                ),
                "development_limit": args.smoke_limit,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "calibration_rows": len(calibration_indices),
                "epochs_requested": args.epochs,
                "epochs_completed": fit["epochs_completed"],
                "best_epoch": fit["best_epoch"],
                "stopping_reason": fit["stopping_reason"],
                "elapsed_seconds": fit["elapsed_seconds"],
                "history_path": str(log_path),
                "partition_path": str(partition_output),
                "quarantine_path": str(quarantine_output),
                "command": shlex.join([sys.executable, "-m", "me2.vcm.train", *sys.argv[1:]]),
            },
            "parameter_count": model_parameters,
            "provenance": {
                "integration_revision": "52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24",
                "ctc_reference_revision": "f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4",
                "dataset_revision": DATASET_REVISION,
            },
        }
        save_bundle(output_path, bundle)
        LOGGER.info("Saved CTC VCM model %s (%d bytes)", output_path, output_path.stat().st_size)
        if not args.skip_test_evaluation:
            from me2.vcm.evaluate import evaluate

            evaluate(
                output_path,
                "test",
                test_output,
                args.dataset_id,
            )
        return

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
    seed_everything(args.seed)
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
        "dataset_revision": DATASET_REVISION,
        "architecture": "TinyAudioCNN",
        "model_version": "intent-cnn-baseline-v1",
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
    save_bundle(output_path, bundle)
    LOGGER.info("Saved canonical VCM model %s (%d bytes)", output_path, output_path.stat().st_size)
    from me2.vcm.evaluate import evaluate

    if not args.skip_test_evaluation:
        evaluate(output_path, "test", ROOT / "results/vcm_test", args.dataset_id)


if __name__ == "__main__":
    main()
