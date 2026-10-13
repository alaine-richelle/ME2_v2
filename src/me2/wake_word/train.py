"""Train compact wake baselines on frozen pronunciation-aware partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from me2.audio import AudioConfig, load_wav
from me2.bundle import save_bundle
from me2.features import FeatureConfig
from me2.model import TinyAudioCNN, parameter_count
from me2.training import fit_classifier, seed_everything
from me2.wake_word.data import (
    CANONICAL_SYNTHESIS_TEXT,
    CANONICAL_WAKE_LABEL,
    CLASS_MAPPING_VERSION,
    CLASSES,
    PRONUNCIATION_GUIDE,
    SUPPORTED_DATASET_VERSIONS,
    SYNTHESIS_RECIPE_VERSION,
    audit_manifest,
    read_manifest,
    trainable_rows,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def compatible_dataset_version(rows: list[dict[str, str]]) -> str:
    observed_versions = {row["dataset_version"] for row in rows}
    if len(observed_versions) != 1 or not observed_versions <= SUPPORTED_DATASET_VERSIONS:
        raise ValueError(
            "Wake training requires exactly one supported connected-a-teh-ko "
            f"dataset version; got {sorted(observed_versions)}"
        )
    return next(iter(observed_versions))


def effective_label(row: dict[str, str], label_mode: str) -> str:
    if label_mode == "binary":
        return (
            CANONICAL_WAKE_LABEL
            if row["label"] == CANONICAL_WAKE_LABEL
            else "non_wake"
        )
    if label_mode == "five_class":
        return row["label"]
    raise ValueError(f"Unsupported wake label mode {label_mode!r}")


class WakeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        audio_config: AudioConfig,
        labels: tuple[str, ...],
        *,
        label_mode: str = "five_class",
    ) -> None:
        self.rows = rows
        self.audio_config = audio_config
        self.labels = labels
        self.label_mode = label_mode
        self.mapping = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        duration = float(row.get("duration_seconds") or 0.0)
        if (
            row["label"] == CANONICAL_WAKE_LABEL
            and duration > self.audio_config.duration_seconds
        ):
            raise ValueError(
                f"Complete-word wake clip {row['recording_id']} is {duration:.3f}s, "
                f"longer than the {self.audio_config.duration_seconds:.3f}s window"
            )
        waveform = load_wav(row["path"], self.audio_config)
        label = effective_label(row, self.label_mode)
        return (
            torch.from_numpy(waveform),
            torch.tensor(self.mapping[label], dtype=torch.long),
        )


def _bounded_rows(
    rows: list[dict[str, str]],
    maximum_per_speaker_kind: int,
) -> list[dict[str, str]]:
    if maximum_per_speaker_kind <= 0:
        return rows
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["speaker_id"], row["kind"], row["label"])].append(row)
    selected: list[dict[str, str]] = []
    for key in sorted(grouped):
        selected.extend(
            sorted(grouped[key], key=lambda row: row["recording_id"])[
                :maximum_per_speaker_kind
            ]
        )
    return selected


def _balanced_subset(
    rows: list[dict[str, str]],
    limit: int,
    label_mode: str,
) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(effective_label(row, label_mode), row["speaker_id"])].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected


def source_aware_sample_weights(
    rows: list[dict[str, str]],
    label_mode: str,
) -> list[float]:
    label_counts = Counter(effective_label(row, label_mode) for row in rows)
    group_counts = Counter(
        (
            effective_label(row, label_mode),
            row["speaker_id"],
            row["source"],
            row["kind"],
        )
        for row in rows
    )
    maximum_group = {
        label: max(
            count
            for (group_label, *_), count in group_counts.items()
            if group_label == label
        )
        for label in label_counts
    }
    raw = []
    for row in rows:
        label = effective_label(row, label_mode)
        group = (
            label,
            row["speaker_id"],
            row["source"],
            row["kind"],
        )
        diversity_factor = min(
            2.0,
            math.sqrt(maximum_group[label] / group_counts[group]),
        )
        raw.append(diversity_factor / label_counts[label])
    mean = float(np.mean(raw))
    return [weight / mean for weight in raw]


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_augmentation_preserves_wake(
    rows: list[dict[str, str]],
    audio_config: AudioConfig,
    augmentation: dict[str, object],
) -> None:
    """Reject settings that could shift or stretch a full keyword out of frame."""

    if not augmentation.get("enabled", True):
        return
    speed_min = float(augmentation.get("speed_min", 1.0))
    shift_seconds = float(augmentation.get("max_shift_ms", 0.0)) / 1000.0
    if speed_min <= 0:
        raise ValueError("Wake augmentation speed_min must be positive")
    safe_keyword_seconds = (
        audio_config.duration_seconds - 2.0 * shift_seconds
    ) * speed_min
    too_long = [
        row
        for row in rows
        if row["label"] == CANONICAL_WAKE_LABEL
        and float(row.get("duration_seconds") or 0.0) > safe_keyword_seconds
    ]
    if too_long:
        example = too_long[0]
        raise ValueError(
            "Wake augmentation could crop a complete keyword after time/shift "
            f"transform: {example['recording_id']} duration="
            f"{float(example['duration_seconds']):.3f}s, safe maximum="
            f"{safe_keyword_seconds:.3f}s. Increase the input window or reduce "
            "speed/shift augmentation."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "wake-word/data/v3/synthetic/manifest.csv",
    )
    parser.add_argument("--label-mode", choices=("five_class", "binary"), default="five_class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-per-speaker-kind", type=int, default=24)
    parser.add_argument("--allow-pending-review", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--disable-source-aware-sampling", action="store_true")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.width < 4:
        parser.error("--width must be at least 4")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    run_name = args.run_name or args.label_mode
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_name):
        parser.error("--run-name may contain only letters, digits, hyphens, and underscores")

    audit = audit_manifest(args.manifest)
    if audit["issues"] or audit["speaker_split_leakage"] or audit["parent_split_leakage"]:
        raise ValueError("Wake manifest failed integrity/leakage audit")
    if audit["cross_split_exact_duplicates"]:
        raise ValueError("Wake manifest has exact duplicate audio across splits")
    all_rows = read_manifest(args.manifest)
    rows = trainable_rows(
        all_rows,
        allow_pending_review=args.allow_pending_review,
    )
    dataset_version = compatible_dataset_version(rows)

    partitions = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "calibration", "test")
    }
    for split, split_rows in partitions.items():
        if not split_rows:
            raise ValueError(f"Wake split {split!r} is empty after review filtering")
    partitions["train"] = _bounded_rows(
        partitions["train"],
        args.max_per_speaker_kind,
    )
    if args.smoke_limit:
        for split, split_rows in partitions.items():
            partitions[split] = _balanced_subset(
                split_rows,
                max(8, args.smoke_limit if split == "train" else args.smoke_limit // 3),
                args.label_mode,
            )

    labels = (
        ("non_wake", CANONICAL_WAKE_LABEL)
        if args.label_mode == "binary"
        else tuple(CLASSES)
    )
    for split, split_rows in partitions.items():
        observed = {effective_label(row, args.label_mode) for row in split_rows}
        missing = set(labels) - observed
        if missing:
            raise ValueError(f"Wake split {split!r} lacks classes {sorted(missing)}")

    config = json.loads((ROOT / "config/default.json").read_text(encoding="utf-8"))
    audio_config = AudioConfig(
        sample_rate=int(config["wake_word"]["sample_rate"]),
        duration_seconds=float(config["wake_word"]["duration_seconds"]),
        silence_threshold_db=float(config["audio"]["silence_threshold_db"]),
        normalize_peak=bool(config["audio"]["normalize_peak"]),
    )
    feature_config = FeatureConfig(
        **{
            key: config["audio"][key]
            for key in FeatureConfig.__dataclass_fields__
        }
    )
    wake_augmentation = (
        {"enabled": False}
        if args.disable_augmentation
        else config["wake_augmentation"]
    )
    validate_augmentation_preserves_wake(
        partitions["train"],
        audio_config,
        wake_augmentation,
    )
    seed_everything(args.seed)
    model = TinyAudioCNN(
        num_classes=len(labels),
        width=args.width,
        dropout=args.dropout,
    )
    train_data = WakeDataset(
        partitions["train"],
        audio_config,
        labels,
        label_mode=args.label_mode,
    )
    log_path = (
        args.log_path
        or ROOT / f"logs/wake_v3_{run_name}_training.jsonl"
    )
    fit = fit_classifier(
        model,
        train_data,
        WakeDataset(
            partitions["validation"],
            audio_config,
            labels,
            label_mode=args.label_mode,
        ),
        feature_config,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=int(config["training"]["patience"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=args.weight_decay,
        num_workers=0,
        log_path=log_path,
        augmentation=wake_augmentation,
        sample_weights=(
            None
            if args.disable_source_aware_sampling
            else source_aware_sample_weights(
                partitions["train"],
                args.label_mode,
            )
        ),
        track_clean_train=True,
    )
    output = args.output or ROOT / (
        f"wake-word/models/wake_word_model_v3_{args.label_mode}.me2"
    )
    pronunciation_review_complete = all(
        row["review_status"] == "accepted" for row in rows
    )
    bundle = {
        "task": "wake_word",
        "architecture": "TinyAudioCNN",
        "model_version": (
            f"wake-tiny-cnn-connected-a-teh-ko-{dataset_version}-{args.label_mode}"
        ),
        "experiment_id": run_name,
        "model_width": args.width,
        "dropout": args.dropout,
        "state_dict": fit["state_dict"],
        "labels": list(labels),
        "label_mode": args.label_mode,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "pronunciation_guide": PRONUNCIATION_GUIDE,
        "canonical_synthesis_text": CANONICAL_SYNTHESIS_TEXT,
        "synthesis_recipe_version": SYNTHESIS_RECIPE_VERSION,
        "wake_dataset_version": dataset_version,
        "wake_manifest_sha256": _manifest_hash(args.manifest),
        "pronunciation_review_complete": pronunciation_review_complete,
        "deployment_ready": pronunciation_review_complete and audit["real_speakers"] > 0,
        "audio": audio_config.__dict__,
        "features": feature_config.__dict__,
        "input_shape": [
            1,
            feature_config.n_mels,
            round(audio_config.sample_count / feature_config.hop_length) + 1,
        ],
        "seed": args.seed,
        "threshold": float(config["wake_word"]["positive_threshold"]),
        "max_false_accept_rate": float(
            config["wake_word"]["max_false_accept_rate"]
        ),
        "streaming_policy": {
            "smoothing_window": int(config["wake_word"]["smoothing_window"]),
            "required_hits": int(config["wake_word"]["required_hits"]),
            "wake_hop_ms": int(config["streaming"]["wake_hop_ms"]),
            "cooldown_ms": int(config["streaming"]["cooldown_ms"]),
            "calibration_mode": "clip_proxy_pending_continuous_validation",
        },
        "dataset_counts": {
            split: dict(
                Counter(
                    effective_label(row, args.label_mode)
                    for row in split_rows
                )
            )
            for split, split_rows in partitions.items()
        },
        "speaker_counts": {
            split: len({row["speaker_id"] for row in split_rows})
            for split, split_rows in partitions.items()
        },
        "best_validation_loss": fit["best_validation_loss"],
        "training_run": {
            "run_name": run_name,
            "epochs_requested": args.epochs,
            "epochs_completed": fit["epochs_completed"],
            "best_epoch": fit["best_epoch"],
            "stopping_reason": fit["stopping_reason"],
            "elapsed_seconds": fit["elapsed_seconds"],
            "history_path": str(log_path),
            "command": shlex.join(
                [sys.executable, "-m", "me2.wake_word.train", *sys.argv[1:]]
            ),
        },
        "parameter_count": parameter_count(model),
        "training_source": sorted({row["source"] for row in rows}),
        "training_assumptions": {
            "allow_pending_review": args.allow_pending_review,
            "synthetic_only": audit["real_speakers"] == 0,
            "controlled_augmentation": not args.disable_augmentation,
            "source_aware_sampling": not args.disable_source_aware_sampling,
        },
    }
    save_bundle(output, bundle)
    LOGGER.info("Saved wake candidate %s (%d bytes)", output, output.stat().st_size)

    from me2.wake_word.calibrate import calibrate_artifact

    calibrate_artifact(
        output,
        args.manifest,
        split="calibration",
        target_false_accept_rate=float(
            config["wake_word"]["max_false_accept_rate"]
        ),
    )
    if not args.skip_test_evaluation:
        from me2.wake_word.evaluate import evaluate

        evaluate(
            output,
            args.manifest,
            "test",
            args.test_output
            or ROOT / f"results/wake_v3_{args.label_mode}_test",
        )


if __name__ == "__main__":
    main()
