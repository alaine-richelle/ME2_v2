"""Hugging Face dataset discovery and speaker-aware VCM data access."""

from __future__ import annotations

import argparse
import io
import logging
import math
import random
import re
from collections import Counter
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, DatasetDict, load_dataset
from torch.utils.data import Dataset

from me2.audio import AudioConfig, prepare_audio

LOGGER = logging.getLogger(__name__)
DATASET_ID = "airimonda/ai231-me2-voice-commands"
REQUIRED_SPLITS = ("train", "test", "holdout")


def label_name(value: Any) -> str:
    text = str(value or "OUT_OF_SCOPE").strip().casefold()
    if text in {"", "none", "nan"}:
        return "out_of_scope"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def command_from_row(row: dict[str, Any]) -> str:
    if row.get("out_of_scope") in (1, True, "1"):
        return "out_of_scope"
    return label_name(row.get("command", "out_of_scope"))


def load_hf_splits(dataset_id: str = DATASET_ID) -> DatasetDict:
    LOGGER.info("Discovering required splits from Hugging Face dataset %s", dataset_id)
    # Load named splits individually so the default config's 66k-row numerals set is not fetched.
    splits = {}
    for name in REQUIRED_SPLITS:
        split = load_dataset(dataset_id, split=name)
        split = split.cast_column("audio", Audio(decode=False))
        splits[name] = split
    dataset = DatasetDict(splits)
    for split_name in REQUIRED_SPLITS:
        split = dataset[split_name]
        LOGGER.info("Dataset split=%s rows=%d columns=%s", split_name, len(split), split.column_names)
        if "audio" not in split.column_names or "command" not in split.column_names:
            raise ValueError(f"Unexpected {split_name} schema; expected audio and command columns")
    return dataset


def decode_audio(value: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Decode an undecoded HF audio cell without TorchCodec or FFmpeg."""
    encoded = value.get("bytes")
    path = value.get("path")
    if encoded:
        samples, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    elif path:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    else:
        raise ValueError("HF audio row has neither encoded bytes nor a file path")
    return samples, int(sample_rate)


def describe_split(dataset: DatasetDict) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in REQUIRED_SPLITS:
        split = dataset[name]
        labels = [command_from_row(row) for row in split]
        speakers = [str(value) for value in split["speaker_id"]] if "speaker_id" in split.column_names else []
        report[name] = {
            "rows": len(split),
            "classes": dict(sorted(Counter(labels).items())),
            "speaker_count": len(set(speakers)),
        }
    return report


def split_train_validation(
    train_split, validation_fraction: float = 0.15, seed: int = 42
) -> tuple[list[int], list[int]]:
    if "speaker_id" not in train_split.column_names:
        raise ValueError("Speaker-disjoint validation requires the source speaker_id column")
    missing_tokens = {"", "unknown", "none", "nan"}
    speaker_ids = [str(value or "").strip() for value in train_split["speaker_id"]]
    known_speakers = sorted({speaker for speaker in speaker_ids if speaker.casefold() not in missing_tokens})
    if len(known_speakers) < 2:
        raise ValueError("At least two identifiable training speakers are needed for a safe validation split")
    shuffled = known_speakers.copy()
    random.Random(seed).shuffle(shuffled)
    validation_speaker_count = min(
        len(known_speakers) - 1, max(1, math.ceil(len(known_speakers) * validation_fraction))
    )
    validation_speakers = set(shuffled[:validation_speaker_count])
    train_indices = [
        index for index, speaker in enumerate(speaker_ids)
        if speaker.casefold() in missing_tokens or speaker not in validation_speakers
    ]
    validation_indices = [
        index for index, speaker in enumerate(speaker_ids) if speaker in validation_speakers
    ]
    train_speaker_ids = {speaker_ids[index] for index in train_indices}
    validation_speaker_ids = {speaker_ids[index] for index in validation_indices}
    if train_speaker_ids & validation_speaker_ids:
        raise AssertionError("Speaker leakage detected between train and validation")
    LOGGER.info(
        "Speaker split train=%d rows/%d speakers validation=%d rows/%d speakers",
        len(train_indices), len({speaker_ids[index] for index in train_indices if speaker_ids[index]}),
        len(validation_indices), len(validation_speakers),
    )
    return train_indices, validation_indices


def valid_audio_indices(split, config: AudioConfig, name: str) -> list[int]:
    valid: list[int] = []
    for index in range(len(split)):
        row_id = split["file"][index] if "file" in split.column_names else index
        try:
            audio = split[index]["audio"]
            samples, sample_rate = decode_audio(audio)
            prepare_audio(samples, sample_rate, config)
            valid.append(index)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            LOGGER.warning("Skipping corrupted audio split=%s item=%s error=%s", name, row_id, error)
    LOGGER.info("Audio inspection split=%s usable=%d/%d", name, len(valid), len(split))
    return valid


class HFAudioDataset(Dataset):
    def __init__(self, split, indices: list[int], labels: dict[str, int], audio_config: AudioConfig):
        self.split = split
        self.indices = indices
        self.labels = labels
        self.audio_config = audio_config

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = self.indices[item]
        row = self.split[source_index]
        samples, sample_rate = decode_audio(row["audio"])
        waveform = prepare_audio(samples, sample_rate, self.audio_config)
        name = command_from_row(row)
        if name not in self.labels:
            raise ValueError(f"Unmapped VCM command class {name!r}")
        return torch.from_numpy(waveform), torch.tensor(self.labels[name], dtype=torch.long)


def label_mapping(train_split) -> dict[str, int]:
    if "out_of_scope" in train_split.column_names:
        labels = sorted({
            command_from_row({"command": command, "out_of_scope": out_of_scope})
            for command, out_of_scope in zip(train_split["command"], train_split["out_of_scope"])
        })
    else:
        labels = sorted({label_name(command) for command in train_split["command"]})
    if "out_of_scope" not in labels:
        labels.append("out_of_scope")
    if len(labels) > 20:
        raise ValueError(f"Expected at most 19 commands plus OUT_OF_SCOPE, found {len(labels)} classes")
    mapping = {name: index for index, name in enumerate(labels)}
    LOGGER.info("VCM labels (%d): %s", len(mapping), ", ".join(mapping))
    return mapping


def validate_official_speakers(dataset: DatasetDict) -> None:
    speaker_sets = {
        split_name: set(map(str, dataset[split_name]["speaker_id"]))
        for split_name in REQUIRED_SPLITS
    }
    for left, right in (("train", "test"), ("train", "holdout"), ("test", "holdout")):
        overlap = speaker_sets[left] & speaker_sets[right]
        if overlap:
            raise ValueError(f"Official dataset speaker leakage between {left} and {right}: {len(overlap)} IDs")
    LOGGER.info("Verified official train/test/holdout speaker sets are disjoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Download and summarize official splits")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = load_hf_splits(args.dataset_id)
    print(__import__("json").dumps(describe_split(dataset), indent=2))


if __name__ == "__main__":
    main()
