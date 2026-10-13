"""Versioned wake-word corpus generation, review, splitting, and audit tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from me2.audio import resample_audio, write_wav

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]

CANONICAL_WAKE_LABEL = "ateko"
PRONUNCIATION_GUIDE = "a-teh-ko"
CANONICAL_SYNTHESIS_TEXT = "a teh ko"
SYNTHESIS_RECIPE_VERSION = "connected-a-teh-ko-v1"
DATASET_VERSION = "wake-v3-connected-a-teh-ko"
EXPANDED_DATASET_VERSION = "wake-v4-connected-a-teh-ko-expanded"
COMMAND_NEGATIVE_DATASET_VERSION = (
    "wake-v5-connected-a-teh-ko-command-negatives"
)
LEGACY_DATASET_VERSION = "wake-v1-historical-unverified"
SUPPORTED_DATASET_VERSIONS = frozenset(
    {
        DATASET_VERSION,
        EXPANDED_DATASET_VERSION,
        COMMAND_NEGATIVE_DATASET_VERSION,
    }
)
CLASS_MAPPING_VERSION = "wake-five-class-te-means-teh-v2"
CLASSES = ("background", "a", "te", "ko", CANONICAL_WAKE_LABEL)
REVIEW_STATUSES = (
    "accepted",
    "pending",
    "incorrect",
    "ambiguous",
    "technical_invalid",
)
CONNECTED_REVIEW_STATUSES = (*REVIEW_STATUSES, "not_applicable")
SPLITS = ("train", "validation", "calibration", "test", "unassigned")
TTS_VOICE_CANDIDATES = (
    "Samantha",
    "Daniel",
    "Karen",
    "Moira",
    "Tessa",
    "Albert",
    "Kathy",
    "Rishi",
    "Tara",
    "Fred",
    "Ralph",
)

MANIFEST_FIELDS = (
    "path",
    "recording_id",
    "parent_recording_id",
    "label",
    "canonical_label",
    "pronunciation_guide",
    "connected_delivery_status",
    "phrase",
    "synthesis_text",
    "speaker_id",
    "session_id",
    "device",
    "room",
    "split",
    "source",
    "source_dataset_id",
    "source_dataset_revision",
    "source_record_id",
    "source_license",
    "is_synthetic",
    "kind",
    "review_status",
    "review_notes",
    "quality_flags",
    "dataset_version",
    "class_mapping_version",
    "tts_engine",
    "tts_voice",
    "tts_rate",
    "tts_settings",
    "augmentation",
    "augmentation_parent_id",
    "duration_seconds",
    "sample_rate",
    "sha256",
    "wake_start_seconds",
    "wake_end_seconds",
    "created_at",
)
FIELDS = MANIFEST_FIELDS

AUDITION_RECIPES = (
    ("guide_connected", "a teh ko", "connected written guide"),
    ("guide_hyphenated", "a-teh-ko", "hyphenated written guide"),
    ("historical_unqualified", "ateko", "historical spelling for comparison"),
)
FULL_RECIPES = (
    ("a", "a", "syllable_cue", "first syllable"),
    ("te", "teh", "syllable_cue", "second class means intended teh sound"),
    ("ko", "ko", "syllable_cue", "third syllable"),
    (
        "ateko",
        CANONICAL_SYNTHESIS_TEXT,
        "complete_word",
        "user-approved connected synthesis recipe",
    ),
    ("background", "a teh", "hard_negative", "incomplete target"),
    ("background", "teh ko", "hard_negative", "incomplete target"),
    ("background", "ko teh a", "hard_negative", "reordered target"),
    ("background", "a tee ko", "hard_negative", "wrong middle-vowel spelling"),
    ("background", "a tay ko", "hard_negative", "wrong middle-vowel spelling"),
    ("background", "take a coat", "hard_negative", "plausible near miss"),
    ("background", "tea and cocoa", "hard_negative", "plausible near miss"),
    ("background", "turn on the light", "command_without_wake", "command without wake"),
    ("background", "play music", "command_without_wake", "command without wake"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_recording_id(*parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return uuid.uuid5(uuid.NAMESPACE_URL, key).hex


def unique_recording_path(
    output_dir: Path,
    speaker_id: str,
    session_id: str,
    recording_id: str,
) -> Path:
    safe_speaker = re.sub(r"[^a-zA-Z0-9_-]+", "_", speaker_id).strip("_")
    safe_session = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id).strip("_")
    path = output_dir / f"{safe_speaker}_{safe_session}_{recording_id[:12]}.wav"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite recording {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _voice_names() -> list[str]:
    result = subprocess.run(
        ["say", "-v", "?"],
        check=True,
        capture_output=True,
        text=True,
    )
    available: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if match and match.group(2).startswith("en_"):
            available.append(match.group(1).strip())
    return [voice for voice in TTS_VOICE_CANDIDATES if voice in available]


def _require_say() -> None:
    if subprocess.run(
        ["/usr/bin/which", "say"],
        capture_output=True,
        check=False,
    ).returncode:
        raise RuntimeError(
            "macOS 'say' is unavailable; use the real-speaker recorder instead"
        )


def _split_map(groups: list[str], seed: int) -> dict[str, str]:
    if len(groups) < 4:
        raise ValueError(
            "At least four speakers/voices are required for "
            "train/validation/calibration/test"
        )
    shuffled = sorted(groups)
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = max(1, math.floor(len(shuffled) * 0.15))
    calibration_count = max(1, math.floor(len(shuffled) * 0.10))
    test_count = max(1, math.floor(len(shuffled) * 0.15))
    train_count = len(shuffled) - validation_count - calibration_count - test_count
    if train_count < 1:
        raise ValueError("Split allocation leaves no training speakers")
    assignments: dict[str, str] = {}
    cursor = 0
    for split, count in (
        ("train", train_count),
        ("validation", validation_count),
        ("calibration", calibration_count),
        ("test", test_count),
    ):
        for group in shuffled[cursor : cursor + count]:
            assignments[group] = split
        cursor += count
    return assignments


def _audio_metadata(path: Path) -> tuple[float, int, str]:
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Invalid audio dimensions for {path}")
    return info.frames / info.samplerate, int(info.samplerate), _sha256(path)


def build_manifest_record(
    path: Path,
    *,
    recording_id: str,
    label: str,
    phrase: str,
    synthesis_text: str,
    speaker_id: str,
    session_id: str,
    split: str,
    source: str,
    is_synthetic: bool,
    kind: str,
    review_status: str,
    review_notes: str,
    connected_delivery_status: str | None = None,
    device: str = "",
    room: str = "",
    tts_engine: str = "",
    tts_voice: str = "",
    tts_rate: str = "",
    tts_settings: str = "",
    parent_recording_id: str = "",
    augmentation: str = "",
    augmentation_parent_id: str = "",
    wake_start_seconds: str = "",
    wake_end_seconds: str = "",
    dataset_version: str = DATASET_VERSION,
    source_dataset_id: str = "",
    source_dataset_revision: str = "",
    source_record_id: str = "",
    source_license: str = "",
) -> dict[str, str]:
    duration, sample_rate, checksum = _audio_metadata(path)
    if connected_delivery_status is None:
        connected_delivery_status = (
            review_status if label == CANONICAL_WAKE_LABEL else "not_applicable"
        )
    return {
        "path": str(path.resolve()),
        "recording_id": recording_id,
        "parent_recording_id": parent_recording_id or recording_id,
        "label": label,
        "canonical_label": CANONICAL_WAKE_LABEL,
        "pronunciation_guide": PRONUNCIATION_GUIDE,
        "connected_delivery_status": connected_delivery_status,
        "phrase": phrase,
        "synthesis_text": synthesis_text,
        "speaker_id": speaker_id,
        "session_id": session_id,
        "device": device,
        "room": room,
        "split": split,
        "source": source,
        "source_dataset_id": source_dataset_id,
        "source_dataset_revision": source_dataset_revision,
        "source_record_id": source_record_id,
        "source_license": source_license,
        "is_synthetic": str(bool(is_synthetic)).lower(),
        "kind": kind,
        "review_status": review_status,
        "review_notes": review_notes,
        "quality_flags": "",
        "dataset_version": dataset_version,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "tts_engine": tts_engine,
        "tts_voice": tts_voice,
        "tts_rate": str(tts_rate),
        "tts_settings": tts_settings,
        "augmentation": augmentation,
        "augmentation_parent_id": augmentation_parent_id,
        "duration_seconds": f"{duration:.6f}",
        "sample_rate": str(sample_rate),
        "sha256": checksum,
        "wake_start_seconds": wake_start_seconds,
        "wake_end_seconds": wake_end_seconds,
        "created_at": _utc_now(),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: str(row.get(field, "")) for field in MANIFEST_FIELDS}
            for row in rows
        )
    temporary.replace(path)


def _legacy_row(row: dict[str, str], manifest_path: Path) -> dict[str, str]:
    path = Path(row["path"])
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    kind = row.get("kind", "")
    status = "accepted" if kind == "noise_only" else "pending"
    notes = (
        "Programmatic non-speech noise; pronunciation review not applicable."
        if status == "accepted"
        else "Historical v1 speech was not reviewed against a-teh-ko."
    )
    duration = sample_rate = checksum = ""
    if path.is_file():
        try:
            observed_duration, observed_rate, observed_hash = _audio_metadata(path)
            duration = f"{observed_duration:.6f}"
            sample_rate = str(observed_rate)
            checksum = observed_hash
        except (OSError, RuntimeError, ValueError):
            status = "technical_invalid"
    recording_id = stable_recording_id("legacy", path)
    return {
        **{field: "" for field in MANIFEST_FIELDS},
        "path": str(path),
        "recording_id": recording_id,
        "parent_recording_id": recording_id,
        "label": row.get("label", ""),
        "canonical_label": CANONICAL_WAKE_LABEL,
        "pronunciation_guide": PRONUNCIATION_GUIDE,
        "connected_delivery_status": (
            status if row.get("label") == CANONICAL_WAKE_LABEL else "not_applicable"
        ),
        "phrase": row.get("phrase", ""),
        "synthesis_text": row.get("phrase", ""),
        "speaker_id": row.get("speaker_id", ""),
        "session_id": "legacy-generated",
        "split": row.get("split", "unassigned"),
        "source": row.get("source", "legacy_unknown"),
        "is_synthetic": str(row.get("source", "") != "real_recording").lower(),
        "kind": kind,
        "review_status": status,
        "review_notes": notes,
        "dataset_version": LEGACY_DATASET_VERSION,
        "class_mapping_version": "wake-five-class-te-ambiguous-v1",
        "tts_engine": "macos_say" if "say" in row.get("source", "") else "",
        "tts_voice": row.get("speaker_id", ""),
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "sha256": checksum,
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Wake manifest is empty: {path}")
    normalized = (
        [_legacy_row(row, path) for row in rows]
        if "recording_id" not in rows[0]
        else rows
    )
    for row in normalized:
        if not row.get("connected_delivery_status"):
            row["connected_delivery_status"] = (
                row.get("review_status", "pending")
                if row.get("label") == CANONICAL_WAKE_LABEL
                else "not_applicable"
            )
        if row.get("label") not in CLASSES:
            raise ValueError(
                f"Unsupported wake label {row.get('label')!r}; expected {CLASSES}"
            )
        if row.get("review_status") not in REVIEW_STATUSES:
            raise ValueError(
                f"Unsupported review status {row.get('review_status')!r}"
            )
        if row.get("connected_delivery_status") not in CONNECTED_REVIEW_STATUSES:
            raise ValueError(
                "Unsupported connected-delivery status "
                f"{row.get('connected_delivery_status')!r}"
            )
        if row.get("split") not in SPLITS:
            raise ValueError(f"Unsupported wake split {row.get('split')!r}")
    return normalized


def _synthesize(path: Path, voice: str, rate: int, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing wake audio: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(rate),
                "-o",
                str(path),
                "--file-format=WAVE",
                "--data-format=LEF32@16000",
                text,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError(
                "synthesizer returned success but produced invalid audio"
            )
    except (OSError, subprocess.CalledProcessError) as error:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Speech synthesis failed voice={voice!r} text={text!r}: {error}"
        ) from error
    except RuntimeError as error:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Speech synthesis failed voice={voice!r} text={text!r}: {error}"
        ) from error


def generate_audition_set(
    output_dir: Path,
    *,
    voice_count: int = 4,
    rate: int = 175,
) -> int:
    """Generate a small pending-review set; no clip is marked approved."""

    _require_say()
    voices = _voice_names()[:voice_count]
    if len(voices) < 3:
        raise RuntimeError("At least three conventional English voices are required")
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Audition manifest already exists: {manifest_path}")
    rows: list[dict[str, str]] = []
    for voice in voices:
        for recipe, text, note in AUDITION_RECIPES:
            recording_id = stable_recording_id(
                DATASET_VERSION,
                "audition",
                voice,
                rate,
                recipe,
            )
            path = output_dir / f"{recording_id[:12]}_{recipe}.wav"
            _synthesize(path, voice, rate, text)
            rows.append(
                build_manifest_record(
                    path,
                    recording_id=recording_id,
                    label=CANONICAL_WAKE_LABEL,
                    phrase=PRONUNCIATION_GUIDE,
                    synthesis_text=text,
                    speaker_id=f"tts:{voice}",
                    session_id="audition-2026-10-05",
                    split="unassigned",
                    source="macos_say_synthetic",
                    is_synthetic=True,
                    kind="pronunciation_audition",
                    review_status="pending",
                    review_notes=note,
                    tts_engine="macos_say",
                    tts_voice=voice,
                    tts_rate=str(rate),
                    tts_settings="WAVE LEF32 16000 Hz",
                )
            )
    write_manifest(manifest_path, rows)
    return len(rows)


def generate_synthetic_dataset(
    output_dir: Path,
    seed: int = 42,
    *,
    voice_count: int = 8,
    rates: tuple[int, ...] = (145, 175, 205),
) -> int:
    """Generate a provisional connected-word corpus pending per-voice review."""

    _require_say()
    voices = _voice_names()[:voice_count]
    if len(voices) < 4:
        raise RuntimeError("At least four conventional English voices are required")
    manifest_path = output_dir / "manifest.csv"
    split_path = output_dir / "splits.json"
    if manifest_path.exists() or split_path.exists():
        raise FileExistsError(f"Refusing to overwrite corpus under {output_dir}")
    assignments = _split_map([f"tts:{voice}" for voice in voices], seed)
    rows: list[dict[str, str]] = []
    for voice in voices:
        speaker = f"tts:{voice}"
        for label, text, kind, note in FULL_RECIPES:
            for rate in rates:
                recipe = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
                recording_id = stable_recording_id(
                    DATASET_VERSION,
                    "synthetic",
                    voice,
                    rate,
                    label,
                    text,
                )
                path = output_dir / "audio" / (
                    f"{recording_id[:16]}_{recipe}_{rate}.wav"
                )
                try:
                    _synthesize(path, voice, rate, text)
                except RuntimeError as error:
                    LOGGER.warning("%s", error)
                    continue
                rows.append(
                    build_manifest_record(
                        path,
                        recording_id=recording_id,
                        label=label,
                        phrase=PRONUNCIATION_GUIDE if label == "ateko" else text,
                        synthesis_text=text,
                        speaker_id=speaker,
                        session_id=f"macos-say-{voice}",
                        split=assignments[speaker],
                        source="macos_say_synthetic",
                        is_synthetic=True,
                        kind=kind,
                        review_status="pending",
                        review_notes=note,
                        tts_engine="macos_say",
                        tts_voice=voice,
                        tts_rate=str(rate),
                        tts_settings="WAVE LEF32 16000 Hz",
                    )
                )
        rng = np.random.default_rng(
            int(stable_recording_id(DATASET_VERSION, voice, seed)[:16], 16)
        )
        for index in range(12):
            recording_id = stable_recording_id(
                DATASET_VERSION,
                "noise",
                voice,
                index,
                seed,
            )
            path = output_dir / "audio" / f"{recording_id[:16]}_noise.wav"
            samples = rng.normal(
                0.0,
                rng.uniform(0.0001, 0.001),
                32_000,
            ).astype(np.float32)
            write_wav(path, samples, 16_000)
            rows.append(
                build_manifest_record(
                    path,
                    recording_id=recording_id,
                    label="background",
                    phrase="",
                    synthesis_text="",
                    speaker_id=speaker,
                    session_id=f"generated-noise-{voice}",
                    split=assignments[speaker],
                    source="generated_noise",
                    is_synthetic=True,
                    kind="noise_only",
                    review_status="accepted",
                    review_notes=(
                        "Programmatic non-speech noise; pronunciation review "
                        "not applicable."
                    ),
                )
            )
    if not rows:
        raise RuntimeError("No wake-word audio was generated")
    write_manifest(manifest_path, rows)
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": DATASET_VERSION,
                "seed": seed,
                "group_key": "speaker_id",
                "assignments": assignments,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(rows)


def _write_split_assignments(
    path: Path,
    *,
    dataset_version: str,
    seed: int,
    assignments: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": dataset_version,
                "seed": seed,
                "group_key": "speaker_id",
                "assignments": dict(sorted(assignments.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expand_synthetic_training_corpus(
    base_manifest: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    extra_rates: tuple[int, ...] = (130, 160, 190, 220),
    new_voice_count: int = 3,
) -> int:
    """Add train-only voice/rate diversity without altering protected splits."""

    _require_say()
    manifest_path = output_dir / "manifest.csv"
    split_path = output_dir / "splits.json"
    if manifest_path.exists() or split_path.exists():
        raise FileExistsError(f"Refusing to overwrite corpus under {output_dir}")
    base_rows = read_manifest(base_manifest)
    versions = {row["dataset_version"] for row in base_rows}
    if versions != {DATASET_VERSION}:
        raise ValueError(
            f"Wake expansion requires exactly {DATASET_VERSION!r}, got {versions}"
        )
    rows = [
        {**row, "dataset_version": EXPANDED_DATASET_VERSION}
        for row in base_rows
    ]
    assignments = {
        row["speaker_id"]: row["split"]
        for row in rows
        if row["speaker_id"] and row["split"] != "unassigned"
    }
    existing_voices = {
        row["tts_voice"] for row in rows if row.get("tts_voice")
    }
    train_voices = sorted(
        {
            row["tts_voice"]
            for row in rows
            if row.get("tts_voice") and row["split"] == "train"
        }
    )
    available = _voice_names()
    new_voices = [
        voice for voice in available if voice not in existing_voices
    ][:new_voice_count]
    for voice in new_voices:
        assignments[f"tts:{voice}"] = "train"

    recipes: list[tuple[str, str, str, str, int]] = []
    for voice in train_voices:
        for rate in extra_rates:
            recipes.append(
                (
                    voice,
                    CANONICAL_WAKE_LABEL,
                    CANONICAL_SYNTHESIS_TEXT,
                    "complete_word",
                    rate,
                )
            )
    for voice in new_voices:
        for label, text, kind, _ in FULL_RECIPES:
            for rate in (145, 175, 205):
                recipes.append((voice, label, text, kind, rate))

    for voice, label, text, kind, rate in recipes:
        speaker = f"tts:{voice}"
        recording_id = stable_recording_id(
            EXPANDED_DATASET_VERSION,
            "synthetic",
            voice,
            rate,
            label,
            text,
        )
        recipe_name = re.sub(
            r"[^a-z0-9]+", "_", text.casefold()
        ).strip("_")
        path = output_dir / "audio" / (
            f"{recording_id[:16]}_{recipe_name}_{rate}.wav"
        )
        _synthesize(path, voice, rate, text)
        rows.append(
            build_manifest_record(
                path,
                recording_id=recording_id,
                label=label,
                phrase=PRONUNCIATION_GUIDE if label == "ateko" else text,
                synthesis_text=text,
                speaker_id=speaker,
                session_id=f"macos-say-{voice}-expanded",
                split="train",
                source="macos_say_synthetic",
                is_synthetic=True,
                kind=kind,
                review_status="pending",
                review_notes=(
                    "Train-only connected-word rate expansion."
                    if label == CANONICAL_WAKE_LABEL
                    else "Train-only matched negative from the established recipe."
                ),
                tts_engine="macos_say",
                tts_voice=voice,
                tts_rate=str(rate),
                tts_settings="WAVE LEF32 16000 Hz",
                dataset_version=EXPANDED_DATASET_VERSION,
            )
        )
    write_manifest(manifest_path, rows)
    _write_split_assignments(
        split_path,
        dataset_version=EXPANDED_DATASET_VERSION,
        seed=seed,
        assignments=assignments,
    )
    return len(rows)


def add_hf_command_negatives(
    base_manifest: Path,
    output_dir: Path,
    *,
    maximum_rows: int = 320,
    maximum_per_speaker: int = 2,
    seed: int = 42,
) -> int:
    """Add transcript-screened command speech as train-only non-wake audio."""

    if maximum_rows <= 0 or maximum_per_speaker <= 0:
        raise ValueError("HF negative limits must be positive")
    manifest_path = output_dir / "manifest.csv"
    split_path = output_dir / "splits.json"
    if manifest_path.exists() or split_path.exists():
        raise FileExistsError(f"Refusing to overwrite corpus under {output_dir}")
    base_rows = read_manifest(base_manifest)
    versions = {row["dataset_version"] for row in base_rows}
    if versions != {EXPANDED_DATASET_VERSION}:
        raise ValueError(
            "HF command negatives require the expanded wake-v4 manifest; "
            f"got {versions}"
        )
    rows = [
        {**row, "dataset_version": COMMAND_NEGATIVE_DATASET_VERSION}
        for row in base_rows
    ]
    assignments = {
        row["speaker_id"]: row["split"]
        for row in rows
        if row["speaker_id"] and row["split"] != "unassigned"
    }

    from me2.vcm.ctc import normalize_transcript
    from me2.vcm.data import (
        DATASET_ID,
        DATASET_REVISION,
        command_from_row,
        decode_audio,
        load_hf_splits,
    )

    train_split = load_hf_splits(DATASET_ID, DATASET_REVISION)["train"]
    candidate_indices = list(range(len(train_split)))
    np.random.default_rng(seed).shuffle(candidate_indices)
    speaker_counts: Counter[str] = Counter()
    added = 0
    skipped: Counter[str] = Counter()
    for index in candidate_indices:
        if added >= maximum_rows:
            break
        row = train_split[index]
        transcript = str(row.get("transcript") or "").strip()
        normalized = normalize_transcript(transcript)
        if not normalized:
            skipped["empty_transcript"] += 1
            continue
        compact = normalized.replace(" ", "")
        if "ateko" in compact:
            skipped["target_in_transcript"] += 1
            continue
        if command_from_row(row) == "out_of_scope":
            skipped["out_of_scope_reserved_for_command_calibration"] += 1
            continue
        source_speaker = str(row.get("speaker_id") or "").strip()
        if not source_speaker:
            skipped["missing_speaker"] += 1
            continue
        speaker_id = f"hf-command:{source_speaker}"
        if speaker_counts[speaker_id] >= maximum_per_speaker:
            skipped["per_speaker_limit"] += 1
            continue
        samples, sample_rate = decode_audio(row["audio"])
        samples = resample_audio(samples, sample_rate, 16_000)
        source_record_id = str(row.get("file") or index)
        recording_id = stable_recording_id(
            COMMAND_NEGATIVE_DATASET_VERSION,
            DATASET_REVISION,
            source_record_id,
        )
        path = output_dir / "audio" / f"{recording_id[:16]}_command.wav"
        write_wav(path, samples, 16_000)
        rows.append(
            build_manifest_record(
                path,
                recording_id=recording_id,
                label="background",
                phrase=transcript,
                synthesis_text="",
                speaker_id=speaker_id,
                session_id=f"hf-command-{source_speaker}",
                split="train",
                source="hf_me2_command_without_wake",
                is_synthetic=bool(row.get("is_synthetic", False)),
                kind="command_without_wake",
                review_status="pending",
                review_notes=(
                    "Transcript-screened command audio; waveform was not "
                    "manually reviewed for an incidental wake occurrence."
                ),
                connected_delivery_status="not_applicable",
                parent_recording_id=recording_id,
                dataset_version=COMMAND_NEGATIVE_DATASET_VERSION,
                source_dataset_id=DATASET_ID,
                source_dataset_revision=DATASET_REVISION,
                source_record_id=source_record_id,
                source_license=(
                    "Research/education dataset; retain component provenance "
                    "and consult upstream LICENSE.md before redistribution."
                ),
            )
        )
        assignments[speaker_id] = "train"
        speaker_counts[speaker_id] += 1
        added += 1
    if added == 0:
        raise RuntimeError("No eligible Hugging Face command negatives were added")
    write_manifest(manifest_path, rows)
    _write_split_assignments(
        split_path,
        dataset_version=COMMAND_NEGATIVE_DATASET_VERSION,
        seed=seed,
        assignments=assignments,
    )
    ledger = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_split": "train",
        "usage": "train-only command-without-wake negatives",
        "selection_seed": seed,
        "rows_added": added,
        "maximum_rows": maximum_rows,
        "maximum_per_speaker": maximum_per_speaker,
        "speakers_added": len(speaker_counts),
        "skipped": dict(skipped),
        "manual_audio_review_complete": False,
        "redistribution_note": (
            "Preserve upstream provenance and review the dataset LICENSE.md "
            "and component terms before redistributing derived audio."
        ),
    }
    (output_dir / "source_ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def apply_review(manifest_path: Path, review_path: Path) -> int:
    rows = read_manifest(manifest_path)
    by_id = {row["recording_id"]: row for row in rows}
    changed = 0
    with review_path.open(newline="", encoding="utf-8") as stream:
        for review in csv.DictReader(stream):
            recording_id = review.get("recording_id", "")
            if recording_id not in by_id:
                raise ValueError(f"Unknown recording_id in review: {recording_id}")
            status = review.get("review_status", "")
            if status not in REVIEW_STATUSES:
                raise ValueError(f"Invalid review status {status!r}")
            by_id[recording_id]["review_status"] = status
            connected_status = review.get("connected_delivery_status", "")
            if connected_status:
                if connected_status not in CONNECTED_REVIEW_STATUSES:
                    raise ValueError(
                        f"Invalid connected_delivery_status {connected_status!r}"
                    )
                by_id[recording_id][
                    "connected_delivery_status"
                ] = connected_status
            by_id[recording_id]["review_notes"] = review.get("review_notes", "")
            changed += 1
    write_manifest(manifest_path, rows)
    return changed


def freeze_splits(
    manifest_path: Path,
    output_path: Path,
    seed: int = 42,
) -> dict[str, str]:
    """Persist assignments once; reruns preserve existing speaker decisions."""

    rows = read_manifest(manifest_path)
    existing: dict[str, str] = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8")).get(
            "assignments", {}
        )
    observed = {
        row["speaker_id"]: row["split"]
        for row in rows
        if row["speaker_id"] and row["split"] != "unassigned"
    }
    for speaker, split in observed.items():
        prior = existing.get(speaker)
        if prior is not None and prior != split:
            raise ValueError(
                f"Frozen split conflict for {speaker}: {prior} != {split}"
            )
        existing[speaker] = split
    unassigned = sorted(
        {row["speaker_id"] for row in rows if row["speaker_id"] not in existing}
    )
    if unassigned:
        existing.update(_split_map(unassigned, seed))
    for row in rows:
        row["split"] = existing[row["speaker_id"]]
    write_manifest(manifest_path, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": sorted(
                    {row["dataset_version"] for row in rows}
                ),
                "seed": seed,
                "group_key": "speaker_id",
                "assignments": dict(sorted(existing.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return existing


def _near_duplicate_fingerprint(path: Path) -> str:
    samples, _ = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if mono.size == 0:
        return "empty"
    peak = max(float(np.max(np.abs(mono))), 1e-8)
    chunks = np.array_split(np.abs(mono / peak), 64)
    summary = np.asarray(
        [float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks]
    )
    return hashlib.sha256(np.round(summary, 2).tobytes()).hexdigest()


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    issues: list[dict[str, str]] = []
    ids: set[str] = set()
    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    near: dict[str, list[dict[str, str]]] = defaultdict(list)
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    durations: list[float] = []
    for row in rows:
        recording_id = row["recording_id"]
        if recording_id in ids:
            issues.append(
                {"recording_id": recording_id, "reason": "duplicate_recording_id"}
            )
        ids.add(recording_id)
        speaker_splits[row["speaker_id"]].add(row["split"])
        parent_splits[row["parent_recording_id"]].add(row["split"])
        path = Path(row["path"])
        if not path.is_file():
            issues.append({"recording_id": recording_id, "reason": "missing_file"})
            continue
        try:
            duration, sample_rate, checksum = _audio_metadata(path)
            durations.append(duration)
            if row.get("sha256") and checksum != row["sha256"]:
                issues.append(
                    {"recording_id": recording_id, "reason": "checksum_mismatch"}
                )
            if sample_rate != 16_000:
                issues.append(
                    {
                        "recording_id": recording_id,
                        "reason": f"unexpected_sample_rate:{sample_rate}",
                    }
                )
            exact[checksum].append(row)
            near[_near_duplicate_fingerprint(path)].append(row)
        except (OSError, RuntimeError, ValueError) as error:
            issues.append(
                {"recording_id": recording_id, "reason": f"invalid_audio:{error}"}
            )
    speaker_leaks = {
        speaker: sorted(splits)
        for speaker, splits in speaker_splits.items()
        if speaker and len(splits - {"unassigned"}) > 1
    }
    parent_leaks = {
        parent: sorted(splits)
        for parent, splits in parent_splits.items()
        if parent and len(splits - {"unassigned"}) > 1
    }
    exact_groups = [
        {
            "sha256": checksum,
            "recording_ids": [row["recording_id"] for row in group],
            "splits": sorted({row["split"] for row in group}),
        }
        for checksum, group in exact.items()
        if len(group) > 1
    ]
    cross_split_duplicates = [
        group
        for group in exact_groups
        if len(set(group["splits"]) - {"unassigned"}) > 1
    ]
    speech_rows = [row for row in rows if row["kind"] != "noise_only"]
    positive_rows = [row for row in rows if row["label"] == CANONICAL_WAKE_LABEL]
    report = {
        "audit_schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "dataset_versions": dict(Counter(row["dataset_version"] for row in rows)),
        "rows": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "splits": dict(Counter(row["split"] for row in rows)),
        "speakers": len({row["speaker_id"] for row in rows if row["speaker_id"]}),
        "real_speakers": len(
            {
                row["speaker_id"]
                for row in rows
                if row["is_synthetic"].casefold() != "true"
            }
        ),
        "sources": dict(Counter(row["source"] for row in rows)),
        "review_status": dict(Counter(row["review_status"] for row in rows)),
        "speech_review_coverage": (
            sum(row["review_status"] != "pending" for row in speech_rows)
            / len(speech_rows)
            if speech_rows
            else None
        ),
        "positive_review_status": dict(
            Counter(row["review_status"] for row in positive_rows)
        ),
        "positive_connected_delivery_status": dict(
            Counter(row["connected_delivery_status"] for row in positive_rows)
        ),
        "connected_delivery_review_coverage": (
            sum(
                row["connected_delivery_status"] != "pending"
                for row in positive_rows
            )
            / len(positive_rows)
            if positive_rows
            else None
        ),
        "duration_seconds": {
            "total": float(sum(durations)),
            "minimum": min(durations, default=None),
            "maximum": max(durations, default=None),
        },
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_group_count": sum(
            len(group) > 1 for group in near.values()
        ),
        "cross_split_exact_duplicates": cross_split_duplicates,
        "speaker_split_leakage": speaker_leaks,
        "parent_split_leakage": parent_leaks,
        "issues": issues,
        "trainable_approved_positives": sum(
            row["label"] == CANONICAL_WAKE_LABEL
            and row["review_status"] == "accepted"
            and row["connected_delivery_status"] == "accepted"
            for row in rows
        ),
    }
    report["structurally_valid"] = not (
        issues or speaker_leaks or parent_leaks or cross_split_duplicates
    )
    report["valid_for_training"] = bool(
        report["structurally_valid"]
        and report["trainable_approved_positives"] > 0
        and report["speech_review_coverage"] == 1.0
    )
    return report


def trainable_rows(
    rows: list[dict[str, str]],
    *,
    allow_pending_review: bool = False,
) -> list[dict[str, str]]:
    statuses = {"accepted", "pending"} if allow_pending_review else {"accepted"}
    selected = [
        row
        for row in rows
        if row["review_status"] in statuses
        and (
            row["label"] != CANONICAL_WAKE_LABEL
            or row["connected_delivery_status"] in statuses
        )
    ]
    incompatible_approved = [
        row
        for row in selected
        if row["review_status"] == "accepted"
        and row["label"] == CANONICAL_WAKE_LABEL
        and row.get("is_synthetic", "").casefold() == "true"
        and row.get("synthesis_text", "") != CANONICAL_SYNTHESIS_TEXT
    ]
    if incompatible_approved:
        raise ValueError(
            "An approved synthetic wake clip does not use the user-confirmed "
            f"connected recipe {CANONICAL_SYNTHESIS_TEXT!r}: "
            f"{incompatible_approved[0]['recording_id']}"
        )
    if not allow_pending_review and not any(
        row["label"] == CANONICAL_WAKE_LABEL for row in selected
    ):
        raise ValueError(
            "No pronunciation-approved complete a-teh-ko positives are available. "
            "Review the audition/corpus manifest first. "
            "--allow-pending-review is only for non-deployment experiments."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview")
    preview.add_argument(
        "--output",
        type=Path,
        default=ROOT / "wake-word/data/v3/audition",
    )
    preview.add_argument("--voices", type=int, default=4)
    preview.add_argument("--rate", type=int, default=175)
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--output",
        type=Path,
        default=ROOT / "wake-word/data/v3/synthetic",
    )
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--voices", type=int, default=8)
    generate.add_argument("--rates", default="145,175,205")
    expand = subparsers.add_parser("expand-synthetic")
    expand.add_argument("base_manifest", type=Path)
    expand.add_argument(
        "--output",
        type=Path,
        default=ROOT / "wake-word/data/v4/synthetic-expanded",
    )
    expand.add_argument("--seed", type=int, default=42)
    expand.add_argument("--extra-rates", default="130,160,190,220")
    expand.add_argument("--new-voices", type=int, default=3)
    command_negatives = subparsers.add_parser("add-command-negatives")
    command_negatives.add_argument("base_manifest", type=Path)
    command_negatives.add_argument(
        "--output",
        type=Path,
        default=ROOT / "wake-word/data/v5/command-negatives",
    )
    command_negatives.add_argument("--maximum-rows", type=int, default=320)
    command_negatives.add_argument("--maximum-per-speaker", type=int, default=2)
    command_negatives.add_argument("--seed", type=int, default=42)
    audit = subparsers.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--output", type=Path)
    review = subparsers.add_parser("apply-review")
    review.add_argument("manifest", type=Path)
    review.add_argument("review_csv", type=Path)
    freeze = subparsers.add_parser("freeze-splits")
    freeze.add_argument("manifest", type=Path)
    freeze.add_argument("output", type=Path)
    freeze.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "preview":
        count = generate_audition_set(
            args.output,
            voice_count=args.voices,
            rate=args.rate,
        )
        print(f"Generated {count} pending-review clips at {args.output}")
    elif args.command == "generate":
        rates = tuple(int(value) for value in args.rates.split(",") if value)
        count = generate_synthetic_dataset(
            args.output,
            args.seed,
            voice_count=args.voices,
            rates=rates,
        )
        print(f"Generated {count} provisional clips at {args.output}")
    elif args.command == "expand-synthetic":
        rates = tuple(
            int(value) for value in args.extra_rates.split(",") if value
        )
        count = expand_synthetic_training_corpus(
            args.base_manifest,
            args.output,
            seed=args.seed,
            extra_rates=rates,
            new_voice_count=args.new_voices,
        )
        print(f"Expanded corpus to {count} clips at {args.output}")
    elif args.command == "add-command-negatives":
        count = add_hf_command_negatives(
            args.base_manifest,
            args.output,
            maximum_rows=args.maximum_rows,
            maximum_per_speaker=args.maximum_per_speaker,
            seed=args.seed,
        )
        print(f"Added command negatives; corpus has {count} clips at {args.output}")
    elif args.command == "audit":
        report = audit_manifest(args.manifest)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    elif args.command == "apply-review":
        print(f"Applied {apply_review(args.manifest, args.review_csv)} reviews")
    else:
        assignments = freeze_splits(args.manifest, args.output, args.seed)
        print(f"Frozen {len(assignments)} speaker assignments at {args.output}")


if __name__ == "__main__":
    main()
