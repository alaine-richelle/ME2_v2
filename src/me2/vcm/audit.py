"""Audit the pinned command dataset, partitions, text contract, and grammar."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from me2.vcm.ctc import (
    CHARSET,
    DECODER_VERSION,
    TEXT_NORMALIZER_VERSION,
    CompactCTCModel,
    coerce_grammar_entries,
    normalize_transcript,
    required_ctc_frames,
)
from me2.vcm.ctc_training import grammar_entry_for_row
from me2.vcm.data import (
    DATASET_ID,
    DATASET_REVISION,
    REQUIRED_SPLITS,
    command_from_row,
    load_hf_splits,
    split_train_validation_calibration,
    validate_official_speakers,
)

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_REVIEW_REVISION = "52b38f7fdaf35a0f5b5a8dbbd239f4bfb7700e24"
CTC_REFERENCE_REVISION = "f5d2dc585b49b46e8518ec6122261a9ca7f2f5e4"


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _split_audit(split) -> dict[str, Any]:
    empty_rows: list[int] = []
    digit_rows = 0
    repeated_rows = 0
    infeasible_declared_duration: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    normalized_semantics: dict[str, set[tuple[str, str, str]]] = {}
    lengths: list[int] = []

    for index, row in enumerate(split):
        text = normalize_transcript(str(row.get("transcript") or ""))
        class_counts[command_from_row(row)] += 1
        source_counts[str(row.get("source") or "")] += 1
        provenance_counts[str(row.get("transcript_source") or "")] += 1
        lengths.append(len(text))
        if not text:
            empty_rows.append(index)
            continue
        digit_rows += int(any(character.isdigit() for character in text))
        repeated_rows += int(any(left == right for left, right in pairwise(text)))
        entry = grammar_entry_for_row(row)
        if entry is not None:
            normalized_semantics.setdefault(entry.text, set()).add(
                (
                    entry.intent,
                    entry.variation,
                    str(entry.slots.get("value", "")),
                )
            )
        duration = float(row.get("duration_s") or 0.0)
        if duration > 0:
            feature_frames = math.floor(duration * 16_000 / 160) + 1
            output_frames = int(
                CompactCTCModel.output_lengths(torch.tensor([feature_frames]))[0]
            )
            required = required_ctc_frames(text)
            if output_frames < required:
                infeasible_declared_duration.append(
                    {
                        "index": index,
                        "file": str(row.get("file") or ""),
                        "duration_s": duration,
                        "output_frames": output_frames,
                        "required_frames": required,
                        "transcript": str(row.get("transcript") or ""),
                    }
                )

    collisions = {
        text: sorted(semantics)
        for text, semantics in normalized_semantics.items()
        if len(semantics) > 1
    }
    sorted_lengths = sorted(lengths)
    return {
        "rows": len(split),
        "classes": dict(sorted(class_counts.items())),
        "speakers": len(set(map(str, split["speaker_id"]))),
        "sources": dict(source_counts.most_common()),
        "transcript_provenance": dict(provenance_counts.most_common()),
        "empty_transcript_rows": empty_rows,
        "digit_transcript_rows": digit_rows,
        "adjacent_repeat_rows": repeated_rows,
        "maximum_normalized_characters": max(lengths, default=0),
        "p99_normalized_characters": (
            sorted_lengths[min(len(sorted_lengths) - 1, int(0.99 * len(sorted_lengths)))]
            if sorted_lengths
            else 0
        ),
        "semantic_collision_count": len(collisions),
        "semantic_collisions": collisions,
        "declared_duration_infeasible_count": len(infeasible_declared_duration),
        "declared_duration_infeasible_examples": infeasible_declared_duration[:50],
    }


def audit_dataset(
    *,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    dataset = load_hf_splits(DATASET_ID, DATASET_REVISION)
    validate_official_speakers(dataset)
    partitions = split_train_validation_calibration(
        dataset["train"],
        validation_fraction,
        calibration_fraction,
        seed,
    )
    grammar_entries = [
        entry
        for index in partitions["train"]
        if (entry := grammar_entry_for_row(dataset["train"][index])) is not None
    ]
    grammar = coerce_grammar_entries(grammar_entries)
    return {
        "audit_schema_version": 1,
        "revisions": {
            "checkout_head": _git_revision(),
            "reviewed_integration_revision": INTEGRATION_REVIEW_REVISION,
            "reviewed_ctc_reference_revision": CTC_REFERENCE_REVISION,
            "dataset_revision": DATASET_REVISION,
        },
        "dataset": {"id": DATASET_ID, "configuration": "default"},
        "text_contract": {
            "normalizer_version": TEXT_NORMALIZER_VERSION,
            "charset": CHARSET,
            "blank_id": len(CHARSET),
            "decoder_version": DECODER_VERSION,
        },
        "partitions": {
            name: {
                "rows": len(indices),
                "speakers": len(
                    {str(dataset["train"][index]["speaker_id"]) for index in indices}
                ),
            }
            for name, indices in partitions.items()
        },
        "grammar": {
            "terminals": len(grammar),
            "provenance": "approved official-train transcripts only",
            "held_out_transcripts_used": False,
        },
        "splits": {name: _split_audit(dataset[name]) for name in REQUIRED_SPLITS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/vcm_ctc_data_audit.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    args = parser.parse_args()
    report = audit_dataset(
        validation_fraction=args.validation_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
