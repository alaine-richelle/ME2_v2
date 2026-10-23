"""Fail-closed audit for an external-only, consented wake-word corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

from me2.wake_word.data import CANONICAL_WAKE_LABEL, CLASSES

PRIVATE_ROOT = Path("/home/rpiadmin/me2-validation-evidence/wake-corpus").resolve()
_PARTITIONS = {"train", "validation", "held_out"}
_SPEAKER_PREFIX = "spk-"
_REQUIRED_SPEAKER = {
    "speaker_id",
    "consent_status",
    "consent_time",
    "recording_condition",
    "device_profile",
    "partition",
}
_REQUIRED_RECORD = {
    "recording_id",
    "source_recording_id",
    "path",
    "sha256",
    "speaker_id",
    "partition",
    "source_partition",
    "label",
    "kind",
    "review_status",
    "connected_delivery_status",
}
_REQUIRED_MANIFEST = {
    "schema_version",
    "corpus_id",
    "source_manifest_sha256",
    "speakers",
    "records",
}


def _private_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if PRIVATE_ROOT not in resolved.parents:
        raise ValueError("external corpus files must remain under the approved evidence root")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_speakers(speakers: Any) -> dict[str, str]:
    if not isinstance(speakers, list) or len(speakers) < 3:
        raise ValueError("at least three consented speakers are required")
    assignments: dict[str, str] = {}
    for speaker in speakers:
        if not isinstance(speaker, dict) or set(speaker) != _REQUIRED_SPEAKER:
            raise ValueError("speaker entries must contain only required consent/provenance fields")
        identifier = _require_string(speaker, "speaker_id")
        if not identifier.startswith(_SPEAKER_PREFIX) or identifier in assignments:
            raise ValueError("speaker_id must be a unique opaque spk- identifier")
        if speaker["consent_status"] != "granted":
            raise ValueError("every included speaker requires granted consent")
        for field in _REQUIRED_SPEAKER - {"speaker_id", "consent_status", "partition"}:
            _require_string(speaker, field)
        partition = speaker["partition"]
        if partition not in _PARTITIONS:
            raise ValueError("speaker partition must be train, validation, or held_out")
        assignments[identifier] = partition
    if set(assignments.values()) != _PARTITIONS:
        raise ValueError("train, validation, and held_out must each have a speaker")
    return assignments


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate external data without emitting paths, raw IDs, or audio content."""
    path = _private_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != _REQUIRED_MANIFEST:
        raise ValueError("manifest has an unsupported schema")
    if manifest["schema_version"] != 1 or not isinstance(manifest["corpus_id"], str):
        raise ValueError("manifest must have schema_version=1 and corpus_id")
    source_hash = _require_string(manifest, "source_manifest_sha256")
    if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        raise ValueError("source_manifest_sha256 must be a lowercase SHA-256 digest")
    assignments = _validate_speakers(manifest["speakers"])
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("manifest requires at least one record")

    issues: list[str] = []
    ids: set[str] = set()
    source_ids: set[str] = set()
    hashes: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    covered_speakers: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _REQUIRED_RECORD:
            raise ValueError("record entries must contain only the approved ingestion fields")
        recording_id = _require_string(record, "recording_id")
        source_id = _require_string(record, "source_recording_id")
        if recording_id in ids or source_id in source_ids:
            issues.append("duplicate_recording_identifier")
        ids.add(recording_id)
        source_ids.add(source_id)
        speaker = _require_string(record, "speaker_id")
        partition = record["partition"]
        if assignments.get(speaker) != partition:
            issues.append("speaker_partition_mismatch")
        if partition not in _PARTITIONS:
            issues.append("unsupported_partition")
        label = record["label"]
        if label not in CLASSES:
            issues.append("unsupported_label")
        if record["review_status"] != "accepted":
            issues.append("unreviewed_record")
        if label == CANONICAL_WAKE_LABEL and record["connected_delivery_status"] != "accepted":
            issues.append("unapproved_complete_wake")
        checksum = _require_string(record, "sha256")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            issues.append("invalid_record_checksum")
            continue
        audio_path = _private_path(Path(_require_string(record, "path")))
        if not audio_path.is_file():
            issues.append("missing_audio")
            continue
        try:
            info = sf.info(audio_path)
        except RuntimeError:
            issues.append("invalid_audio")
            continue
        if info.samplerate != 16_000 or info.channels != 1 or info.subtype != "PCM_16":
            issues.append("unsupported_audio_format")
        if _sha256(audio_path) != checksum:
            issues.append("checksum_mismatch")
        hashes[checksum].add(partition)
        covered_speakers.add(speaker)
        counts[("partition", partition)] += 1
        counts[("label", label)] += 1
        counts[("kind", record["kind"])] += 1

    cross_partition_duplicates = sum(len(partitions) > 1 for partitions in hashes.values())
    if cross_partition_duplicates:
        issues.append("cross_partition_duplicate_audio")
    missing_speaker_data = set(assignments) - covered_speakers
    if missing_speaker_data:
        issues.append("speaker_without_records")
    valid = not issues
    return {
        "audit_schema_version": 1,
        "corpus_id": manifest["corpus_id"],
        "records": len(records),
        "speakers": len(assignments),
        "partitions": {name: counts[("partition", name)] for name in sorted(_PARTITIONS)},
        "labels": dict(sorted((name, count) for (_, name), count in counts.items() if _ == "label")),
        "kinds": dict(sorted((name, count) for (_, name), count in counts.items() if _ == "kind")),
        "cross_partition_duplicate_groups": cross_partition_duplicates,
        "valid_for_training_proposal": valid,
        "issues": sorted(set(issues)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = _private_path(args.output)
    report = audit_manifest(args.manifest)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
