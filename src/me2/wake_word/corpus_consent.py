"""Validate external-only, consented speaker-disjoint wake corpus manifests."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PRIVATE_ROOT = Path("/home/rpiadmin/me2-validation-evidence/wake-corpus").resolve()
_SPEAKER = re.compile(r"^spk-[a-z0-9]{6,32}$")
_PARTITIONS = {"train", "validation", "held_out"}


def validate_manifest(manifest: dict) -> dict:
    if set(manifest) != {"schema_version", "speakers"} or manifest["schema_version"] != 1:
        raise ValueError("manifest must contain schema_version=1 and speakers")
    speakers = manifest["speakers"]
    if not isinstance(speakers, list) or len(speakers) < 3:
        raise ValueError("at least three consented independent speakers are required")
    seen, partitions = set(), set()
    required = {"speaker_id", "consent_status", "consent_time", "recording_condition", "device_profile", "partition"}
    for entry in speakers:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("speaker entries must contain only the required opaque consent fields")
        identifier = entry["speaker_id"]
        if not isinstance(identifier, str) or not _SPEAKER.fullmatch(identifier) or identifier in seen:
            raise ValueError("speaker_id must be unique opaque spk- identifier")
        if entry["consent_status"] != "granted" or not all(isinstance(entry[key], str) and entry[key] for key in required - {"partition"}):
            raise ValueError("every speaker requires granted consent and non-empty provenance fields")
        if entry["partition"] not in _PARTITIONS:
            raise ValueError("invalid speaker partition")
        seen.add(identifier); partitions.add(entry["partition"])
    if partitions != _PARTITIONS:
        raise ValueError("train, validation, and held_out must each have a distinct speaker")
    return {"schema_version": 1, "speaker_count": len(seen), "partitions": sorted(partitions), "recording_enabled": True}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.manifest.expanduser().resolve()
    if PRIVATE_ROOT not in path.parents:
        raise ValueError("manifest must remain under approved external evidence root")
    print(json.dumps(validate_manifest(json.loads(path.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
