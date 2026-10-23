"""Create non-private templates for a consented external wake corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from me2.wake_word.corpus_consent import PRIVATE_ROOT, SCHEMA_VERSION, TRAINING_PARTITIONS

_TEMPLATE_DIRECTORY = "templates"
_TEMPLATES: dict[str, dict[str, Any]] = {
    "consent-manifest.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ME2 external wake consent manifest",
        "type": "object",
        "required": ["schema_version", "speakers"],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "speakers": {
                "type": "array",
                "minItems": len(TRAINING_PARTITIONS),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "speaker_id",
                        "consent_status",
                        "consent_time",
                        "recording_condition",
                        "device_profile",
                        "partition",
                    ],
                    "properties": {
                        "speaker_id": {"type": "string", "pattern": "^spk-[a-z0-9]{6,32}$"},
                        "consent_status": {"const": "granted"},
                        "consent_time": {"type": "string", "minLength": 1},
                        "recording_condition": {"type": "string", "minLength": 1},
                        "device_profile": {"type": "string", "minLength": 1},
                        "partition": {"enum": sorted(TRAINING_PARTITIONS)},
                    },
                },
            },
        },
    },
    "derived-manifest.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ME2 external wake derived manifest",
        "type": "object",
        "required": [
            "schema_version",
            "corpus_id",
            "source_manifest_sha256",
            "speakers",
            "records",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "corpus_id": {"type": "string", "minLength": 1},
            "source_manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "speakers": {"type": "array", "minItems": len(TRAINING_PARTITIONS)},
            "records": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
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
                    ],
                },
            },
        },
    },
}


def _inventory(root: Path) -> dict[str, int]:
    files = directories = 0
    for entry in root.iterdir():
        if entry.is_dir():
            directories += 1
        else:
            files += 1
    return {"files": files, "directories": directories}


def bootstrap(root: Path = PRIVATE_ROOT) -> dict[str, Any]:
    """Create static templates without reading or reporting private evidence."""
    resolved = root.expanduser().resolve()
    if resolved != PRIVATE_ROOT:
        raise ValueError("bootstrap root must be the approved external evidence root")
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    inventory_before = _inventory(resolved)
    template_directory = resolved / _TEMPLATE_DIRECTORY
    template_directory.mkdir(mode=0o700, exist_ok=True)
    created = existing = 0
    for name, contents in _TEMPLATES.items():
        target = template_directory / name
        if target.exists():
            existing += 1
            continue
        target.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        target.chmod(0o600)
        created += 1
    return {
        "bootstrap_schema_version": 1,
        "inventory_before": inventory_before,
        "templates_created": created,
        "templates_already_present": existing,
        "root_ready_for_human_consent_registration": True,
    }


def main() -> None:
    print(json.dumps(bootstrap(), sort_keys=True))


if __name__ == "__main__":
    main()
