"""Produce a non-executing private wake-training readiness proposal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from me2.wake_word.corpus_consent import TRAINING_PARTITIONS
from me2.wake_word.data import CANONICAL_WAKE_LABEL, CLASSES
from me2.wake_word.ingestion_audit import _private_path


def _require_nonnegative_counts(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(count, int) or count < 0
        for key, count in value.items()
    ):
        raise ValueError(f"{field} must contain non-negative aggregate counts")
    return value


def review_audit_report(audit_report_path: Path, label_mode: str) -> dict[str, Any]:
    """Validate aggregate evidence; never authorizes or invokes training."""
    path = _private_path(audit_report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("audit_schema_version") != 2:
        raise ValueError("audit report must use audit_schema_version=2")
    if report.get("valid_for_training_proposal") is not True or report.get("issues") != []:
        raise ValueError("audit report is not a passing ingestion gate")

    partitions = _require_nonnegative_counts(report.get("partitions"), "partitions")
    if set(partitions) != TRAINING_PARTITIONS or any(count == 0 for count in partitions.values()):
        raise ValueError("audit report lacks a non-empty frozen training partition")
    labels_by_partition = report.get("labels_by_partition")
    if not isinstance(labels_by_partition, dict) or set(labels_by_partition) != TRAINING_PARTITIONS:
        raise ValueError("audit report lacks label coverage by frozen partition")

    required_labels = set(CLASSES) if label_mode == "five_class" else {CANONICAL_WAKE_LABEL, "non_wake"}
    for partition in TRAINING_PARTITIONS:
        labels = _require_nonnegative_counts(
            labels_by_partition[partition], f"labels_by_partition[{partition}]"
        )
        observed = set(labels)
        if label_mode == "binary":
            observed = ({CANONICAL_WAKE_LABEL} if labels.get(CANONICAL_WAKE_LABEL, 0) else set()) | (
                {"non_wake"} if sum(count for label, count in labels.items() if label != CANONICAL_WAKE_LABEL) else set()
            )
        if required_labels - observed:
            raise ValueError(f"frozen partition {partition!r} lacks required label coverage")

    return {
        "proposal_schema_version": 1,
        "audit_schema_version": 2,
        "label_mode": label_mode,
        "partitions": {name: partitions[name] for name in sorted(TRAINING_PARTITIONS)},
        "labels_by_partition": {
            name: labels_by_partition[name] for name in sorted(TRAINING_PARTITIONS)
        },
        "ready_for_private_manifest_preparation": True,
        "training_authorized": False,
        "next_gate": "manual review of this aggregate proposal and explicit training authorization",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--label-mode", choices=("five_class", "binary"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = _private_path(args.output)
    audit_report = _private_path(args.audit_report)
    if output == audit_report or output.exists() or output.suffix.casefold() != ".json":
        raise ValueError("output must be a new .json file distinct from the audit report")
    proposal = review_audit_report(audit_report, args.label_mode)
    output.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(proposal, sort_keys=True))


if __name__ == "__main__":
    main()
