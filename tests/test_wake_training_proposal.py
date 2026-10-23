import json

import pytest

from me2.wake_word import ingestion_audit, training_proposal
from tests.test_wake_ingestion_audit import _manifest


def _audit_report(tmp_path):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_path = tmp_path / "audit-report.json"
    report_path.write_text(json.dumps(ingestion_audit.audit_manifest(manifest_path)), encoding="utf-8")
    return report_path


def test_training_proposal_requires_passing_four_way_label_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    monkeypatch.setattr(training_proposal, "_private_path", ingestion_audit._private_path)
    report_path = _audit_report(tmp_path)
    proposal = training_proposal.review_audit_report(report_path, "five_class")
    assert proposal["ready_for_private_manifest_preparation"]
    assert not proposal["training_authorized"]


def test_training_proposal_refuses_missing_partition_label_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    monkeypatch.setattr(training_proposal, "_private_path", ingestion_audit._private_path)
    report_path = _audit_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["labels_by_partition"]["test"]["ateko"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks required label coverage"):
        training_proposal.review_audit_report(report_path, "five_class")
