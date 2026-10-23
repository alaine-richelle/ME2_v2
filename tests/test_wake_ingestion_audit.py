import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from me2.wake_word import ingestion_audit


def _record(path, *, index, speaker, partition, label):
    sf.write(path, np.full(16_000, 0.1 + index * 0.01), 16_000, subtype="PCM_16")
    return {
        "recording_id": f"rec-{index + 100000}",
        "source_recording_id": f"src-{index + 100000}",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "speaker_id": speaker,
        "partition": partition,
        "source_partition": partition,
        "label": label,
        "kind": "complete_word" if label == "ateko" else "recorded_negative",
        "review_status": "accepted",
        "connected_delivery_status": "accepted" if label == "ateko" else "not_applicable",
    }


def _manifest(root):
    speakers = []
    records = []
    record_index = 0
    for speaker_index, partition in enumerate(("train", "validation", "calibration", "test")):
        speaker = f"spk-{speaker_index + 123456}"
        speakers.append({
            "speaker_id": speaker,
            "consent_status": "granted",
            "consent_time": "2026-10-24T00:00:00Z",
            "recording_condition": "quiet-room",
            "device_profile": "pi-usb-microphone",
            "partition": partition,
        })
        for label in ("ateko", "background", "a", "te", "ko"):
            records.append(
                _record(
                    root / f"{record_index}-{label}.wav",
                    index=record_index,
                    speaker=speaker,
                    partition=partition,
                    label=label,
                )
            )
            record_index += 1
    return {
        "schema_version": 2,
        "corpus_id": "wake-pilot-v6-derived",
        "source_manifest_sha256": "a" * 64,
        "speakers": speakers,
        "records": records,
    }


def test_external_ingestion_audit_reports_only_aggregate_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = ingestion_audit.audit_manifest(path)
    assert report["valid_for_training_proposal"]
    assert report["partitions"] == {"calibration": 5, "test": 5, "train": 5, "validation": 5}
    assert report["labels_by_partition"]["train"] == {
        "a": 1, "ateko": 1, "background": 1, "ko": 1, "te": 1
    }
    assert "path" not in json.dumps(report)
    assert "rec-" not in json.dumps(report)


def test_external_ingestion_audit_rejects_partition_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    manifest = _manifest(tmp_path)
    manifest["records"][5]["speaker_id"] = manifest["speakers"][0]["speaker_id"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = ingestion_audit.audit_manifest(path)
    assert not report["valid_for_training_proposal"]
    assert "speaker_partition_mismatch" in report["issues"]


def test_external_ingestion_audit_rejects_non_partitioned_source_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    manifest = _manifest(tmp_path)
    manifest["records"][0]["source_partition"] = "unassigned"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = ingestion_audit.audit_manifest(path)
    assert not report["valid_for_training_proposal"]
    assert "unsupported_source_partition" in report["issues"]


def test_external_ingestion_audit_rejects_non_opaque_recording_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    manifest = _manifest(tmp_path)
    manifest["records"][0]["recording_id"] = "alice-wake-01"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = ingestion_audit.audit_manifest(path)
    assert not report["valid_for_training_proposal"]
    assert "non_opaque_recording_identifier" in report["issues"]


def test_external_ingestion_audit_writes_a_new_aggregate_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "audit-report.json"
    ingestion_audit.main(["--manifest", str(path), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid_for_training_proposal"]
    assert "path" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="new .json"):
        ingestion_audit.main(["--manifest", str(path), "--output", str(output)])


def test_external_ingestion_audit_refuses_non_external_manifest(tmp_path):
    with pytest.raises(ValueError, match="approved evidence root"):
        ingestion_audit.audit_manifest(tmp_path / "manifest.json")
