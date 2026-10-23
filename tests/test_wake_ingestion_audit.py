import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from me2.wake_word import ingestion_audit


def _record(path, *, index, speaker, partition, label):
    sf.write(path, np.full(16_000, 0.1 + index * 0.01), 16_000, subtype="PCM_16")
    return {
        "recording_id": f"record-{index}",
        "source_recording_id": f"source-{index}",
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
    for index, partition in enumerate(("train", "validation", "held_out")):
        speaker = f"spk-{index + 123456}"
        speakers.append({
            "speaker_id": speaker,
            "consent_status": "granted",
            "consent_time": "2026-10-24T00:00:00Z",
            "recording_condition": "quiet-room",
            "device_profile": "pi-usb-microphone",
            "partition": partition,
        })
        records.append(_record(root / f"{index}.wav", index=index, speaker=speaker, partition=partition, label="ateko" if index == 0 else "background"))
    return {
        "schema_version": 1,
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
    assert report["partitions"] == {"held_out": 1, "train": 1, "validation": 1}
    assert "path" not in json.dumps(report)
    assert "record-" not in json.dumps(report)


def test_external_ingestion_audit_rejects_partition_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_audit, "PRIVATE_ROOT", tmp_path.resolve())
    manifest = _manifest(tmp_path)
    manifest["records"][1]["speaker_id"] = manifest["speakers"][0]["speaker_id"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = ingestion_audit.audit_manifest(path)
    assert not report["valid_for_training_proposal"]
    assert "speaker_partition_mismatch" in report["issues"]


def test_external_ingestion_audit_refuses_non_external_manifest(tmp_path):
    with pytest.raises(ValueError, match="approved evidence root"):
        ingestion_audit.audit_manifest(tmp_path / "manifest.json")
