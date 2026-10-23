import pytest

from me2.wake_word.corpus_consent import validate_manifest


def _speaker(index: int, partition: str) -> dict[str, str]:
    return {
        "speaker_id": f"spk-{index + 123456}",
        "consent_status": "granted",
        "consent_time": "2026-10-24T00:00:00Z",
        "recording_condition": "quiet-room",
        "device_profile": "pi-usb-microphone",
        "partition": partition,
    }


def test_consent_manifest_requires_four_disjoint_training_partitions():
    manifest = {
        "schema_version": 2,
        "speakers": [
            _speaker(index, partition)
            for index, partition in enumerate(("train", "validation", "calibration", "test"))
        ],
    }
    report = validate_manifest(manifest)
    assert report["recording_enabled"]
    assert report["partitions"] == ["calibration", "test", "train", "validation"]


def test_consent_manifest_rejects_held_out_alias_without_calibration():
    manifest = {
        "schema_version": 2,
        "speakers": [
            _speaker(index, partition)
            for index, partition in enumerate(("train", "validation", "held_out", "test"))
        ],
    }
    with pytest.raises(ValueError, match="invalid speaker partition"):
        validate_manifest(manifest)
