import hashlib
from pathlib import Path

import numpy as np
import pytest

from me2.app import _require_live_wake_readiness
from me2.audio import AudioConfig, prepare_audio, write_wav
from me2.bundle import save_bundle
from me2.features import FeatureConfig
from me2.model import TinyAudioCNN
from me2.wake_word.calibrate import calibrate_artifact
from me2.wake_word.data import (
    CANONICAL_SYNTHESIS_TEXT,
    CLASS_MAPPING_VERSION,
    COMMAND_NEGATIVE_DATASET_VERSION,
    DATASET_VERSION,
    EXPANDED_DATASET_VERSION,
    FULL_RECIPES,
    PRONUNCIATION_GUIDE,
    audit_manifest,
    build_manifest_record,
    read_manifest,
    trainable_rows,
    unique_recording_path,
    write_manifest,
)
from me2.wake_word.detector import WakeEventDetector, WakeSmoothingPolicy
from me2.wake_word.evaluate import evaluate
from me2.wake_word.train import (
    compatible_dataset_version,
    source_aware_sample_weights,
    validate_augmentation_preserves_wake,
)


def _row(
    path: Path,
    *,
    recording_id: str,
    label: str,
    speaker: str,
    split: str,
    status: str = "accepted",
    parent: str = "",
    augmentation: str = "",
) -> dict[str, str]:
    return build_manifest_record(
        path,
        recording_id=recording_id,
        label=label,
        phrase=PRONUNCIATION_GUIDE if label == "ateko" else "negative",
        synthesis_text="",
        speaker_id=speaker,
        session_id=f"session-{speaker}",
        split=split,
        source="test_recording",
        is_synthetic=False,
        kind="complete_word" if label == "ateko" else "recorded_negative",
        review_status=status,
        review_notes="test",
        parent_recording_id=parent,
        augmentation=augmentation,
        augmentation_parent_id=parent if augmentation else "",
    )


def test_pronunciation_semantics_and_review_gate():
    assert PRONUNCIATION_GUIDE == "a-teh-ko"
    assert DATASET_VERSION == "wake-v3-connected-a-teh-ko"
    assert CANONICAL_SYNTHESIS_TEXT == "a teh ko"
    assert CLASS_MAPPING_VERSION == "wake-five-class-te-means-teh-v2"
    positive_recipes = [
        text for label, text, *_ in FULL_RECIPES if label == "ateko"
    ]
    assert positive_recipes == ["a teh ko"]
    pending = [
        {
            "label": "ateko",
            "review_status": "pending",
            "connected_delivery_status": "pending",
        }
    ]
    with pytest.raises(ValueError, match="No pronunciation-approved"):
        trainable_rows(pending)
    assert trainable_rows(pending, allow_pending_review=True) == pending
    assert compatible_dataset_version(
        [{"dataset_version": EXPANDED_DATASET_VERSION}]
    ) == EXPANDED_DATASET_VERSION
    assert compatible_dataset_version(
        [{"dataset_version": COMMAND_NEGATIVE_DATASET_VERSION}]
    ) == COMMAND_NEGATIVE_DATASET_VERSION
    with pytest.raises(ValueError, match="exactly one supported"):
        compatible_dataset_version(
            [
                {"dataset_version": DATASET_VERSION},
                {"dataset_version": EXPANDED_DATASET_VERSION},
            ]
        )


def test_approved_synthetic_positive_must_use_connected_recipe():
    with pytest.raises(ValueError, match="user-confirmed connected recipe"):
        trainable_rows(
            [
                {
                    "recording_id": "wrong-recipe",
                    "label": "ateko",
                    "review_status": "accepted",
                    "connected_delivery_status": "accepted",
                    "is_synthetic": "true",
                    "synthesis_text": "a-teh-ko",
                }
            ]
        )


def test_manifest_detects_cross_split_duplicates_and_lineage(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    samples = np.sin(np.linspace(0, 20, 4_000, dtype=np.float32))
    write_wav(first, samples)
    write_wav(second, samples)
    rows = [
        _row(
            first,
            recording_id="one",
            label="ateko",
            speaker="speaker-one",
            split="train",
        ),
        _row(
            second,
            recording_id="two",
            label="background",
            speaker="speaker-two",
            split="test",
            parent="parent-two",
            augmentation="noise_mix",
        ),
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    report = audit_manifest(manifest)
    assert report["cross_split_exact_duplicates"]
    assert rows[1]["augmentation_parent_id"] == "parent-two"


def test_manifest_preserves_external_source_provenance(tmp_path):
    audio = tmp_path / "negative.wav"
    write_wav(audio, np.zeros(2_000, dtype=np.float32))
    row = build_manifest_record(
        audio,
        recording_id="external-negative",
        label="background",
        phrase="turn on the light",
        synthesis_text="",
        speaker_id="hf-command:speaker",
        session_id="hf-command-speaker",
        split="train",
        source="hf_me2_command_without_wake",
        is_synthetic=False,
        kind="command_without_wake",
        review_status="pending",
        review_notes="not listened",
        source_dataset_id="example/dataset",
        source_dataset_revision="abc123",
        source_record_id="clip.wav",
        source_license="upstream terms apply",
    )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [row])
    loaded = read_manifest(manifest)[0]
    assert loaded["source_dataset_id"] == "example/dataset"
    assert loaded["source_dataset_revision"] == "abc123"
    assert loaded["source_record_id"] == "clip.wav"


def test_manifest_detects_speaker_and_parent_split_leakage(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.wav"
        write_wav(path, np.full(2_000, 0.1 + index * 0.01, dtype=np.float32))
        paths.append(path)
    rows = [
        _row(
            paths[0],
            recording_id="left",
            label="ateko",
            speaker="same-speaker",
            split="train",
            parent="same-parent",
        ),
        _row(
            paths[1],
            recording_id="right",
            label="background",
            speaker="same-speaker",
            split="test",
            parent="same-parent",
        ),
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    report = audit_manifest(manifest)
    assert "same-speaker" in report["speaker_split_leakage"]
    assert "same-parent" in report["parent_split_leakage"]


def test_recorder_path_never_overwrites(tmp_path):
    path = unique_recording_path(tmp_path, "speaker", "session", "abc123")
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        unique_recording_path(tmp_path, "speaker", "session", "abc123")


def test_complete_word_shorter_than_window_is_not_cropped():
    samples = np.ones(28_800, dtype=np.float32) * 0.1
    prepared = prepare_audio(
        samples,
        16_000,
        AudioConfig(duration_seconds=2.0),
    )
    assert prepared.shape == (32_000,)
    assert np.count_nonzero(prepared) == samples.size


def test_wake_smoothing_requires_hits_and_resets_after_event():
    detector = WakeEventDetector(
        WakeSmoothingPolicy(threshold=0.7, smoothing_window=3, required_hits=2)
    )
    assert not detector.update("ateko", 0.8)
    assert not detector.update("background", 0.9)
    assert detector.update("ateko", 0.85)
    assert not detector.update("ateko", 0.9)


def test_source_aware_sampling_balances_wake_and_non_wake_mass():
    rows = [
        {
            "label": "ateko" if index < 2 else "background",
            "speaker_id": f"speaker-{index % 2}",
            "source": "recording",
            "kind": "complete_word" if index < 2 else "recorded_negative",
        }
        for index in range(10)
    ]
    weights = source_aware_sample_weights(rows, "binary")
    wake_mass = sum(weights[:2])
    non_wake_mass = sum(weights[2:])
    assert wake_mass == pytest.approx(non_wake_mass)


def test_wake_augmentation_refuses_possible_complete_word_crop():
    with pytest.raises(ValueError, match="could crop a complete keyword"):
        validate_augmentation_preserves_wake(
            [
                {
                    "recording_id": "long-wake",
                    "label": "ateko",
                    "duration_seconds": "1.95",
                }
            ],
            AudioConfig(duration_seconds=2.0),
            {
                "enabled": True,
                "speed_min": 0.96,
                "max_shift_ms": 60.0,
            },
        )


def test_live_use_rejects_explicitly_unverified_wake_artifact():
    with pytest.raises(ValueError, match="deployment_ready=false"):
        _require_live_wake_readiness(
            {"deployment_ready": False},
            allow_unverified=False,
        )
    _require_live_wake_readiness(
        {"deployment_ready": False},
        allow_unverified=True,
    )


def test_evaluation_is_read_only_and_checks_dataset_version(tmp_path):
    positive = tmp_path / "positive.wav"
    negative = tmp_path / "negative.wav"
    write_wav(positive, np.sin(np.linspace(0, 100, 8_000)).astype(np.float32))
    write_wav(negative, np.zeros(8_000, dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            _row(
                positive,
                recording_id="positive",
                label="ateko",
                speaker="speaker-one",
                split="test",
            ),
            _row(
                negative,
                recording_id="negative",
                label="background",
                speaker="speaker-two",
                split="test",
            ),
        ],
    )
    model = TinyAudioCNN(num_classes=2, width=4)
    model_path = tmp_path / "wake.me2"
    save_bundle(
        model_path,
        {
            "task": "wake_word",
            "architecture": "TinyAudioCNN",
            "model_version": "test",
            "model_width": 4,
            "labels": ["non_wake", "ateko"],
            "label_mode": "binary",
            "pronunciation_guide": PRONUNCIATION_GUIDE,
            "wake_dataset_version": DATASET_VERSION,
            "threshold": 0.7,
            "audio": AudioConfig(duration_seconds=1.0).__dict__,
            "features": FeatureConfig().__dict__,
            "input_shape": [1, 40, 101],
            "state_dict": model.state_dict(),
        },
    )
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    evaluate(model_path, manifest, "test", tmp_path / "results")
    after = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert before == after

    rows = manifest.read_text(encoding="utf-8").replace(
        DATASET_VERSION, "incompatible-version"
    )
    manifest.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match="dataset/model mismatch"):
        evaluate(model_path, manifest, "test", tmp_path / "mismatch")


def test_calibration_prefers_lower_threshold_when_operating_point_ties(
    tmp_path,
    monkeypatch,
):
    model = TinyAudioCNN(num_classes=2, width=4)
    model_path = tmp_path / "wake.me2"
    bundle = {
        "task": "wake_word",
        "architecture": "TinyAudioCNN",
        "model_version": "test",
        "model_width": 4,
        "labels": ["non_wake", "ateko"],
        "label_mode": "binary",
        "pronunciation_guide": PRONUNCIATION_GUIDE,
        "wake_dataset_version": DATASET_VERSION,
        "threshold": 0.7,
        "audio": AudioConfig(duration_seconds=1.0).__dict__,
        "features": FeatureConfig().__dict__,
        "input_shape": [1, 40, 101],
        "state_dict": model.state_dict(),
    }
    save_bundle(model_path, bundle)

    def fake_score_rows(*args, **kwargs):
        return (
            [{"recording_id": str(index)} for index in range(4)],
            np.asarray([1, 1, 0, 0]),
            np.asarray(
                [
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.9, 0.1],
                    [0.8, 0.2],
                ]
            ),
            {**bundle, "state_dict": model.state_dict()},
        )

    monkeypatch.setattr(
        "me2.wake_word.calibrate.score_rows",
        fake_score_rows,
    )
    report = calibrate_artifact(
        model_path,
        tmp_path / "unused.csv",
        target_false_accept_rate=0.0,
    )
    assert report["threshold"] == pytest.approx(0.1)
