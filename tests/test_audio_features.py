import numpy as np
import torch

from me2.app import build_parser
from me2.audio import AudioConfig, load_wav, prepare_audio, write_wav
from me2.bundle import load_bundle, save_bundle
from me2.features import FeatureConfig, LogMel
from me2.model import TinyAudioCNN, parameter_count
from me2.training import _augment_batch
from me2.vcm.data import split_train_validation


def test_prepare_audio_resamples_mono_and_fixes_duration():
    source = np.zeros((8_000, 2), dtype=np.float32)
    source[1_000:7_000] = 0.25
    prepared = prepare_audio(source, 8_000, AudioConfig(duration_seconds=1.0))
    assert prepared.shape == (16_000,)
    assert prepared.dtype == np.float32
    assert np.max(np.abs(prepared)) <= 1.0


def test_quiet_background_is_not_peak_amplified():
    source = np.full(8_000, 1e-4, dtype=np.float32)
    prepared = prepare_audio(source, 8_000, AudioConfig(duration_seconds=1.0))
    assert float(np.max(np.abs(prepared))) < 1e-3


def test_wav_round_trip(tmp_path):
    path = tmp_path / "sample.wav"
    source = np.sin(np.linspace(0, 30, 8_000, dtype=np.float32)) * 0.2
    write_wav(path, source, 8_000)
    prepared = load_wav(path, AudioConfig(duration_seconds=1.0))
    assert prepared.shape == (16_000,)
    assert np.isfinite(prepared).all()


def test_logmel_shape_and_cnn_forward():
    config = FeatureConfig()
    features = LogMel(config)(np.zeros(16_000, dtype=np.float32))
    assert features.shape == (1, 1, 40, 101)
    model = TinyAudioCNN(num_classes=3)
    logits = model(features)
    assert logits.shape == (1, 3)
    assert parameter_count(model) < 100_000
    assert torch.isfinite(logits).all()


def test_training_augmentation_is_reproducible_and_shape_preserving():
    waveforms = torch.zeros((3, 8_000))
    waveforms[:, 2_000:6_000] = 0.1
    first = _augment_batch(waveforms, 16_000, torch.Generator().manual_seed(19))
    second = _augment_batch(waveforms, 16_000, torch.Generator().manual_seed(19))
    assert first.shape == waveforms.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, waveforms)


def test_vcm_validation_partition_is_speaker_disjoint():
    class FakeSplit:
        def __init__(self):
            self.column_names = ["speaker_id"]
            self.rows = [{"speaker_id": f"speaker-{index // 3}"} for index in range(30)]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                return [row[key] for row in self.rows]
            return self.rows[key]

    train, validation = split_train_validation(FakeSplit(), 0.2, 42)
    speakers = FakeSplit()["speaker_id"]
    assert {speakers[index] for index in train}.isdisjoint(
        {speakers[index] for index in validation}
    )
    assert train and validation


def test_bundle_round_trip_and_cli_modes(tmp_path):
    model = TinyAudioCNN(num_classes=2, width=12)
    bundle = {
        "model_width": 12,
        "labels": ["background", "ateko"],
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "model.pkl"
    save_bundle(path, bundle)
    loaded_model, loaded_bundle = load_bundle(path)
    assert loaded_bundle["labels"] == bundle["labels"]
    assert parameter_count(loaded_model) == parameter_count(model)
    assert build_parser().parse_args(["--interactive", "--wake-audio", "wake.wav"]).interactive
