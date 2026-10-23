import numpy as np
import soundfile as sf

from me2.wake_word import private_calibration


def test_private_calibration_requires_five_valid_samples_and_four_detections(tmp_path, monkeypatch):
    monkeypatch.setattr(private_calibration, "PRIVATE_ROOT", tmp_path.resolve())
    samples = []
    for index in range(5):
        path = tmp_path / f"sample-{index}.wav"
        sf.write(path, np.full(16_000, 0.1, dtype=np.float32), 16_000)
        samples.append(path)

    monkeypatch.setattr(private_calibration, "predict_waveform", lambda *_: ("ateko", .9, []))
    report = private_calibration.score_samples(object(), {"audio": {}, "threshold": .7}, samples)
    assert report["passed"]
    assert report["full_detections"] == 5
    assert report["training_permitted"] is False


def test_private_calibration_refuses_non_private_paths(tmp_path):
    with np.testing.assert_raises(ValueError):
        private_calibration._private_path(tmp_path / "outside.wav")


def test_private_calibration_rejects_non_wake_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(private_calibration, "PRIVATE_ROOT", tmp_path.resolve())
    monkeypatch.setattr(private_calibration, "load_bundle", lambda _path: (object(), {"task": "vcm"}))
    with np.testing.assert_raises_regex(ValueError, "wake_word"):
        private_calibration.main([
            "--model", "ignored.me2", "--sample", str(tmp_path / "a.wav"),
            "--output", str(tmp_path / "summary.json"),
        ])
