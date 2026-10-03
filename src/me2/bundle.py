"""Canonical inference bundle serialization."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from me2.model import TinyAudioCNN


def save_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_bundle(path: str | Path) -> tuple[TinyAudioCNN, dict[str, Any]]:
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    model = TinyAudioCNN(
        num_classes=len(bundle["labels"]), width=int(bundle.get("model_width", 16))
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def predict_waveform(model: TinyAudioCNN, bundle: dict[str, Any], waveform) -> tuple[str, float, list[float]]:
    from me2.features import FeatureConfig, LogMel

    config = FeatureConfig(**bundle["features"])
    with torch.inference_mode():
        logits = model(LogMel(config)(waveform))
        probabilities = torch.softmax(logits, dim=1)[0]
    index = int(probabilities.argmax())
    return bundle["labels"][index], float(probabilities[index]), probabilities.tolist()
