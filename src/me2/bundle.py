"""Versioned, validated model artifacts with an explicit legacy migration path."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from me2.audio import AudioConfig
from me2.contracts import finite, integer
from me2.features import FeatureConfig
from me2.model import TinyAudioCNN
from me2.vcm.ctc import (
    CHARSET,
    LEGACY_CHARSET,
    AcceptancePolicy,
    CharacterAudioCNN,
    CommandResult,
    CompactCTCModel,
    DecoderBudgetError,
    DecoderLimits,
    InferenceInputError,
    coerce_grammar_entries,
    decode_command,
)

BUNDLE_SCHEMA_VERSION = 2
METADATA_MEMBER = "metadata.json"
WEIGHTS_MEMBER = "weights.npz"
ALLOWED_MEMBERS = {METADATA_MEMBER, WEIGHTS_MEMBER}
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_WEIGHTS_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_WEIGHTS_BYTES = 256 * 1024 * 1024


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON metadata key")
        result[key] = value
    return result


def _validate_weight_archive(weights_bytes: bytes) -> None:
    """Bound the inner NPZ, including declared NPY dimensions, before allocation."""
    with zipfile.ZipFile(io.BytesIO(weights_bytes)) as archive:
        items = archive.infolist()
        if not items or len(items) > 4096 or len({i.filename for i in items}) != len(items):
            raise ValueError("Invalid or duplicate tensor archive members")
        if sum(i.file_size for i in items) > MAX_EXPANDED_WEIGHTS_BYTES:
            raise ValueError("Expanded tensor archive exceeds runtime budget")
        for item in items:
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported tensor array format")
                if dtype.kind not in "fiub" or len(shape) > 8:
                    raise ValueError("Tensor must have bounded numeric dimensions")
                count = 1
                for dimension in shape:
                    count *= integer("tensor dimension", dimension, 0, MAX_EXPANDED_WEIGHTS_BYTES)
                if count * dtype.itemsize > item.file_size:
                    raise ValueError("Tensor dimensions exceed stored payload")


def save_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    """Atomically save metadata plus plain tensor arrays without object pickle."""

    destination = Path(path)
    if destination.suffix == ".pkl":
        raise ValueError(
            "New artifacts must not use .pkl; choose a .me2 path and migrate "
            "trusted legacy files explicitly"
        )
    state_dict = bundle.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Bundle requires a non-empty state_dict")
    metadata = {key: value for key, value in bundle.items() if key != "state_dict"}
    metadata["schema_version"] = BUNDLE_SCHEMA_VERSION
    _validate_metadata(metadata)

    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
    }
    if len(arrays) != len(state_dict):
        raise TypeError("state_dict may contain only torch.Tensor values")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Artifact tensors contain non-finite values")
    weights_buffer = io.BytesIO()
    np.savez_compressed(weights_buffer, **arrays)
    weights_bytes = weights_buffer.getvalue()
    metadata["weights_sha256"] = hashlib.sha256(weights_bytes).hexdigest()
    metadata_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling avoids concurrent writes sharing or deleting one temp file.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.writestr(METADATA_MEMBER, metadata_bytes)
            archive.writestr(WEIGHTS_MEMBER, weights_bytes)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=16)
def _cached_frontend(feature_items: tuple[tuple[str, Any], ...]):
    from me2.features import FeatureConfig, LogMel

    return LogMel(FeatureConfig(**dict(feature_items)))


def _bundle_frontend(bundle: dict[str, Any]):
    return _cached_frontend(tuple(sorted(bundle["features"].items())))


def load_bundle(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Load only the safe v2 archive format."""

    source = Path(path)
    if source.suffix == ".pkl":
        raise ValueError(
            f"Refusing unrestricted pickle artifact {source}. Run "
            "`python -m me2.bundle migrate --trusted-legacy ...` after verifying "
            "the file's provenance."
        )
    try:
        with zipfile.ZipFile(source) as archive:
            items = archive.infolist()
            members = set(archive.namelist())
            if members != ALLOWED_MEMBERS or len(items) != len(ALLOWED_MEMBERS):
                raise ValueError(
                    "Unexpected, missing, or duplicate artifact members"
                )
            if archive.getinfo(METADATA_MEMBER).file_size > MAX_METADATA_BYTES:
                raise ValueError("Artifact metadata exceeds runtime budget")
            if archive.getinfo(WEIGHTS_MEMBER).file_size > MAX_WEIGHTS_BYTES:
                raise ValueError("Artifact weights exceed runtime budget")
            metadata_bytes = archive.read(METADATA_MEMBER)
            weights_bytes = archive.read(WEIGHTS_MEMBER)
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"Invalid ME2 model artifact {source}: {error}") from error
    try:
        metadata = json.loads(metadata_bytes, object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Artifact metadata is not valid UTF-8 JSON") from error
    _validate_metadata(metadata)
    observed_hash = hashlib.sha256(weights_bytes).hexdigest()
    if observed_hash != metadata.get("weights_sha256"):
        raise ValueError("Artifact weight hash mismatch")
    _validate_weight_archive(weights_bytes)
    with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as arrays:
        state_dict = {
            name: torch.from_numpy(np.array(arrays[name], copy=True))
            for name in arrays.files
        }
    if any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
        raise ValueError("Artifact tensors contain non-finite values")
    model = _build_model(metadata)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Artifact weights are incompatible with metadata: {error}") from error
    model.eval()
    bundle = dict(metadata)
    bundle["state_dict"] = state_dict
    return model, bundle


def load_legacy_bundle(
    path: str | Path,
    *,
    trusted: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a legacy pickle only after the caller explicitly marks it trusted."""

    if not trusted:
        raise ValueError("Legacy pickle loading requires trusted=True")
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if not isinstance(bundle, dict):
        raise TypeError("Legacy bundle must be a dictionary")
    model = _build_model(bundle)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model, bundle


def migrate_legacy_bundle(source: str | Path, destination: str | Path) -> Path:
    """Migrate one locally verified pickle while preserving the original."""

    _, legacy = load_legacy_bundle(source, trusted=True)
    migrated = dict(legacy)
    architecture = migrated.get("architecture", "TinyAudioCNN")
    if architecture == "CharacterAudioCNN":
        migrated.setdefault("charset", LEGACY_CHARSET)
        migrated.setdefault("blank_id", len(LEGACY_CHARSET))
        migrated.setdefault("model_version", "legacy-character-audio-cnn-v1")
        migrated.setdefault("grammar_version", "legacy-training-transcripts")
        migrated.setdefault(
            "acceptance_policy",
            {
                "min_normalized_log_score": -1.0e30,
                "min_margin": -1.0e30,
                "min_candidate_posterior": 0.0,
                "version": "legacy-uncalibrated",
            },
        )
    else:
        migrated.setdefault("model_version", f"legacy-{architecture.casefold()}-v1")
    save_bundle(destination, migrated)
    return Path(destination)


def predict_command(
    model: nn.Module,
    bundle: dict[str, Any],
    waveform: np.ndarray | torch.Tensor,
    *,
    sample_length: int | None = None,
) -> CommandResult:
    started = time.perf_counter()
    try:
        _validate_waveform(waveform, bundle, sample_length)
    except ValueError as error:
        raise InferenceInputError("empty_invalid_audio_input", str(error)) from error
    frontend = _bundle_frontend(bundle)
    with torch.inference_mode():
        try:
            features = frontend(waveform)
        except ValueError as error:
            raise InferenceInputError("empty_invalid_feature_input", str(error)) from error
        frontend_ms = (time.perf_counter() - started) * 1000.0
        architecture = bundle.get("architecture", "TinyAudioCNN")
        if architecture in {"CharacterAudioCNN", "CompactCTCModel"}:
            if sample_length is None:
                sample_length = int(torch.as_tensor(waveform).shape[-1])
            feature_lengths = frontend.frame_lengths(
                torch.tensor([sample_length], dtype=torch.long)
            )
            output_lengths = model.output_lengths(feature_lengths)
            model_started = time.perf_counter()
            logits = model(features, feature_lengths).squeeze(0)
            frame_diagnostics = {
                "usable_samples": sample_length,
                "padded_samples": int(torch.as_tensor(waveform).shape[-1]),
                "feature_frames": features.shape[-1],
                "valid_feature_frames": int(feature_lengths[0]),
                "output_frames": logits.shape[0],
                "valid_output_frames": int(output_lengths[0]),
            }
            model_ms = (time.perf_counter() - model_started) * 1000.0
            decode_started = time.perf_counter()
            try:
                result = decode_command(
                    logits, bundle["grammar"],
                    charset=str(bundle.get("charset", LEGACY_CHARSET)),
                    blank_id=int(bundle.get("blank_id", len(str(bundle.get("charset", LEGACY_CHARSET))))),
                    beam_width=int(bundle.get("beam_width", 16)),
                    input_length=int(output_lengths[0]),
                    acceptance_policy=bundle.get("acceptance_policy"),
                    model_version=str(bundle.get("model_version", "")),
                    grammar_version=str(bundle.get("grammar_version", "")),
                    limits=DecoderLimits(**bundle.get("decoder_limits", {})),
                )
            except (InferenceInputError, DecoderBudgetError) as error:
                error.diagnostics = {**frame_diagnostics, **error.diagnostics}
                raise
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            return CommandResult(
                **{
                    **result.__dict__,
                    "diagnostics": {**frame_diagnostics, **result.diagnostics},
                    "stage_timings_ms": {
                        "frontend": frontend_ms,
                        "acoustic_model": model_ms,
                        "decoder_and_rejection": decode_ms,
                    },
                }
            )

        model_started = time.perf_counter()
        logits = model(features)
        _validate_classifier_logits(logits, bundle)
        probabilities = torch.softmax(logits, dim=1)[0]
        model_ms = (time.perf_counter() - model_started) * 1000.0
    index = int(probabilities.argmax())
    label = bundle["labels"][index]
    accepted = label != "out_of_scope"
    return CommandResult(
        decoded_text="",
        intent=label if accepted else None,
        slots={},
        variation="",
        raw_log_score=float(torch.log(probabilities[index].clamp_min(1e-12))),
        normalized_log_score=float(torch.log(probabilities[index].clamp_min(1e-12))),
        candidate_posterior=float(probabilities[index]),
        margin=0.0,
        accepted=accepted,
        rejection_reason=None if accepted else "classifier_out_of_scope",
        model_version=str(bundle.get("model_version", "")),
        grammar_version="",
        stage_timings_ms={"frontend": frontend_ms, "acoustic_model": model_ms},
    )


def predict_waveform(
    model: nn.Module,
    bundle: dict[str, Any],
    waveform: np.ndarray | torch.Tensor,
) -> tuple[str, float, list[float]]:
    """Compatibility adapter used by the wake classifier and older callers."""

    architecture = bundle.get("architecture", "TinyAudioCNN")
    if architecture in {"CharacterAudioCNN", "CompactCTCModel"}:
        result = predict_command(model, bundle, waveform)
        label = result.actionable_intent or "out_of_scope"
        labels = bundle["labels"]
        probabilities = [0.0] * len(labels)
        if label in labels:
            probabilities[labels.index(label)] = result.candidate_posterior
        if "out_of_scope" in labels and not result.accepted:
            probabilities[labels.index("out_of_scope")] = 1.0
        total = sum(probabilities)
        if total <= 0:
            probabilities[labels.index(label)] = 1.0
            total = 1.0
        probabilities = [value / total for value in probabilities]
        return label, max(probabilities), probabilities

    _validate_waveform(waveform, bundle, None)
    with torch.inference_mode():
        logits = model(_bundle_frontend(bundle)(waveform))
        _validate_classifier_logits(logits, bundle)
        probabilities = torch.softmax(logits, dim=1)[0]
    index = int(probabilities.argmax())
    return (
        bundle["labels"][index],
        float(probabilities[index]),
        probabilities.tolist(),
    )


def _validate_waveform(waveform, bundle, sample_length) -> None:
    tensor = torch.as_tensor(waveform)
    if tensor.ndim != 1:
        raise ValueError("Prediction requires one mono [samples] waveform")
    maximum = round(bundle["audio"]["sample_rate"] * bundle["audio"]["duration_seconds"])
    integer("waveform samples", tensor.numel(), bundle["features"]["n_fft"] // 2 + 1, maximum)
    if sample_length is not None:
        integer("sample_length", sample_length, 1, tensor.numel())
    if not torch.isfinite(tensor).all():
        raise ValueError("Prediction audio contains non-finite values")


def _validate_classifier_logits(logits, bundle) -> None:
    if logits.shape != (1, len(bundle["labels"])) or not torch.isfinite(logits).all():
        raise ValueError("Classifier logits have invalid shape or non-finite values")


def _build_model(metadata: dict[str, Any]) -> nn.Module:
    architecture = metadata.get("architecture", "TinyAudioCNN")
    if architecture == "CompactCTCModel":
        model_config = metadata.get("model_config", {})
        return CompactCTCModel(
            n_mels=int(metadata["features"]["n_mels"]),
            width=int(model_config.get("width", 48)),
            projection_size=int(model_config.get("projection_size", 128)),
            hidden_size=int(model_config.get("hidden_size", 96)),
            output_symbols=len(str(metadata.get("charset", CHARSET))) + 1,
        )
    if architecture == "CharacterAudioCNN":
        charset = str(metadata.get("charset", LEGACY_CHARSET))
        return CharacterAudioCNN(
            n_mels=int(metadata["features"]["n_mels"]),
            hidden_size=int(metadata.get("hidden_size", 256)),
            output_symbols=len(charset) + 1,
        )
    if architecture == "TinyAudioCNN":
        return TinyAudioCNN(
            num_classes=len(metadata["labels"]),
            width=int(metadata.get("model_width", 16)),
            dropout=float(metadata.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model architecture {architecture!r}")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("Artifact metadata must be an object")
    schema_version = metadata.get("schema_version")
    integer("schema_version", schema_version, BUNDLE_SCHEMA_VERSION, BUNDLE_SCHEMA_VERSION)
    if schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema {schema_version}; expected {BUNDLE_SCHEMA_VERSION}"
        )
    if metadata.get("task") not in {"vcm", "wake_word"}:
        raise ValueError(f"Unsupported bundle task {metadata.get('task')!r}")
    labels = metadata.get("labels")
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 256
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Bundle labels must be a non-empty unique list")
    for section in ("audio", "features"):
        if not isinstance(metadata.get(section), dict):
            raise TypeError(f"Bundle requires {section} configuration")
    audio = AudioConfig(**metadata["audio"])
    features = FeatureConfig(**metadata["features"])
    if audio.sample_rate != features.sample_rate:
        raise ValueError("Audio/frontend sample rates must match")
    if audio.sample_count <= features.n_fft // 2:
        raise ValueError("Artifact duration is too short for the frontend")
    feature_frames = audio.sample_count // features.hop_length + 1
    if feature_frames > 4096 or feature_frames * features.n_fft > 8_388_608:
        raise ValueError("Artifact frontend exceeds runtime budget")
    architecture = metadata.get("architecture", "TinyAudioCNN")
    if architecture not in {"TinyAudioCNN", "CharacterAudioCNN", "CompactCTCModel"}:
        raise ValueError("Unsupported model architecture")
    if metadata["task"] == "wake_word":
        if architecture != "TinyAudioCNN" or "ateko" not in labels:
            raise ValueError("Wake artifacts require a classifier and the complete ateko label")
        finite("wake threshold", metadata.get("threshold", 0.7), 0.0, 1.0)
        from me2.wake_word.detector import WakeSmoothingPolicy

        WakeSmoothingPolicy(
            threshold=metadata.get("threshold", 0.7),
            smoothing_window=metadata.get("streaming_policy", {}).get("smoothing_window", 1),
            required_hits=metadata.get("streaming_policy", {}).get("required_hits", 1),
        )
        if "deployment_ready" in metadata and not isinstance(metadata["deployment_ready"], bool):
            raise ValueError("deployment_ready must be boolean")
    integer("model_width", metadata.get("model_width", 16), 1, 128)
    finite("dropout", metadata.get("dropout", 0.0), 0.0, 0.9)
    integer("hidden_size", metadata.get("hidden_size", 256), 1, 512)
    model_config = metadata.get("model_config", {})
    if not isinstance(model_config, dict) or set(model_config) - {"width", "projection_size", "hidden_size"}:
        raise ValueError("Invalid compact model configuration")
    for name, default in (("width", 48), ("projection_size", 128), ("hidden_size", 96)):
        integer(name, model_config.get(name, default), 1, 512)
    if architecture in {"CharacterAudioCNN", "CompactCTCModel"}:
        charset = metadata.get("charset")
        if not isinstance(charset, str) or not charset or len(set(charset)) != len(charset):
            raise ValueError("Bundle charset must contain unique characters")
        blank_id = metadata.get("blank_id")
        integer("blank_id", blank_id, 1, 256)
        if blank_id != len(charset):
            raise ValueError("Bundle blank_id must immediately follow charset")
        if architecture == "CompactCTCModel" and charset != CHARSET:
            raise ValueError("CompactCTCModel requires the versioned digit-preserving charset")
        entries = coerce_grammar_entries(metadata.get("grammar", []))
        if len(entries) > 10_000 or sum(len(entry.text) for entry in entries) > 100_000:
            raise ValueError("Grammar exceeds runtime budget")
        if any(set(entry.text) - set(charset) or entry.intent not in labels for entry in entries):
            raise ValueError("Grammar contains unsupported characters or intents")
        integer("beam_width", metadata.get("beam_width", 16), 1, 256)
        DecoderLimits(**metadata.get("decoder_limits", {}))
        if "acceptance_policy" not in metadata:
            raise ValueError("CTC bundle requires an explicit acceptance policy")
        AcceptancePolicy.from_mapping(metadata["acceptance_policy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate", help="Migrate a verified legacy pickle")
    migrate.add_argument("--trusted-legacy", action="store_true", required=True)
    migrate.add_argument("source", type=Path)
    migrate.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "migrate":
        migrated = migrate_legacy_bundle(args.source, args.destination)
        print(f"Migrated trusted legacy artifact to {migrated}")


if __name__ == "__main__":
    main()
