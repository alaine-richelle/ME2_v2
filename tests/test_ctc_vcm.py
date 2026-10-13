import itertools
import json
import math
import zipfile
from typing import ClassVar

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from me2.audio import AudioConfig
from me2.bundle import load_bundle, predict_command, save_bundle
from me2.features import FeatureConfig, LogMel
from me2.model import parameter_count
from me2.vcm.analyze import analyze_predictions
from me2.vcm.calibrate import select_acceptance_policy
from me2.vcm.ctc import (
    BLANK_ID,
    CHARSET,
    AcceptancePolicy,
    CompactCTCModel,
    GrammarEntry,
    coerce_grammar_entries,
    decode_command,
    greedy_ctc_decode,
    normalize_transcript,
    prefix_beam_search,
    required_ctc_frames,
)
from me2.vcm.ctc_training import fit_ctc_classifier


def _logits(path: list[int], symbols: int, high: float = 8.0) -> np.ndarray:
    logits = np.full((len(path), symbols), -high, dtype=np.float64)
    logits[np.arange(len(path)), path] = high
    return logits


def _collapse(path: tuple[int, ...], charset: str) -> str:
    blank = len(charset)
    collapsed = [
        symbol
        for symbol, group in itertools.groupby(path)
        if symbol != blank
        for _ in [group]
    ]
    return "".join(charset[symbol] for symbol in collapsed)


def _brute_force_probability(logits: np.ndarray, text: str, charset: str) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    total = 0.0
    for path in itertools.product(range(len(charset) + 1), repeat=logits.shape[0]):
        if _collapse(path, charset) == text:
            total += float(np.prod(probabilities[np.arange(len(path)), path]))
    return total


def test_compact_character_model_emits_framewise_logits_and_lengths():
    features = LogMel(FeatureConfig())(np.zeros(20_320, dtype=np.float32))
    model = CompactCTCModel(hidden_size=16, projection_size=24, width=16)
    feature_lengths = torch.tensor([features.size(-1)])
    logits = model(features, feature_lengths)
    assert logits.shape == (1, 64, len(CHARSET) + 1)
    assert model.output_lengths(feature_lengths).tolist() == [64]
    assert parameter_count(model) < 500_000


def test_ctc_repeated_nonblank_self_loop_collapses_to_one_character():
    charset = "a"
    logits = _logits([0, 0], len(charset) + 1)
    result = decode_command(logits, [GrammarEntry("a", "intent")], charset=charset)
    assert result.accepted
    assert result.decoded_text == "a"


def test_ctc_blank_separates_identical_output_characters():
    charset = "a"
    blank = len(charset)
    logits = _logits([0, blank, 0], len(charset) + 1)
    result = decode_command(logits, [GrammarEntry("aa", "intent")], charset=charset)
    assert result.accepted
    assert result.decoded_text == "aa"
    assert required_ctc_frames("aa") == 3


def test_greedy_ctc_collapse_handles_repeats_blanks_and_padding():
    charset = "a"
    blank = len(charset)
    assert greedy_ctc_decode(
        _logits([0, 0], 2), charset=charset
    ) == "a"
    assert greedy_ctc_decode(
        _logits([0, blank, 0], 2), charset=charset
    ) == "aa"
    assert greedy_ctc_decode(
        _logits([0, blank, 0], 2), charset=charset, input_length=2
    ) == "a"


def test_prefix_beam_matches_brute_force_ctc_probability():
    charset = "ab"
    logits = np.asarray(
        [
            [1.4, 0.2, -0.1],
            [0.8, 1.0, 0.1],
            [0.3, 1.3, 0.2],
        ],
        dtype=np.float64,
    )
    candidates = prefix_beam_search(
        logits,
        [GrammarEntry("a", "a"), GrammarEntry("ab", "ab")],
        charset=charset,
        beam_width=64,
        top_k=8,
    )
    observed = {candidate.entry.text: math.exp(candidate.raw_log_score) for candidate in candidates}
    assert observed["a"] == pytest.approx(
        _brute_force_probability(logits, "a", charset), rel=1e-10, abs=1e-12
    )
    assert observed["ab"] == pytest.approx(
        _brute_force_probability(logits, "ab", charset), rel=1e-10, abs=1e-12
    )


def test_decoder_consumes_trailing_frames_and_handles_prefix_terminals():
    charset = "ab"
    logits = _logits([0, 1, 1, 1], len(charset) + 1)
    result = decode_command(
        logits,
        [GrammarEntry("a", "short"), GrammarEntry("ab", "long")],
        charset=charset,
        beam_width=16,
    )
    assert result.decoded_text == "ab"
    assert result.intent == "long"


def test_padding_frames_are_excluded_from_decoding():
    charset = "ab"
    blank = len(charset)
    logits = _logits([0, blank, 1], len(charset) + 1)
    full = decode_command(
        logits, [GrammarEntry("a", "a"), GrammarEntry("ab", "ab")], charset=charset
    )
    truncated = decode_command(
        logits,
        [GrammarEntry("a", "a"), GrammarEntry("ab", "ab")],
        charset=charset,
        input_length=2,
    )
    assert full.decoded_text == "ab"
    assert truncated.decoded_text == "a"


def test_invalid_logits_and_conflicting_grammar_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        decode_command(
            np.asarray([[np.nan, 0.0]]),
            [GrammarEntry("a", "x")],
            charset="a",
        )
    with pytest.raises(ValueError, match="Conflicting grammar semantics"):
        coerce_grammar_entries(
            [GrammarEntry("same", "x"), GrammarEntry("same", "y")]
        )


def test_normalizer_preserves_digits_and_rejects_empty_targets():
    assert normalize_transcript("Timer 10 seconds!") == "timer 10 seconds"
    assert BLANK_ID == len(CHARSET)
    with pytest.raises(ValueError, match="empty"):
        from me2.vcm.ctc import transcript_target

        transcript_target("...")


def test_acceptance_policy_rejects_low_scoring_candidate():
    charset = "a"
    result = decode_command(
        np.zeros((2, 2), dtype=np.float64),
        [GrammarEntry("a", "intent")],
        charset=charset,
        acceptance_policy=AcceptancePolicy(min_normalized_log_score=-0.1),
    )
    assert not result.accepted
    assert result.actionable_intent is None
    assert result.rejection_reason == "acoustic_score_below_threshold"


def test_command_calibration_uses_rejection_to_reduce_wrong_actions():
    observations = [
        {
            "gold_intent": "light_on",
            "candidate_intent": "light_on",
            "normalized_log_score": -0.2,
            "margin": 0.8,
            "candidate_posterior": 0.9,
        },
        {
            "gold_intent": "light_off",
            "candidate_intent": "light_on",
            "normalized_log_score": -1.8,
            "margin": 0.05,
            "candidate_posterior": 0.51,
        },
        {
            "gold_intent": "out_of_scope",
            "candidate_intent": "light_on",
            "normalized_log_score": -2.0,
            "margin": 0.02,
            "candidate_posterior": 0.5,
        },
        {
            "gold_intent": "out_of_scope",
            "candidate_intent": None,
            "normalized_log_score": float("-inf"),
            "margin": float("-inf"),
            "candidate_posterior": 0.0,
        },
    ]
    policy, metrics, predicted = select_acceptance_policy(
        observations,
        ["light_on", "light_off", "out_of_scope"],
        target_false_accept_rate=0.0,
    )
    assert predicted[0] == "light_on"
    assert predicted[2:] == ["out_of_scope", "out_of_scope"]
    assert metrics["false_accepts"] == 0
    assert policy.version == "calibrated-score-margin-posterior-v1"


def test_reject_all_acceptance_policy_is_finite_and_json_safe():
    policy = AcceptancePolicy(
        min_normalized_log_score=1.0e30,
        version="calibrated-score-reject-all-v1",
    )
    assert math.isfinite(policy.min_normalized_log_score)
    json.dumps(policy.__dict__, allow_nan=False)
    result = decode_command(
        _logits([0], 2),
        [GrammarEntry("a", "intent")],
        charset="a",
        acceptance_policy=policy,
    )
    assert not result.accepted


def test_safe_ctc_bundle_round_trip_uses_architecture_dispatch(tmp_path):
    model = CompactCTCModel(width=16, projection_size=24, hidden_size=16)
    bundle = {
        "task": "vcm",
        "architecture": "CompactCTCModel",
        "model_version": "test",
        "model_config": {"width": 16, "projection_size": 24, "hidden_size": 16},
        "features": FeatureConfig().__dict__,
        "audio": {
            "sample_rate": 16_000,
            "duration_seconds": 1.0,
            "silence_threshold_db": -48.0,
            "normalize_peak": True,
        },
        "labels": ["add_intent", "out_of_scope"],
        "grammar": [GrammarEntry("add", "add_intent").to_dict()],
        "grammar_version": "test",
        "charset": CHARSET,
        "blank_id": BLANK_ID,
        "beam_width": 8,
        "acceptance_policy": AcceptancePolicy().__dict__,
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "ctc.me2"
    save_bundle(path, bundle)
    loaded_model, loaded_bundle = load_bundle(path)
    assert isinstance(loaded_model, CompactCTCModel)
    assert loaded_bundle["architecture"] == "CompactCTCModel"
    result = predict_command(
        loaded_model, loaded_bundle, np.zeros(16_000, dtype=np.float32)
    )
    assert isinstance(result.accepted, bool)
    assert result.model_version == "test"


def test_safe_loader_refuses_pickle_and_detects_weight_corruption(tmp_path):
    pickle_path = tmp_path / "legacy.pkl"
    pickle_path.write_bytes(b"not trusted")
    with pytest.raises(ValueError, match="Refusing unrestricted pickle"):
        load_bundle(pickle_path)

    model = CompactCTCModel(width=8, projection_size=12, hidden_size=8)
    bundle = {
        "task": "vcm",
        "architecture": "CompactCTCModel",
        "model_version": "test",
        "model_config": {"width": 8, "projection_size": 12, "hidden_size": 8},
        "features": FeatureConfig().__dict__,
        "audio": AudioConfig(duration_seconds=1.0).__dict__,
        "labels": ["intent", "out_of_scope"],
        "grammar": [GrammarEntry("a", "intent").to_dict()],
        "grammar_version": "test",
        "charset": CHARSET,
        "blank_id": BLANK_ID,
        "beam_width": 4,
        "acceptance_policy": AcceptancePolicy().__dict__,
        "state_dict": model.state_dict(),
    }
    valid_path = tmp_path / "valid.me2"
    corrupt_path = tmp_path / "corrupt.me2"
    save_bundle(valid_path, bundle)
    with zipfile.ZipFile(valid_path) as archive:
        metadata = archive.read("metadata.json")
        weights = archive.read("weights.npz")
    with zipfile.ZipFile(corrupt_path, "w") as archive:
        archive.writestr("metadata.json", metadata)
        archive.writestr("weights.npz", weights + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(corrupt_path)


def test_model_initialization_is_reproducible_when_seeded_first():
    torch.manual_seed(17)
    first = CompactCTCModel(width=8, projection_size=12, hidden_size=8)
    torch.manual_seed(17)
    second = CompactCTCModel(width=8, projection_size=12, hidden_size=8)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_ctc_trainer_smoke_with_synthetic_waveforms(tmp_path):
    class SyntheticDataset(Dataset):
        class_ids: ClassVar[list[int]] = [0, 1]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            target = torch.tensor([0 if index == 0 else 1])
            return torch.zeros(8_000), target, index, 8_000

    labels = ["a_intent", "out_of_scope"]
    model = CompactCTCModel(
        width=8,
        projection_size=12,
        hidden_size=8,
        output_symbols=len(CHARSET) + 1,
    )
    result = fit_ctc_classifier(
        model,
        SyntheticDataset(),
        SyntheticDataset(),
        FeatureConfig(),
        [GrammarEntry("a", "a_intent").to_dict()],
        labels,
        seed=3,
        batch_size=2,
        epochs=1,
        patience=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        log_path=tmp_path / "ctc-training.jsonl",
        augmentation={"enabled": False},
        beam_width=2,
    )
    assert "state_dict" in result
    assert (tmp_path / "ctc-training.jsonl").is_file()


def test_command_analysis_keeps_rejection_and_directional_confusions(tmp_path):
    model = CompactCTCModel(width=8, projection_size=12, hidden_size=8)
    bundle = {
        "task": "vcm",
        "architecture": "CompactCTCModel",
        "model_version": "analysis-test",
        "model_config": {"width": 8, "projection_size": 12, "hidden_size": 8},
        "features": FeatureConfig().__dict__,
        "audio": AudioConfig(duration_seconds=1.0).__dict__,
        "labels": ["alpha", "beta", "out_of_scope"],
        "grammar": [
            GrammarEntry("alpha", "alpha").to_dict(),
            GrammarEntry("beta", "beta").to_dict(),
        ],
        "grammar_version": "analysis-grammar",
        "charset": CHARSET,
        "blank_id": BLANK_ID,
        "beam_width": 4,
        "acceptance_policy": AcceptancePolicy().__dict__,
        "state_dict": model.state_dict(),
    }
    model_path = tmp_path / "model.me2"
    save_bundle(model_path, bundle)
    predictions = tmp_path / "predictions.jsonl"
    prediction_rows = [
        {
            "input_audio": "a.wav",
            "gold_command": "alpha",
            "predicted_command": "out_of_scope",
            "accepted": False,
            "actual_transcript": "alpha",
            "greedy_text": "alpha",
            "decoded_text": "alpha",
            "trie_candidate_intent": "alpha",
        },
        {
            "input_audio": "b.wav",
            "gold_command": "beta",
            "predicted_command": "alpha",
            "accepted": True,
            "actual_transcript": "beta",
            "greedy_text": "alpha",
            "decoded_text": "alpha",
            "trie_candidate_intent": "alpha",
        },
        {
            "input_audio": "noise.wav",
            "gold_command": "out_of_scope",
            "predicted_command": "alpha",
            "accepted": True,
            "actual_transcript": "ordinary speech",
            "greedy_text": "alpha",
            "decoded_text": "alpha",
            "trie_candidate_intent": "alpha",
        },
    ]
    predictions.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    summary = analyze_predictions(predictions, model_path, output)
    assert summary["in_scope_false_rejects"] == 1
    assert summary["out_of_scope_false_accepts"] == 1
    errors = (output / "errors.csv").read_text(encoding="utf-8")
    assert "__rejected__" in errors
    assert "wrong_command" in errors
