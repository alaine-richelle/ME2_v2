"""Calibrate CTC command rejection on a frozen speaker-disjoint partition."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from me2.audio import AudioConfig
from me2.bundle import load_bundle, save_bundle
from me2.features import FeatureConfig, LogMel
from me2.vcm.ctc import AcceptancePolicy, decode_command
from me2.vcm.ctc_training import HFCtcAudioDataset, ctc_collate
from me2.vcm.data import (
    DATASET_ID,
    DATASET_REVISION,
    audit_audio_indices,
    load_hf_splits,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _threshold_grid(
    values: list[float],
    *,
    accept_all: float,
    quantiles: int = 12,
) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return [accept_all]
    candidates = {
        accept_all,
        *np.quantile(
            finite, np.linspace(0.0, 1.0, quantiles + 1)
        ).tolist(),
        float(np.nextafter(finite.max(), np.inf)),
    }
    return sorted(float(value) for value in candidates)


def _accepted(
    observation: dict[str, Any],
    policy: AcceptancePolicy,
) -> bool:
    return bool(
        observation["candidate_intent"]
        and observation["normalized_log_score"]
        >= policy.min_normalized_log_score
        and observation["margin"] >= policy.min_margin
        and observation["candidate_posterior"]
        >= policy.min_candidate_posterior
    )


def select_acceptance_policy(
    observations: list[dict[str, Any]],
    labels: list[str],
    *,
    target_false_accept_rate: float,
) -> tuple[AcceptancePolicy, dict[str, Any], list[str]]:
    """Select thresholds without consulting official test or holdout rows."""

    if not observations:
        raise ValueError("Command calibration observations are empty")
    if not 0.0 <= target_false_accept_rate < 1.0:
        raise ValueError("target_false_accept_rate must be in [0, 1)")
    oos = "out_of_scope"
    oos_count = sum(row["gold_intent"] == oos for row in observations)
    in_scope_count = len(observations) - oos_count
    if not oos_count or not in_scope_count:
        raise ValueError("Command calibration requires in-scope and OOS rows")

    score_grid = _threshold_grid(
        [row["normalized_log_score"] for row in observations],
        accept_all=-1.0e30,
    )
    margin_grid = _threshold_grid(
        [row["margin"] for row in observations],
        accept_all=-1.0e30,
        quantiles=8,
    )
    posterior_grid = sorted(
        {
            0.0,
            *np.quantile(
                np.asarray(
                    [row["candidate_posterior"] for row in observations]
                ),
                np.linspace(0.0, 1.0, 9),
            ).tolist(),
            1.0,
        }
    )
    gold = [str(row["gold_intent"]) for row in observations]
    best: tuple[
        tuple[float, ...],
        AcceptancePolicy,
        dict[str, Any],
        list[str],
    ] | None = None
    for score_threshold in score_grid:
        for margin_threshold in margin_grid:
            for posterior_threshold in posterior_grid:
                policy = AcceptancePolicy(
                    min_normalized_log_score=score_threshold,
                    min_margin=margin_threshold,
                    min_candidate_posterior=posterior_threshold,
                    version="calibrated-score-margin-posterior-v1",
                )
                predicted = [
                    (
                        str(row["candidate_intent"])
                        if _accepted(row, policy)
                        else oos
                    )
                    for row in observations
                ]
                false_accepts = sum(
                    reference == oos and prediction != oos
                    for reference, prediction in zip(gold, predicted)
                )
                false_accept_rate = false_accepts / oos_count
                if false_accept_rate > target_false_accept_rate:
                    continue
                wrong_actions = sum(
                    reference != oos
                    and prediction not in {reference, oos}
                    for reference, prediction in zip(gold, predicted)
                )
                in_scope_rejects = sum(
                    reference != oos and prediction == oos
                    for reference, prediction in zip(gold, predicted)
                )
                macro_f1 = float(
                    f1_score(
                        gold,
                        predicted,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                accuracy = float(accuracy_score(gold, predicted))
                wrong_action_rate = wrong_actions / in_scope_count
                utility = macro_f1 - 0.5 * wrong_action_rate
                metrics = {
                    "macro_f1": macro_f1,
                    "accuracy": accuracy,
                    "false_accepts": false_accepts,
                    "false_accept_rate": false_accept_rate,
                    "wrong_actions": wrong_actions,
                    "wrong_action_rate": wrong_action_rate,
                    "in_scope_rejects": in_scope_rejects,
                    "in_scope_reject_rate": in_scope_rejects / in_scope_count,
                    "acceptance_coverage": sum(
                        prediction != oos for prediction in predicted
                    )
                    / len(predicted),
                    "utility": utility,
                }
                objective = (
                    utility,
                    macro_f1,
                    accuracy,
                    -wrong_actions,
                    -in_scope_rejects,
                    -false_accepts,
                )
                if best is None or objective > best[0]:
                    best = (objective, policy, metrics, predicted)
    if best is not None:
        return best[1], best[2], best[3]

    policy = AcceptancePolicy(
        min_normalized_log_score=1.0e30,
        min_margin=1.0e30,
        min_candidate_posterior=1.0,
        version="calibrated-reject-all-no-eligible-policy-v1",
    )
    predicted = [oos] * len(observations)
    metrics = {
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(gold, predicted)),
        "false_accepts": 0,
        "false_accept_rate": 0.0,
        "wrong_actions": 0,
        "wrong_action_rate": 0.0,
        "in_scope_rejects": in_scope_count,
        "in_scope_reject_rate": 1.0,
        "acceptance_coverage": 0.0,
        "utility": 0.0,
    }
    return policy, metrics, predicted


def calibrate_artifact(
    model_path: Path,
    output_model: Path,
    partition_manifest: Path,
    output_dir: Path,
    *,
    target_false_accept_rate: float = 0.05,
    batch_size: int = 32,
) -> dict[str, Any]:
    model, bundle = load_bundle(model_path)
    if bundle.get("architecture") not in {
        "CharacterAudioCNN",
        "CompactCTCModel",
    }:
        raise ValueError("Command calibration requires a CTC artifact")
    partition_data = json.loads(
        partition_manifest.read_text(encoding="utf-8")
    )
    if partition_data.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Partition manifest dataset revision mismatch")
    calibration_indices = partition_data["partitions"]["calibration"][
        "indices"
    ]
    dataset = load_hf_splits(DATASET_ID, DATASET_REVISION)
    split = dataset["train"]
    audio_config = AudioConfig(**bundle["audio"])
    valid_indices, audio_issues = audit_audio_indices(
        split,
        audio_config,
        "train/calibration",
        calibration_indices,
        sequence_mode=True,
    )
    labels = list(bundle["labels"])
    label_ids = {label: index for index, label in enumerate(labels)}
    calibration_data = HFCtcAudioDataset(
        split,
        valid_indices,
        label_ids,
        audio_config,
        allow_empty_targets=True,
    )
    loader = DataLoader(
        calibration_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate,
    )
    frontend = LogMel(FeatureConfig(**bundle["features"]))
    observations: list[dict[str, Any]] = []
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for waveforms, _, _, targets, sample_lengths in loader:
            features = frontend(waveforms)
            feature_lengths = frontend.frame_lengths(sample_lengths)
            output_lengths = model.output_lengths(feature_lengths)
            logits = model(features, feature_lengths)
            for row_logits, output_length, gold_id in zip(
                logits,
                output_lengths.tolist(),
                targets.tolist(),
            ):
                source_index = valid_indices[cursor]
                source_row = split[source_index]
                result = decode_command(
                    row_logits,
                    bundle["grammar"],
                    charset=str(bundle["charset"]),
                    blank_id=int(bundle["blank_id"]),
                    beam_width=int(bundle.get("beam_width", 16)),
                    input_length=int(output_length),
                )
                observations.append(
                    {
                        "source_index": source_index,
                        "audio_id": str(
                            source_row.get("file") or source_index
                        ),
                        "speaker_id": str(
                            source_row.get("speaker_id") or ""
                        ),
                        "gold_intent": labels[int(gold_id)],
                        "candidate_intent": result.intent,
                        "decoded_text": result.decoded_text,
                        "normalized_log_score": (
                            result.normalized_log_score
                        ),
                        "margin": result.margin,
                        "candidate_posterior": (
                            result.candidate_posterior
                        ),
                    }
                )
                cursor += 1
    policy, selected_metrics, predicted = select_acceptance_policy(
        observations,
        labels,
        target_false_accept_rate=target_false_accept_rate,
    )
    for row, prediction in zip(observations, predicted):
        row["selected_prediction"] = prediction
        row["selected_accepted"] = prediction != "out_of_scope"
    summary = {
        "calibration_schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "partition_manifest": str(partition_manifest.resolve()),
        "partition": "official_train/speaker_disjoint_calibration",
        "rows_requested": len(calibration_indices),
        "rows_used": len(observations),
        "audio_issues": audio_issues,
        "target_false_accept_rate": target_false_accept_rate,
        "policy": policy.__dict__,
        "selected_metrics": selected_metrics,
        "official_test_used_for_selection": False,
        "warning": (
            "Candidate-posterior and beam scores are decoder scores, not "
            "calibrated correctness probabilities."
        ),
    }
    calibrated = dict(bundle)
    calibrated["acceptance_policy"] = policy.__dict__
    calibrated["acceptance_calibration"] = summary
    calibrated["system_version"] = (
        f"{bundle.get('model_version', '')}+{policy.version}"
    )
    save_bundle(output_model, calibrated)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in observations:
            stream.write(json.dumps(row) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "vcm/models/vcm_ctc_model.me2",
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        type=Path,
        default=ROOT / "results/vcm_ctc_partitions.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    report = calibrate_artifact(
        args.model,
        args.output_model,
        args.partitions,
        args.output,
        target_false_accept_rate=args.target_far,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
