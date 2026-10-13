"""Held-out VCM evaluation and per-record error analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from me2.audio import AudioConfig
from me2.bundle import load_bundle
from me2.features import FeatureConfig, LogMel
from me2.metrics import classification_metrics, profile_model, save_evaluation
from me2.vcm.ctc import (
    CHARSET,
    decode_command,
    greedy_ctc_decode,
    normalize_transcript,
)
from me2.vcm.ctc_training import HFCtcAudioDataset, ctc_collate
from me2.vcm.data import (
    HFAudioDataset,
    audit_audio_indices,
    command_from_row,
    load_hf_splits,
    valid_audio_indices,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + int(reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def evaluate(bundle_path: Path, split_name: str, output_dir: Path, dataset_id: str) -> dict:
    model, bundle = load_bundle(bundle_path)
    dataset = load_hf_splits(dataset_id)
    if split_name not in dataset:
        raise ValueError(f"Unknown evaluation split {split_name!r}")
    labels = {name: index for index, name in enumerate(bundle["labels"])}
    audio_config = AudioConfig(**bundle["audio"])
    feature_config = FeatureConfig(**bundle["features"])
    is_ctc = bundle.get("architecture") in {"CharacterAudioCNN", "CompactCTCModel"}
    if is_ctc:
        indices, audio_issues = audit_audio_indices(
            dataset[split_name],
            audio_config,
            split_name,
            sequence_mode=True,
        )
        eval_data = HFCtcAudioDataset(
            dataset[split_name],
            indices,
            labels,
            audio_config,
            allow_empty_targets=True,
        )
        loader = DataLoader(
            eval_data,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=ctc_collate,
        )
    else:
        indices = valid_audio_indices(dataset[split_name], audio_config, split_name)
        audio_issues = []
        eval_data = HFAudioDataset(dataset[split_name], indices, labels, audio_config)
        loader = DataLoader(eval_data, batch_size=64, shuffle=False, num_workers=0)
    frontend = LogMel(feature_config)
    all_gold: list[int] = []
    all_pred: list[int] = []
    all_probabilities: list[list[float]] = []
    metadata: list[dict] = []
    accepted_exact_transcript_matches = 0
    trie_exact_transcript_matches = 0
    greedy_exact_transcript_matches = 0
    transcript_rows = 0
    greedy_character_errors = 0
    reference_characters = 0
    supported_slot_matches = 0
    supported_slot_rows = 0
    rejected_rows = 0
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if is_ctc:
                waveforms, _, _, targets, sample_lengths = batch
            else:
                waveforms, targets = batch
            features = frontend(waveforms)
            decoded = None
            greedy_texts = None
            if is_ctc:
                feature_lengths = frontend.frame_lengths(sample_lengths)
                output_lengths = model.output_lengths(feature_lengths)
                frame_logits = model(features, feature_lengths)
                decoded = [
                    decode_command(
                        logits,
                        bundle["grammar"],
                        charset=str(bundle.get("charset", CHARSET)),
                        blank_id=int(bundle.get("blank_id", len(CHARSET))),
                        beam_width=int(bundle.get("beam_width", 16)),
                        input_length=int(output_length),
                        acceptance_policy=bundle["acceptance_policy"],
                        model_version=str(bundle.get("model_version", "")),
                        grammar_version=str(bundle.get("grammar_version", "")),
                    )
                    for logits, output_length in zip(frame_logits, output_lengths)
                ]
                greedy_texts = [
                    greedy_ctc_decode(
                        logits,
                        charset=str(bundle.get("charset", CHARSET)),
                        blank_id=int(bundle.get("blank_id", len(CHARSET))),
                        input_length=int(output_length),
                    )
                    for logits, output_length in zip(frame_logits, output_lengths)
                ]
                predictions = np.asarray(
                    [
                        labels.get(result.actionable_intent or "out_of_scope")
                        for result in decoded
                    ],
                    dtype=np.int64,
                )
                probabilities = np.zeros(
                    (len(decoded), len(bundle["labels"])), dtype=np.float64
                )
                oos_id = labels.get("out_of_scope")
                for row_index, (result, prediction) in enumerate(
                    zip(decoded, predictions)
                ):
                    if result.accepted:
                        probabilities[row_index, prediction] = max(
                            result.candidate_posterior, 1e-6
                        )
                        if oos_id is not None:
                            probabilities[row_index, oos_id] = max(
                                0.0, 1.0 - result.candidate_posterior
                            )
                    elif oos_id is not None:
                        probabilities[row_index, oos_id] = 1.0
                    total = probabilities[row_index].sum()
                    probabilities[row_index] /= max(total, 1e-12)
            else:
                probabilities = torch.softmax(model(features), dim=1).cpu().numpy()
                predictions = probabilities.argmax(axis=1)
            all_gold.extend(targets.numpy().tolist())
            all_pred.extend(predictions.tolist())
            all_probabilities.extend(probabilities.tolist())
            for local_index in range(len(targets)):
                row = dataset[split_name][indices[cursor]]
                gold = command_from_row(row)
                result = decoded[local_index] if decoded is not None else None
                predicted = bundle["labels"][int(predictions[local_index])]
                normalized_gold = normalize_transcript(str(row.get("transcript") or ""))
                if result is not None:
                    greedy_text = greedy_texts[local_index]
                    if normalized_gold:
                        transcript_rows += 1
                        accepted_exact_transcript_matches += int(
                            result.accepted and result.decoded_text == normalized_gold
                        )
                        trie_exact_transcript_matches += int(
                            result.decoded_text == normalized_gold
                        )
                        greedy_exact_transcript_matches += int(
                            greedy_text == normalized_gold
                        )
                        greedy_character_errors += _edit_distance(
                            normalized_gold, greedy_text
                        )
                        reference_characters += len(normalized_gold)
                    rejected_rows += int(not result.accepted)
                    slot_value = str(row.get("slot_value") or "").strip()
                    if slot_value.casefold() not in {"", "none", "nan"}:
                        supported_slot_rows += 1
                        supported_slot_matches += int(
                            result.accepted
                            and result.slots.get("value") == slot_value
                            and result.intent == gold
                        )
                metadata.append({
                    "input_audio": str(row.get("file") or row.get("audio", {}).get("path") or indices[cursor]),
                    "gold_command": gold,
                    "predicted_command": predicted,
                    "confidence": float(probabilities[local_index].max()),
                    "correct": gold == predicted,
                    "speaker_id": str(row.get("speaker_id", "")),
                    "source": str(row.get("source", "")),
                    "actual_transcript": str(row.get("transcript") or ""),
                    "reference_normalized_text": normalized_gold,
                    "transcript_source": str(row.get("transcript_source") or ""),
                    "variation": str(row.get("variation") or ""),
                    "slot_value": str(row.get("slot_value") or ""),
                    **(
                        {
                            "decoded_text": result.decoded_text,
                            "greedy_text": greedy_texts[local_index],
                            "trie_candidate_intent": result.intent,
                            "trie_candidate_slots": dict(result.slots),
                            "accepted": result.accepted,
                            "rejection_reason": result.rejection_reason,
                            "raw_log_score": result.raw_log_score,
                            "normalized_log_score": result.normalized_log_score,
                            "candidate_posterior": result.candidate_posterior,
                            "semantic_margin": result.margin,
                            "predicted_slots": dict(result.slots),
                            "model_version": result.model_version,
                            "grammar_version": result.grammar_version,
                        }
                        if result is not None
                        else {}
                    ),
                })
                cursor += 1
    if audio_issues:
        oos_id = labels["out_of_scope"]
        for issue in audio_issues:
            row = dataset[split_name][issue["index"]]
            gold = command_from_row(row)
            all_gold.append(labels[gold])
            all_pred.append(oos_id)
            rejected_probability = [0.0] * len(bundle["labels"])
            rejected_probability[oos_id] = 1.0
            all_probabilities.append(rejected_probability)
            rejected_rows += 1
            metadata.append(
                {
                    "input_audio": issue["item"],
                    "gold_command": gold,
                    "predicted_command": "out_of_scope",
                    "confidence": 1.0,
                    "correct": gold == "out_of_scope",
                    "speaker_id": str(row.get("speaker_id", "")),
                    "source": str(row.get("source", "")),
                    "actual_transcript": str(row.get("transcript") or ""),
                    "reference_normalized_text": normalize_transcript(
                        str(row.get("transcript") or "")
                    ),
                    "transcript_source": str(row.get("transcript_source") or ""),
                    "variation": str(row.get("variation") or ""),
                    "slot_value": str(row.get("slot_value") or ""),
                    "decoded_text": "",
                    "greedy_text": "",
                    "trie_candidate_intent": None,
                    "trie_candidate_slots": {},
                    "accepted": False,
                    "rejection_reason": f"audio_contract_error: {issue['reason']}",
                    "raw_log_score": None,
                    "normalized_log_score": None,
                    "candidate_posterior": 0.0,
                    "semantic_margin": None,
                    "predicted_slots": {},
                    "model_version": str(bundle.get("model_version", "")),
                    "grammar_version": str(bundle.get("grammar_version", "")),
                }
            )
    metrics = classification_metrics(all_gold, all_pred, np.asarray(all_probabilities), bundle["labels"])
    oos_id = labels.get("out_of_scope")
    if oos_id is not None:
        gold = np.asarray(all_gold)
        predicted = np.asarray(all_pred)
        metrics["out_of_scope_false_accepts"] = int(((gold == oos_id) & (predicted != oos_id)).sum())
        metrics["out_of_scope_count"] = int((gold == oos_id).sum())
        metrics["out_of_scope_false_accept_rate"] = (
            metrics["out_of_scope_false_accepts"] / metrics["out_of_scope_count"]
            if metrics["out_of_scope_count"] else None
        )
        command_mask = gold != oos_id
        metrics["in_scope_false_rejects"] = int((command_mask & (predicted == oos_id)).sum())
        metrics["in_scope_count"] = int(command_mask.sum())
    metrics["split"] = split_name
    metrics["rows_evaluated"] = len(all_gold)
    if is_ctc:
        metrics["normalized_exact_transcript_accuracy"] = (
            accepted_exact_transcript_matches / transcript_rows
            if transcript_rows
            else None
        )
        metrics["trie_candidate_exact_transcript_accuracy"] = (
            trie_exact_transcript_matches / transcript_rows
            if transcript_rows
            else None
        )
        metrics["greedy_exact_transcript_accuracy"] = (
            greedy_exact_transcript_matches / transcript_rows
            if transcript_rows
            else None
        )
        metrics["greedy_character_error_rate"] = (
            greedy_character_errors / reference_characters
            if reference_characters
            else None
        )
        metrics["transcript_rows"] = transcript_rows
        metrics["supported_slot_joint_accuracy"] = (
            supported_slot_matches / supported_slot_rows
            if supported_slot_rows
            else None
        )
        metrics["supported_slot_rows"] = supported_slot_rows
        metrics["rejected_rows"] = rejected_rows
        metrics["acceptance_coverage"] = (
            (len(all_gold) - rejected_rows) / len(all_gold) if all_gold else None
        )
        metrics["acceptance_policy"] = bundle["acceptance_policy"]
        metrics["grammar_version"] = bundle.get("grammar_version")
        metrics["decoder_version"] = bundle.get("decoder_version")
    metrics.update(profile_model(model, bundle["input_shape"], bundle_path))
    save_evaluation(output_dir, metrics, metadata)
    LOGGER.info("VCM %s metrics: %s", split_name, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "vcm/models/vcm_ctc_model.me2")
    parser.add_argument("--split", choices=("test", "holdout"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "results/vcm_ctc_test_v2")
    parser.add_argument("--dataset-id", default="airimonda/ai231-me2-voice-commands")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(args.model, args.split, args.output, args.dataset_id)


if __name__ == "__main__":
    main()
