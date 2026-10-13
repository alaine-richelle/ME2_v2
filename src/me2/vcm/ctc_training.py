"""Training and calibration helpers for the compact character CTC VCM."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from me2.audio import AudioConfig, prepare_audio_sequence
from me2.features import FeatureConfig, LogMel
from me2.metrics import classification_metrics
from me2.training import _augment_batch, seed_everything
from me2.vcm.ctc import (
    BLANK_ID,
    CHARSET,
    AcceptancePolicy,
    CompactCTCModel,
    GrammarEntry,
    coerce_grammar_entries,
    decode_command,
    decode_grammar,
    normalize_transcript,
    required_ctc_frames,
    transcript_target,
)
from me2.vcm.data import command_from_row, decode_audio

LOGGER = logging.getLogger(__name__)


def transcript_for_row(row: dict[str, Any]) -> str:
    """Return normalized words actually recorded; never replace them with intent names."""

    return normalize_transcript(str(row.get("transcript") or ""))


def grammar_entry_for_row(row: dict[str, Any]) -> GrammarEntry | None:
    if command_from_row(row) == "out_of_scope":
        return None
    slot_value = str(row.get("slot_value") or "").strip()
    slots = {"value": slot_value} if slot_value.casefold() not in {"", "none", "nan"} else {}
    return GrammarEntry(
        text=transcript_for_row(row),
        intent=command_from_row(row),
        variation=str(row.get("variation") or "").strip(),
        slots=slots,
        provenance="approved_official_train_transcript",
    )


def valid_ctc_text_indices(split, indices: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
    valid: list[int] = []
    quarantined: list[dict[str, Any]] = []
    for index in indices:
        row = split[index]
        text = transcript_for_row(row)
        if not text:
            quarantined.append(
                {
                    "index": index,
                    "file": str(row.get("file") or ""),
                    "reason": "empty_or_nonspeech_transcript",
                    "transcript_source": str(row.get("transcript_source") or ""),
                }
            )
            continue
        try:
            transcript_target(text)
        except ValueError as error:
            quarantined.append(
                {
                    "index": index,
                    "file": str(row.get("file") or ""),
                    "reason": str(error),
                    "transcript": str(row.get("transcript") or ""),
                }
            )
            continue
        valid.append(index)
    return valid, quarantined


class HFCtcAudioDataset(Dataset):
    def __init__(
        self,
        split,
        indices: list[int],
        labels: dict[str, int],
        audio_config: AudioConfig,
        *,
        allow_empty_targets: bool = False,
    ) -> None:
        self.split = split
        self.indices = indices
        self.labels = labels
        self.audio_config = audio_config
        self.allow_empty_targets = allow_empty_targets
        self.class_ids = [labels[command_from_row(split[index])] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, int, int]:
        row = self.split[self.indices[item]]
        samples, sample_rate = decode_audio(row["audio"])
        waveform, sample_length = prepare_audio_sequence(
            samples, sample_rate, self.audio_config
        )
        text = transcript_for_row(row)
        target = [] if self.allow_empty_targets and not text else transcript_target(text)
        return (
            torch.from_numpy(waveform),
            torch.tensor(target, dtype=torch.long),
            self.labels[command_from_row(row)],
            sample_length,
        )


def ctc_collate(
    batch: list[tuple[Tensor, Tensor, int, int] | tuple[Tensor, Tensor, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    normalized = [
        item if len(item) == 4 else (*item, int(item[0].numel()))
        for item in batch
    ]
    waveforms, targets, labels, sample_lengths = zip(*normalized)
    target_lengths = torch.tensor(
        [target.numel() for target in targets], dtype=torch.long
    )
    return (
        torch.stack(waveforms),
        torch.cat(targets),
        target_lengths,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(sample_lengths, dtype=torch.long),
    )


def training_grammar(split, indices: list[int]) -> list[dict[str, Any]]:
    """Build an unambiguous product grammar from approved official training rows."""

    entries = [
        entry
        for index in indices
        if (entry := grammar_entry_for_row(split[index])) is not None
    ]
    return [entry.to_dict() for entry in coerce_grammar_entries(entries)]


def _ctc_lengths(
    frontend: LogMel,
    model: CompactCTCModel,
    sample_lengths: Tensor,
    maximum_frames: int,
) -> tuple[Tensor, Tensor]:
    feature_lengths = frontend.frame_lengths(sample_lengths)
    output_lengths = model.output_lengths(feature_lengths).clamp(max=maximum_frames)
    return feature_lengths, output_lengths


def _assert_ctc_feasible(
    targets: Tensor,
    target_lengths: Tensor,
    output_lengths: Tensor,
) -> None:
    offset = 0
    failures: list[str] = []
    for row, (target_length, output_length) in enumerate(
        zip(target_lengths.tolist(), output_lengths.tolist())
    ):
        target = targets[offset : offset + target_length]
        required = required_ctc_frames(target.tolist())
        if output_length < required:
            failures.append(
                f"row={row} output_frames={output_length} required_frames={required} "
                f"target_length={target_length}"
            )
        offset += target_length
    if failures:
        raise ValueError("CTC-infeasible batch: " + "; ".join(failures[:8]))


def _evaluate_ctc(
    model: CompactCTCModel,
    loader: DataLoader,
    frontend: LogMel,
    criterion: nn.CTCLoss,
    grammar: list[dict[str, Any]],
    labels: list[str],
    beam_width: int,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_total = 0.0
    row_count = 0
    gold: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    label_to_id = {label: index for index, label in enumerate(labels)}
    with torch.inference_mode():
        for waveforms, targets, target_lengths, batch_labels, sample_lengths in loader:
            waveforms = waveforms.to(device)
            sample_lengths = sample_lengths.to(device)
            features = frontend(waveforms)
            feature_lengths, input_lengths = _ctc_lengths(
                frontend, model, sample_lengths, features.size(-1)
            )
            logits = model(features, feature_lengths)
            _assert_ctc_feasible(targets, target_lengths, input_lengths.cpu())
            log_probs = logits.log_softmax(dim=2).transpose(0, 1)
            loss = criterion(
                log_probs,
                targets.to(device),
                input_lengths,
                target_lengths.to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation CTC loss is non-finite")
            loss_total += float(loss) * waveforms.size(0)
            row_count += waveforms.size(0)
            for row_logits, row_length, label_id in zip(
                logits, input_lengths.tolist(), batch_labels.tolist()
            ):
                predicted, _, class_probabilities, _ = decode_grammar(
                    row_logits,
                    grammar,
                    labels,
                    beam_width,
                    charset=CHARSET,
                    input_length=row_length,
                )
                gold.append(label_id)
                predictions.append(label_to_id[predicted])
                probabilities.append(class_probabilities)
    metrics = classification_metrics(
        gold, predictions, np.asarray(probabilities), labels
    )
    return loss_total / max(row_count, 1), metrics


def fit_ctc_classifier(
    model: CompactCTCModel,
    train_dataset: HFCtcAudioDataset,
    validation_dataset: HFCtcAudioDataset,
    feature_config: FeatureConfig,
    grammar: list[dict[str, Any]],
    labels: list[str],
    *,
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    log_path: str | Path,
    augmentation: dict[str, Any] | None = None,
    beam_width: int = 16,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frontend = LogMel(feature_config)
    counts = Counter(train_dataset.class_ids)
    sample_weights = [1.0 / counts[class_id] for class_id in train_dataset.class_ids]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=ctc_collate,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2
    )
    generator = torch.Generator().manual_seed(seed)
    history_path = Path(log_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    epochs_completed = 0
    stopping_reason = "maximum_epochs_reached"
    epochs_without_improvement = 0
    started = time.perf_counter()

    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            epochs_completed = epoch
            model.train()
            total_loss = 0.0
            sample_count = 0
            for (
                waveforms,
                targets,
                target_lengths,
                _,
                sample_lengths,
            ) in train_loader:
                waveforms = _augment_batch(
                    waveforms, feature_config.sample_rate, generator, augmentation
                ).to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)
                sample_lengths = sample_lengths.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features = frontend(waveforms)
                feature_lengths, input_lengths = _ctc_lengths(
                    frontend, model, sample_lengths, features.size(-1)
                )
                _assert_ctc_feasible(
                    targets, target_lengths, input_lengths.detach().cpu()
                )
                logits = model(features, feature_lengths)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("CTC model produced non-finite logits")
                log_probs = logits.log_softmax(dim=2).transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training CTC loss is non-finite")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                if not math.isfinite(float(gradient_norm)):
                    raise FloatingPointError("CTC gradient norm is non-finite")
                optimizer.step()
                total_loss += float(loss.detach()) * waveforms.size(0)
                sample_count += waveforms.size(0)

            train_loss = total_loss / max(sample_count, 1)
            validation_loss, validation_metrics = _evaluate_ctc(
                model,
                validation_loader,
                frontend,
                criterion,
                grammar,
                labels,
                beam_width,
                device,
            )
            validation_macro_f1 = validation_metrics["f1_macro"]
            scheduler.step(validation_macro_f1)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_ctc_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            LOGGER.info(
                "CTC epoch=%d train_loss=%.4f val_loss=%.4f "
                "val_acc=%.4f val_macro_f1=%.4f",
                epoch,
                train_loss,
                validation_loss,
                validation_metrics["accuracy"],
                validation_macro_f1,
            )
            if validation_macro_f1 > best_macro_f1:
                best_macro_f1 = validation_macro_f1
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    LOGGER.info("CTC early stopping after %d epochs", epoch)
                    stopping_reason = "early_stopping"
                    break

    if best_state is None:
        raise RuntimeError("CTC training produced no valid model state")
    model.load_state_dict(best_state)
    return {
        "state_dict": best_state,
        "best_validation_macro_f1": best_macro_f1,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "stopping_reason": stopping_reason,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }


def calibrate_acceptance_policy(
    model: CompactCTCModel,
    calibration_dataset: HFCtcAudioDataset,
    feature_config: FeatureConfig,
    grammar: list[dict[str, Any]],
    labels: list[str],
    *,
    beam_width: int,
    batch_size: int,
    target_false_accept_rate: float = 0.05,
) -> tuple[AcceptancePolicy, dict[str, Any]]:
    """Choose a score threshold on calibration data only."""

    if not 0 <= target_false_accept_rate < 1:
        raise ValueError("target_false_accept_rate must be in [0, 1)")
    device = next(model.parameters()).device
    frontend = LogMel(feature_config)
    loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate,
    )
    observations: list[tuple[float, bool, bool]] = []
    out_of_scope_id = labels.index("out_of_scope")
    model.eval()
    with torch.inference_mode():
        for waveforms, _, _, batch_labels, sample_lengths in loader:
            waveforms = waveforms.to(device)
            sample_lengths = sample_lengths.to(device)
            features = frontend(waveforms)
            feature_lengths, input_lengths = _ctc_lengths(
                frontend, model, sample_lengths, features.size(-1)
            )
            logits = model(features, feature_lengths)
            for row_logits, row_length, gold_id in zip(
                logits, input_lengths.tolist(), batch_labels.tolist()
            ):
                result = decode_command(
                    row_logits,
                    grammar,
                    charset=CHARSET,
                    blank_id=BLANK_ID,
                    beam_width=beam_width,
                    input_length=row_length,
                )
                observations.append(
                    (
                        result.normalized_log_score,
                        gold_id == out_of_scope_id,
                        result.intent == labels[gold_id],
                    )
                )
    oos_count = sum(is_oos for _, is_oos, _ in observations)
    in_scope_count = len(observations) - oos_count
    if not oos_count or not in_scope_count:
        raise ValueError("Calibration requires both in-scope and out-of-scope examples")
    thresholds = sorted(
        {score for score, _, _ in observations if math.isfinite(score)},
        reverse=True,
    )
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        false_accepts = sum(
            is_oos and score >= threshold for score, is_oos, _ in observations
        )
        false_accept_rate = false_accepts / oos_count
        if false_accept_rate > target_false_accept_rate:
            continue
        false_rejects = sum(
            (not is_oos) and score < threshold for score, is_oos, _ in observations
        )
        false_reject_rate = false_rejects / in_scope_count
        candidate = (false_reject_rate, -threshold, false_accept_rate)
        if best is None or candidate < best:
            best = candidate
    calibration_status = "target_far_satisfied"
    if best is None:
        # A finite reject-all sentinel keeps the artifact JSON-safe. This occurs
        # when no finite terminal score can satisfy the requested false-accept rate.
        threshold = 1.0e30
        false_reject_rate = 1.0
        false_accept_rate = 0.0
        calibration_status = "reject_all_no_eligible_threshold"
    else:
        false_reject_rate, negative_threshold, false_accept_rate = best
        threshold = -negative_threshold
    policy = AcceptancePolicy(
        min_normalized_log_score=threshold,
        version=(
            "calibrated-score-v1"
            if calibration_status == "target_far_satisfied"
            else "calibrated-score-reject-all-v1"
        ),
    )
    summary = {
        "rows": len(observations),
        "out_of_scope_rows": oos_count,
        "in_scope_rows": in_scope_count,
        "target_false_accept_rate": target_false_accept_rate,
        "observed_false_accept_rate": false_accept_rate,
        "observed_in_scope_false_reject_rate": false_reject_rate,
        "min_normalized_log_score": threshold,
        "status": calibration_status,
        "correct_candidate_rate_before_rejection": sum(
            correct for _, _, correct in observations
        )
        / len(observations),
    }
    return policy, summary
