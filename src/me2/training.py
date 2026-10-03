"""Reproducible training loop shared by the wake and command classifiers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from me2.features import FeatureConfig, LogMel

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _augment_batch(
    waveforms: Tensor,
    sample_rate: int,
    generator: torch.Generator,
    config: dict[str, Any] | None = None,
) -> Tensor:
    config = config or {}
    if not config.get("enabled", True):
        return waveforms
    batch_size, sample_count = waveforms.shape
    device = waveforms.device
    gain_db = float(config.get("gain_db", 8.0))
    gain_limit = 10.0 ** (gain_db / 20.0)
    gains = torch.empty(batch_size, 1, device=device).uniform_(
        1.0 / gain_limit, gain_limit, generator=generator
    )
    result = waveforms * gains
    speed_min = float(config.get("speed_min", 0.92))
    speed_max = float(config.get("speed_max", 1.08))
    if speed_min != 1.0 or speed_max != 1.0:
        speed = torch.empty(batch_size, 1, device=device).uniform_(
            speed_min, speed_max, generator=generator
        )
        center = (sample_count - 1) / 2.0
        positions = (torch.arange(sample_count, device=device).float()[None, :] - center) * speed + center
        outside = (positions < 0) | (positions > sample_count - 1)
        lower_unclamped = positions.floor().long()
        fraction = (positions - lower_unclamped).clamp(0.0, 1.0)
        lower = lower_unclamped.clamp(0, sample_count - 1)
        upper = (lower + 1).clamp(0, sample_count - 1)
        result = result.gather(1, lower) * (1.0 - fraction) + result.gather(1, upper) * fraction
        result = result.masked_fill(outside, 0.0)
    if sample_count > 1:
        shift_limit = min(
            round(sample_rate * float(config.get("max_shift_ms", 100.0)) / 1000.0),
            sample_count // 8,
        )
        shifts = torch.randint(
            -shift_limit, shift_limit + 1, (batch_size,), generator=generator, device="cpu"
        )
        shifted = torch.zeros_like(result)
        for row, shift_value in enumerate(shifts.tolist()):
            if shift_value > 0:
                shifted[row, shift_value:] = result[row, :-shift_value]
            elif shift_value < 0:
                shifted[row, :shift_value] = result[row, -shift_value:]
            else:
                shifted[row] = result[row]
        result = shifted
    rms = result.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    snr = torch.empty(batch_size, 1, device=device).uniform_(
        float(config.get("noise_snr_db_min", 8.0)),
        float(config.get("noise_snr_db_max", 30.0)),
        generator=generator,
    )
    noise_scale = rms / torch.pow(10.0, snr / 20.0)
    noise = torch.randn(result.shape, generator=generator, device=device)
    apply_noise = torch.rand(batch_size, 1, generator=generator, device=device) < 0.5
    result = result + noise * noise_scale * apply_noise
    return result.clamp(-1.0, 1.0)


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    frontend: LogMel,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(frontend(waveforms))
            total_loss += float(criterion(logits, labels)) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def fit_classifier(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    feature_config: FeatureConfig,
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
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frontend = LogMel(feature_config)
    train_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    history_path = Path(log_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            count = 0
            for waveforms, labels in train_loader:
                waveforms = _augment_batch(
                    waveforms, feature_config.sample_rate, train_generator, augmentation
                ).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(frontend(waveforms))
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * labels.size(0)
                correct += int((logits.detach().argmax(dim=1) == labels).sum())
                count += labels.size(0)

            train_loss = total_loss / max(count, 1)
            train_accuracy = correct / max(count, 1)
            validation_loss, validation_accuracy = _evaluate_loss(
                model, validation_loader, frontend, criterion, device
            )
            scheduler.step(validation_loss)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            LOGGER.info(
                "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    LOGGER.info("Early stopping after %d epochs", epoch)
                    break

    if best_state is None:
        raise RuntimeError("Training produced no valid model state")
    model.load_state_dict(best_state)
    return {"state_dict": best_state, "best_validation_loss": best_loss, "device": str(device)}
