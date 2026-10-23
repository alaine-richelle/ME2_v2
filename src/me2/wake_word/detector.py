"""Wake probability smoothing shared by streaming evaluation and runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from me2.contracts import finite, integer


@dataclass(frozen=True)
class WakeSmoothingPolicy:
    threshold: float
    smoothing_window: int = 3
    required_hits: int = 2

    def __post_init__(self) -> None:
        finite("Wake threshold", self.threshold, 0.0, 1.0)
        integer("smoothing_window", self.smoothing_window, 1, 128)
        integer("required_hits", self.required_hits, 1, self.smoothing_window)


class WakeEventDetector:
    """Emit one event after enough complete-word hits in a bounded window."""

    def __init__(self, policy: WakeSmoothingPolicy) -> None:
        self.policy = policy
        self._hits: deque[bool] = deque(maxlen=policy.smoothing_window)
        self._last_diagnostics: dict[str, int | float | bool | str] = {}

    def update(self, predicted_label: str, wake_probability: float) -> bool:
        hit = (
            predicted_label == "ateko"
            and wake_probability >= self.policy.threshold
        )
        self._hits.append(hit)
        hit_count = sum(self._hits)
        detected = hit_count >= self.policy.required_hits
        if predicted_label != "ateko":
            reason = "wake_predicted_non_ateko"
        elif wake_probability < self.policy.threshold:
            reason = "wake_score_below_threshold"
        elif not detected:
            reason = "wake_persistence_pending"
        else:
            reason = "wake_persistence_satisfied"
        self._last_diagnostics = {
            "wake_effective_threshold": self.policy.threshold,
            "wake_persistence_rule": "n_of_m",
            "wake_smoothing_window": self.policy.smoothing_window,
            "wake_required_hits": self.policy.required_hits,
            "wake_observed_hits": hit_count,
            "wake_current_hit": hit,
            "wake_nontrigger_reason": reason,
        }
        if not detected:
            return False
        self.reset()
        return True

    def diagnostics(self) -> dict[str, int | float | bool | str]:
        """Return decision-only metadata; no acoustic samples are retained."""
        return dict(self._last_diagnostics)

    def reset(self) -> None:
        self._hits.clear()
