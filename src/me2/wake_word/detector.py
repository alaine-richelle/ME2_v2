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

    def update(self, predicted_label: str, wake_probability: float) -> bool:
        hit = (
            predicted_label == "ateko"
            and wake_probability >= self.policy.threshold
        )
        self._hits.append(hit)
        if sum(self._hits) < self.policy.required_hits:
            return False
        self.reset()
        return True

    def reset(self) -> None:
        self._hits.clear()
