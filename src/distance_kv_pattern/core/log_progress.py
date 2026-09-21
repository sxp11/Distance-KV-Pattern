"""Low-noise progress bars designed for persistent job log files."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, TextIO


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class LogProgressBar:
    """Print a durable progress-bar line at coarse milestones.

    Unlike an interactive tqdm bar, every refresh is a complete newline.  It
    therefore renders cleanly in ``run_*.log`` and when the log is tailed.
    Milestones limit normal output to roughly 20 lines per long-running loop;
    the time interval also provides a heartbeat when individual items are
    unusually slow.
    """

    total: int
    label: str
    unit: str = "items"
    width: int = 28
    milestones: int = 20
    min_interval_seconds: float = 120.0
    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stdout, repr=False)
    time_fn: Callable[[], float] = field(default=time.perf_counter, repr=False)
    _started_at: float = field(init=False, repr=False)
    _last_printed_at: float = field(init=False, repr=False)
    _last_bucket: int = field(init=False, default=-1, repr=False)
    _last_completed: int = field(init=False, default=-1, repr=False)

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("progress total must be positive")
        if not self.label:
            raise ValueError("progress label cannot be empty")
        if self.width <= 0 or self.milestones <= 0:
            raise ValueError("progress width and milestones must be positive")
        if self.min_interval_seconds < 0:
            raise ValueError("progress interval cannot be negative")
        now = self.time_fn()
        self._started_at = now
        self._last_printed_at = now

    def start(self, *, completed: int = 0, detail: str | None = None) -> None:
        self.update(completed, detail=detail, force=True)

    def update(
        self,
        completed: int,
        *,
        detail: str | None = None,
        force: bool = False,
    ) -> None:
        if not 0 <= completed <= self.total:
            raise ValueError("completed progress lies outside [0, total]")
        if completed < self._last_completed:
            raise ValueError("progress cannot move backwards")
        if not self.enabled:
            self._last_completed = completed
            return

        now = self.time_fn()
        bucket = min(
            self.milestones,
            completed * self.milestones // self.total,
        )
        due_to_milestone = bucket > self._last_bucket
        due_to_time = now - self._last_printed_at >= self.min_interval_seconds
        if not (force or due_to_milestone or due_to_time or completed == self.total):
            self._last_completed = completed
            return

        fraction = completed / self.total
        filled = min(self.width, int(fraction * self.width))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = max(0.0, now - self._started_at)
        eta = None if completed == 0 else elapsed * (self.total - completed) / completed
        line = (
            f"progress {self.label} [{bar}] {fraction * 100:5.1f}% | "
            f"{completed}/{self.total} {self.unit} | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
        )
        if detail:
            line += f" | {detail}"
        print(line, file=self.stream, flush=True)
        self._last_bucket = bucket
        self._last_printed_at = now
        self._last_completed = completed

    def finish(self, *, detail: str | None = None) -> None:
        self.update(self.total, detail=detail, force=True)
