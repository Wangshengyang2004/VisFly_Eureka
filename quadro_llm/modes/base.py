from __future__ import annotations

from typing import Protocol

from ..core.models import OptimizationReport


class OptimizationMode(Protocol):
    """A runnable optimization mode (e.g. Eureka, TempTuner)."""

    name: str

    def run(self) -> OptimizationReport: ...

