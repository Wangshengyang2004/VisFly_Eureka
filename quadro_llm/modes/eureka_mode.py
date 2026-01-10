from __future__ import annotations

from pathlib import Path

from ..core.models import OptimizationReport
from ..pipeline import EurekaPipeline
from ..eureka_visfly import EurekaVisFly


class EurekaMode:
    name = "eureka"

    def __init__(self, controller: EurekaVisFly, output_dir: str | Path):
        self._pipeline = EurekaPipeline(
            eureka_controller=controller,
            output_dir=str(output_dir),
        )

    def run(self) -> OptimizationReport:
        return self._pipeline.run_optimization()

