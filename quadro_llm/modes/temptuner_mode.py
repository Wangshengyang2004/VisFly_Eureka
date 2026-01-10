from __future__ import annotations

from pathlib import Path

from ..core.models import OptimizationReport
from ..eureka_visfly import EurekaVisFly
from ..pipeline import EurekaPipeline


class TempTunerMode:
    """
    Coefficient tuning optimization mode.
    
    This mode uses a fixed reward function structure (from envs/NavigationEnv.py)
    and only allows LLM to tune the coefficients. The reward function structure
    itself is never modified by the LLM.
    
    Key differences from Eureka mode:
    - LLM only outputs coefficient values (JSON or key-value pairs)
    - Reward function structure is fixed and generated from coefficients
    - Designed for navigation + BPTT experiments
    """

    name = "temptuner"

    def __init__(self, controller: EurekaVisFly, output_dir: str | Path):
        # The controller should already have use_coefficient_tuning=True set
        # via bootstrap.py when mode=temptuner
        self._pipeline = EurekaPipeline(
            eureka_controller=controller,
            output_dir=str(output_dir),
        )

    def run(self) -> OptimizationReport:
        return self._pipeline.run_optimization()




