import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from .base import BaseLogger

if TYPE_CHECKING:
    from comet_ml import Experiment

class CometLogger(BaseLogger):
    """Logger for Comet ML."""

    def __init__(
        self,
        project_name: str,
        workspace: Optional[str] = None,
        run_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize CometML experiment.

        Args:
            project_name: Name of the Comet project
            workspace: Comet workspace (optional, uses default if not specified)
            run_name: Name for this experiment run (optional)
            api_key: Comet API key (optional, uses COMET_API_KEY env var if not specified)
        """
        try:
            from comet_ml import Experiment
        except ImportError as e:
            raise ImportError(
                "Comet ML is not installed. Please install it with 'pip install comet-ml'"
            ) from e
        self._experiment = Experiment

        self._experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        if run_name:
            self._experiment.set_name(run_name)

    def log_params(self, params: Dict[str, Any]):
        self._experiment.log_parameters(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._experiment.log_metrics(metrics, step=step)

    def close(self):
        self._experiment.end()
