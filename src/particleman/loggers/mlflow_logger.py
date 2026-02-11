from typing import Any, Dict, Optional, TYPE_CHECKING
from .base import BaseLogger

if TYPE_CHECKING:
    import mlflow

class MLFlowLogger(BaseLogger):
    """Logger for MLFlow."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Initialize MLFlow experiment.
        Args:
            experiment_name: Name of the MLFlow experiment
            run_name: Name for this experiment run (optional)

        Raises:
            ImportError: If mlflow is not installed
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "mlflow is not installed. Please install it to use MLFlowLogger."
            ) from e
        # Store mlflow module to use in other methods, needed for conditional import
        self._mlflow = mlflow

        self._mlflow.set_experiment(experiment_name)
        self._mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        # mlflow has a limit on param value length, so we truncate
        truncated_params = {}
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 250:
                truncated_params[key] = value[:247] + "..."
            else:
                truncated_params[key] = value
        self._mlflow.log_params(truncated_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._mlflow.log_metrics(metrics, step=step)

    def close(self):
        self._mlflow.end_run()