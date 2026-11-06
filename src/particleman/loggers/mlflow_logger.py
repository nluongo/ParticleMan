from typing import Any, Dict, Optional, TYPE_CHECKING
from .base import BaseLogger

if TYPE_CHECKING:
    import mlflow

class MLFlowLogger(BaseLogger):
    """Logger for MLFlow."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        async_logging: bool = False,
    ):
        """
        Initialize MLFlow experiment.
        Args:
            experiment_name: Name of the MLFlow experiment
            run_name: Name for this experiment run (optional)
            async_logging: Log metrics and params on MLflow's background thread
                instead of blocking on each request. ``close`` flushes the queue.

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
        self._async = async_logging

        self._mlflow.set_experiment(experiment_name)
        self._mlflow.enable_system_metrics_logging()
        self._mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        # mlflow has a limit on param value length, so we truncate
        truncated_params = {}
        for key, value in params.items():
            text = value if isinstance(value, str) else str(value)
            if len(text) > 250:
                truncated_params[key] = text[:247] + "..."
            else:
                truncated_params[key] = value
        # mlflow accepts at most 100 params per batch request
        items = list(truncated_params.items())
        for start in range(0, len(items), 100):
            self._mlflow.log_params(
                dict(items[start : start + 100]), synchronous=not self._async
            )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._mlflow.log_metrics(metrics, step=step, synchronous=not self._async)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        self._mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def close(self):
        if self._async:
            # Send queued metrics/params before the run is marked FINISHED
            self._mlflow.flush_async_logging()
        self._mlflow.end_run()
