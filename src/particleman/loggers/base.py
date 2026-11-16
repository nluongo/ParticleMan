from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

class BaseLogger(ABC):
    """Abstract base class for logging and experiment tracking."""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and finish any pending operations."""
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload a local file to the run. Loggers without artifacts ignore it."""
        pass

    def log_artifacts(
        self, local_dir: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Upload a local directory to the run. Loggers without artifacts ignore it."""
        pass

class NoOpLogger(BaseLogger):
    """A logger that does nothing. Used as a default."""
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def close(self) -> None:
        pass