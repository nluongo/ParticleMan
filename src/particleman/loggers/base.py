from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseLogger(ABC):
    """Abstract base class for logging and experiment tracking."""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log a dictionary of hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dictionary of metrics."""
        pass

    @abstractmethod
    def close(self):
        """Close the logger and finish any pending operations."""
        pass

class NoOpLogger(BaseLogger):
    """A logger that does nothing. Used as a default."""
    def log_params(self, params: Dict[str, Any]):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def close(self):
        pass