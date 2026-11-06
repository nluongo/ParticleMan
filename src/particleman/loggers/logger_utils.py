import logging
from .base import BaseLogger, NoOpLogger

logger = logging.getLogger(__name__)

def create_logger(
    logger_type: str,
    experiment_name: str,
    async_logging: bool = False,
    **kwargs,
) -> BaseLogger:
    """Factory function to create a logger based on the specified type.

    Args:
        logger_type: "mlflow", "comet" or "noop".
        experiment_name: Experiment to log the run under.
        async_logging: Log without blocking training. Only MLflow supports it;
            other loggers ignore it.
        **kwargs: Passed to the logger's constructor (e.g. ``run_name``).
    """
    if logger_type == "noop":
        logger.info("Using NoOpLogger (no logging will be performed).")
        return NoOpLogger()
    elif logger_type == "mlflow":
        from .mlflow_logger import MLFlowLogger
        logger.info("Using MLFlowLogger for experiment tracking.")
        return MLFlowLogger(experiment_name, async_logging=async_logging, **kwargs)
    elif logger_type == "comet":
        from .comet_logger import CometLogger
        logger.info("Using CometLogger for experiment tracking.")
        return CometLogger(experiment_name, **kwargs)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")