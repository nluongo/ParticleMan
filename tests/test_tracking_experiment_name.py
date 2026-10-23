"""Tests for the MLflow experiment-name prefix."""

from omegaconf import OmegaConf

from particleman.training.pipeline import tracking_experiment_name


def _output(logger: str, prefix: object) -> object:
    return OmegaConf.create(
        {"experiment_logger": logger, "mlflow_experiment_prefix": prefix}
    )


def test_prefix_applied_for_mlflow() -> None:
    cfg = _output("mlflow", "/Users/me@example.com/")
    assert tracking_experiment_name(cfg, "ParticleMan") == (
        "/Users/me@example.com/ParticleMan"
    )


def test_prefix_ignored_for_other_loggers() -> None:
    assert tracking_experiment_name(_output("comet", "/Users/x"), "P") == "P"


def test_no_prefix() -> None:
    assert tracking_experiment_name(_output("mlflow", None), "P") == "P"
