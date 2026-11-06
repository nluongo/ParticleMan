"""Tests for MLFlowLogger's synchronous/asynchronous logging switch."""

import sys
from unittest.mock import MagicMock, call

import pytest

from particleman.loggers.logger_utils import create_logger
from particleman.loggers.mlflow_logger import MLFlowLogger


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the mlflow module with a mock so no tracking server is needed."""
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", mock)
    return mock


@pytest.mark.parametrize("async_logging", [True, False])
def test_metrics_and_params_synchronous_flag(
    fake_mlflow: MagicMock, async_logging: bool
) -> None:
    lg = MLFlowLogger("exp", run_name="run", async_logging=async_logging)
    lg.log_metrics({"loss": 1.0}, step=3)
    lg.log_params({"a": 1})

    fake_mlflow.log_metrics.assert_called_once_with(
        {"loss": 1.0}, step=3, synchronous=not async_logging
    )
    fake_mlflow.log_params.assert_called_once_with(
        {"a": 1}, synchronous=not async_logging
    )


def test_default_is_synchronous(fake_mlflow: MagicMock) -> None:
    lg = MLFlowLogger("exp")
    lg.log_metrics({"loss": 1.0})
    assert fake_mlflow.log_metrics.call_args.kwargs["synchronous"] is True


def test_close_flushes_before_end_run_when_async(fake_mlflow: MagicMock) -> None:
    MLFlowLogger("exp", async_logging=True).close()
    calls = [c for c in fake_mlflow.mock_calls if c[0] in ("flush_async_logging", "end_run")]
    assert calls == [call.flush_async_logging(), call.end_run()]


def test_close_does_not_flush_when_sync(fake_mlflow: MagicMock) -> None:
    MLFlowLogger("exp", async_logging=False).close()
    fake_mlflow.flush_async_logging.assert_not_called()
    fake_mlflow.end_run.assert_called_once()


def test_create_logger_passes_async_to_mlflow(fake_mlflow: MagicMock) -> None:
    lg = create_logger("mlflow", "exp", async_logging=True, run_name="r")
    assert isinstance(lg, MLFlowLogger)
    lg.log_metrics({"loss": 1.0})
    assert fake_mlflow.log_metrics.call_args.kwargs["synchronous"] is False


def test_create_logger_does_not_pass_async_to_comet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import particleman.loggers.comet_logger as comet_module

    fake_cls = MagicMock()
    monkeypatch.setattr(comet_module, "CometLogger", fake_cls)
    create_logger("comet", "exp", async_logging=True, run_name="r")
    fake_cls.assert_called_once_with("exp", run_name="r")
