"""Tests for scripts/run.py mode selection (particleman.scheduling.run_mode)."""

from typing import Dict, Optional

import pytest

from particleman.scheduling import forwarded_args, resolve_mode

TORCHRUN_ENV = {"LOCAL_RANK": "0", "WORLD_SIZE": "4", "RANK": "0"}
PBS_JOB_ENV = {"PBS_JOBID": "123.pbs"}
SLURM_JOB_ENV = {"SLURM_JOB_ID": "456"}


def _no_gpu_count() -> int:
    raise AssertionError("local_gpu_count should not be called")


@pytest.mark.parametrize(
    "scheduler,nnodes,gpus,env,expected",
    [
        # No scheduler: train unless more than one GPU/node is wanted
        ("none", 1, 1, {}, "train"),
        ("none", 1, 4, {}, "launch"),
        ("none", 2, 1, {}, "launch"),
        # Scheduler configured, outside a job: submit
        ("slurm", 1, 1, {}, "submit"),
        ("pbs", 2, 4, {}, "submit"),
        # Inside a job (batch or interactive): launch, whatever the size
        ("slurm", 1, 1, SLURM_JOB_ENV, "launch"),
        ("pbs", 1, 1, PBS_JOB_ENV, "launch"),
        ("none", 1, 1, SLURM_JOB_ENV, "launch"),
        # Started by torchrun: always train
        ("none", 1, 4, TORCHRUN_ENV, "train"),
        ("slurm", 2, 4, {**TORCHRUN_ENV, **SLURM_JOB_ENV}, "train"),
        ("none", 1, 4, {"TORCHELASTIC_RUN_ID": "x"}, "train"),
    ],
)
def test_auto_mode(
    scheduler: str, nnodes: int, gpus: int, env: Dict[str, str], expected: str
) -> None:
    mode = resolve_mode("auto", scheduler, nnodes, gpus, env, _no_gpu_count)
    assert mode == expected


@pytest.mark.parametrize(
    "local_gpus,expected", [(0, "train"), (1, "train"), (8, "launch")]
)
def test_auto_mode_detects_local_gpus(local_gpus: int, expected: str) -> None:
    mode = resolve_mode("auto", "none", 1, None, {}, lambda: local_gpus)
    assert mode == expected


@pytest.mark.parametrize("mode", ["train", "launch", "submit"])
def test_explicit_mode_skips_detection(mode: str) -> None:
    assert resolve_mode(mode, "slurm", 1, None, TORCHRUN_ENV, _no_gpu_count) == mode


def test_submit_needs_scheduler() -> None:
    with pytest.raises(ValueError, match="cluster="):
        resolve_mode("submit", "none", 1, 1, {}, _no_gpu_count)


def test_unknown_mode() -> None:
    with pytest.raises(ValueError, match="job.mode"):
        resolve_mode("run", "none", 1, 1, {}, _no_gpu_count)


@pytest.mark.parametrize(
    "config_name,timestamp,expected",
    [
        (None, None, ["data=x"]),
        ("config", "17", ["data=x", "output.timestamp=17"]),
        ("bbllv08", "17", ["--config-name=bbllv08", "data=x", "output.timestamp=17"]),
    ],
)
def test_forwarded_args(
    config_name: Optional[str], timestamp: Optional[str], expected: list
) -> None:
    assert forwarded_args(["data=x"], config_name, timestamp) == expected
