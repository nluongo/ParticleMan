"""Decide how scripts/run.py runs a job: train, launch or submit."""

import time
from typing import Callable, List, Mapping, Optional, Sequence

#: Values of ``job.mode``
MODES = ("auto", "train", "launch", "submit")

#: Override key scripts/run.py adds to pass the run timestamp to the workers
TIMESTAMP_KEY = "output.timestamp"


def in_allocation(env: Mapping[str, str]) -> bool:
    """Whether this process runs inside a PBS or Slurm job (batch or interactive)."""
    return bool(env.get("PBS_JOBID") or env.get("SLURM_JOB_ID"))


def is_torchrun_worker(env: Mapping[str, str]) -> bool:
    """Whether this process was started by torchrun."""
    return "TORCHELASTIC_RUN_ID" in env or ("LOCAL_RANK" in env and "WORLD_SIZE" in env)


def resolve_mode(
    mode: str,
    scheduler: str,
    nnodes: int,
    gpus_per_node: Optional[int],
    env: Mapping[str, str],
    local_gpu_count: Callable[[], int],
) -> str:
    """
    Resolve ``job.mode`` to ``"train"``, ``"launch"`` or ``"submit"``.

    With ``mode="auto"``:

    1. train, when already running as a torchrun worker
    2. submit, when a scheduler is configured and this isn't a job already
    3. launch, inside a job (launch.sh handles nodes and GPUs), or when the
       run needs more than one node or GPU
    4. train otherwise

    Args:
        mode: ``job.mode`` (one of ``MODES``)
        scheduler: ``cluster.scheduler`` ("none", "pbs" or "slurm")
        nnodes: ``resources.nnodes``
        gpus_per_node: ``resources.gpus_per_node``; None means all local GPUs
        env: Environment mapping (usually os.environ)
        local_gpu_count: Returns the number of local GPUs; only called when
            ``gpus_per_node`` is None

    Returns:
        The resolved mode

    Raises:
        ValueError: For an unknown mode, or submit without a scheduler
    """
    if mode not in MODES:
        raise ValueError(f"Unknown job.mode={mode!r}; expected one of {list(MODES)}")
    if mode == "submit" and scheduler == "none":
        raise ValueError(
            "job.mode=submit needs a batch scheduler; select a site with "
            "cluster=<name> (configs/cluster/)"
        )
    if mode != "auto":
        return mode

    if is_torchrun_worker(env):
        return "train"
    if scheduler != "none" and not in_allocation(env):
        return "submit"
    if in_allocation(env) or nnodes > 1:
        return "launch"
    gpus = gpus_per_node if gpus_per_node is not None else local_gpu_count()
    return "launch" if gpus > 1 else "train"


def forwarded_args(
    overrides: Sequence[str],
    config_name: Optional[str],
    timestamp: Optional[str] = None,
) -> List[str]:
    """
    Build the Hydra arguments for scripts/train.py in the launched workers.

    Args:
        overrides: The command-line overrides given to scripts/run.py
        config_name: Root config name (``--config-name``); "config" or None
            for the default
        timestamp: Run timestamp to add as ``output.timestamp``; None to leave
            the overrides as they are

    Returns:
        Argument list for scripts/train.py
    """
    args = [] if config_name in (None, "config") else [f"--config-name={config_name}"]
    args += list(overrides)
    if timestamp is not None:
        args.append(f"{TIMESTAMP_KEY}={timestamp}")
    return args


def new_timestamp() -> str:
    """Timestamp portion of the run name (seconds since the epoch)."""
    return str(round(time.time()))
