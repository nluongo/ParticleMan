"""Job scheduler support (PBS, Slurm) and run-mode selection for scripts/run.py."""

from particleman.scheduling.job_env import build_job_env
from particleman.scheduling.run_mode import (
    MODES,
    forwarded_args,
    in_allocation,
    is_torchrun_worker,
    new_timestamp,
    resolve_mode,
)
from particleman.scheduling.schedulers import (
    LAUNCHERS_BY_SCHEDULER,
    PBSScheduler,
    Scheduler,
    SlurmScheduler,
    check_launcher,
    get_scheduler,
)
from particleman.scheduling.submit import format_env, submit_job, to_dict

__all__ = [
    "LAUNCHERS_BY_SCHEDULER",
    "MODES",
    "PBSScheduler",
    "Scheduler",
    "SlurmScheduler",
    "build_job_env",
    "check_launcher",
    "format_env",
    "forwarded_args",
    "get_scheduler",
    "in_allocation",
    "is_torchrun_worker",
    "new_timestamp",
    "resolve_mode",
    "submit_job",
    "to_dict",
]
