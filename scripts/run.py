#!/usr/bin/env python3
"""
Single entry point for ParticleMan training.

Composes the same Hydra config as scripts/train.py, then, depending on
job.mode (configs/job/default.yaml), either:

    train   trains in this process (scripts/train.py's code path)
    launch  starts scripts/train.py workers with scripts/jobs/launch.sh
            (torchrun, srun or mpiexec; one or more nodes)
    submit  submits a PBS/Slurm batch job that runs launch.sh

With job.mode=auto (the default) the mode is picked for you: submit when
cluster.scheduler is set and this isn't already a job, launch inside a job or
when using more than one GPU, and train otherwise. See
particleman.scheduling.resolve_mode.

Usage:
    # Train locally (default cluster=local)
    python scripts/run.py data=bbllv08 training=quick_test

    # All GPUs on this machine / in this allocation (qsub -I, salloc)
    python scripts/run.py resources.gpus_per_node=4 data=bbllv08

    # Submit a batch job
    python scripts/run.py cluster=swing resources=multi_gpu data=bbllv08
    python scripts/run.py -cn bbllv08 cluster=slurm_example resources=multi_node

    # Show the submit command and job environment without submitting
    python scripts/run.py cluster=swing resources=single_gpu job.dry_run=true

    # Force a mode
    python scripts/run.py job.mode=train ...

    # Print the resolved config
    python scripts/run.py --cfg job
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
src_path = REPO_ROOT / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from particleman.config import register_configs
from particleman.scheduling import (
    build_job_env,
    forwarded_args,
    new_timestamp,
    resolve_mode,
    submit_job,
    to_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

register_configs()

LAUNCH_SCRIPT = REPO_ROOT / "scripts" / "jobs" / "launch.sh"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def local_gpu_count() -> int:
    """Number of GPUs visible to this process."""
    import torch

    return torch.cuda.device_count()


def launch(cfg: DictConfig, hydra_args: list, timestamp: str) -> None:
    """Start scripts/train.py workers with scripts/jobs/launch.sh."""
    job_env = build_job_env(
        cluster=to_dict(cfg.cluster),
        resources=to_dict(cfg.resources),
        job=to_dict(cfg.job),
        output=to_dict(cfg.output),
        repo_root=str(REPO_ROOT),
        train_script=str(TRAIN_SCRIPT),
        hydra_args=hydra_args,
        timestamp=timestamp,
    )
    cmd = ["bash", str(LAUNCH_SCRIPT), str(TRAIN_SCRIPT), *hydra_args]
    logger.info(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, env={**os.environ, **job_env})  # nosec B603
    if result.returncode != 0:
        sys.exit(result.returncode)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train, launch or submit, depending on job.mode."""
    try:
        mode = resolve_mode(
            mode=cfg.job.mode,
            scheduler=cfg.cluster.scheduler,
            nnodes=cfg.resources.nnodes,
            gpus_per_node=cfg.resources.gpus_per_node,
            env=os.environ,
            local_gpu_count=local_gpu_count,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Run mode: {mode} (job.mode={cfg.job.mode})")

    if mode == "train":
        from particleman.training import run_training

        run_training(cfg)
        return

    # Workers get the same root config and overrides, plus a shared run
    # timestamp so the run directory is known before they start
    timestamp = cfg.output.timestamp or new_timestamp()
    hydra_cfg = HydraConfig.get()
    hydra_args = forwarded_args(
        hydra_cfg.overrides.task,
        hydra_cfg.job.config_name,
        timestamp=None if cfg.output.timestamp else timestamp,
    )

    if mode == "launch":
        launch(cfg, hydra_args, str(timestamp))
        return

    try:
        submit_job(
            cfg,
            hydra_args,
            timestamp=str(timestamp),
            repo_root=REPO_ROOT,
            dry_run=cfg.job.dry_run,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
