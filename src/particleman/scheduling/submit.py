"""Submit a training run as a PBS or Slurm batch job."""

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from particleman.scheduling.job_env import build_job_env
from particleman.scheduling.schedulers import check_launcher, get_scheduler

logger = logging.getLogger(__name__)


def to_dict(node: Any) -> Dict[str, Any]:
    """Resolve a config node into a plain dict (enums become strings)."""
    container = OmegaConf.to_container(node, resolve=True, enum_to_str=True)
    assert isinstance(container, dict)
    return {str(k): v for k, v in container.items()}


def format_env(env: Mapping[str, str]) -> str:
    """Format env vars as shell-quoted ``KEY=value`` lines."""
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def submit_job(
    cfg: DictConfig,
    hydra_args: Sequence[str],
    timestamp: str,
    repo_root: Path,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Submit a batch job that runs scripts/jobs/launch.sh with scripts/train.py.

    Args:
        cfg: Composed root config; its ``cluster``, ``resources``, ``job`` and
            ``output`` sections describe the job
        hydra_args: Hydra arguments for scripts/train.py in the job
            (including ``output.timestamp``)
        timestamp: Run timestamp, the first part of the run directory name
        repo_root: Absolute path of the repository
        dry_run: Log the submit command and job environment without submitting

    Returns:
        The job ID, or None for a dry run

    Raises:
        ValueError: If the cluster/resources settings can't be submitted
        RuntimeError: If the submit command fails
    """
    cluster = to_dict(cfg.cluster)
    resources = to_dict(cfg.resources)
    job = to_dict(cfg.job)
    output = to_dict(cfg.output)

    scheduler = get_scheduler(cluster["scheduler"])
    check_launcher(cluster["scheduler"], cluster["launcher"])
    if resources.get("gpus_per_node") is None:
        raise ValueError(
            "resources.gpus_per_node must be set to submit a job; pick a job "
            "size with resources=<name> (configs/resources/) or set "
            "resources.gpus_per_node"
        )

    job_env = build_job_env(
        cluster=cluster,
        resources=resources,
        job=job,
        output=output,
        repo_root=str(repo_root),
        train_script=str(repo_root / "scripts" / "train.py"),
        hydra_args=hydra_args,
        timestamp=timestamp,
    )
    job_script = repo_root / "scripts" / "jobs" / "launch.sh"
    cmd = scheduler.build_submit_command(
        cluster, resources, job, str(job_script), list(job_env)
    )

    logger.info(f"Scheduler: {scheduler.name}  Launcher: {cluster['launcher']}")
    logger.info(
        f"  Nodes: {resources['nnodes']}  GPUs/node: {resources['gpus_per_node']}"
        f"  Walltime: {resources['walltime']}"
    )
    logger.info(f"Submit command:\n  {shlex.join(cmd)}")
    logger.info(f"Job environment:\n{format_env(job_env)}")

    if dry_run:
        logger.info("Dry run: not submitting.")
        return None

    result = subprocess.run(  # nosec B603: command built from config, no shell
        cmd, capture_output=True, text=True, env={**os.environ, **job_env}
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Submission failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )

    job_id = scheduler.parse_job_id(result.stdout)
    logger.info(f"Submitted job {job_id}")

    # Pre-create the run directory the job will use, and record how it was
    # submitted. Imported here so submitting doesn't need torch at import time.
    from particleman.training.output_manager import OutputManager

    output_manager = OutputManager(
        output["experiment_name"],
        run_name=output.get("run_name") or f"{timestamp}_{scheduler.name}{job_id}",
        base_dir=output.get("output_dir") or "outputs",
    )
    submit_file = output_manager.run_dir / "submit_command.sh"
    submit_file.write_text(
        "#!/bin/bash\n# Submitted from "
        + os.getcwd()
        + "\n"
        + "\n".join(f"export {line}" for line in format_env(job_env).splitlines())
        + "\n"
        + shlex.join(cmd)
        + "\n"
    )
    logger.info(f"Run directory: {output_manager.run_dir}")
    return job_id
