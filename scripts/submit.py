#!/usr/bin/env python3
"""
Batch submission wrapper for ParticleMan training (PBS or Slurm).

Accepts Hydra overrides for the submit config (cluster=, resources=, job.*)
and for the training config (data.*, model.*, training.*, etc.). Training
overrides are validated by composing the training config before submission,
then forwarded to the job via HYDRA_ARGS. The job runs scripts/jobs/launch.sh.

The cluster= group picks the site (scheduler, account, queue, launcher,
environment); resources= picks the job size. Individual fields can be
overridden with cluster.<field>=<value> / resources.<field>=<value>.

Usage:
    # Submit with defaults (configs/submit_config.yaml: pbs_example, single_gpu)
    python scripts/submit.py data=bbllv08 training.epochs=80

    # Use a top-level training config (instead of the default configs/config.yaml)
    python scripts/submit.py --config-name bbllv08

    # Slurm site, 8 GPUs on one node
    python scripts/submit.py cluster=slurm_example resources=multi_gpu data=bbllv08

    # Multi-node run
    python scripts/submit.py resources=multi_node data=bbllv08 training.epochs=200

    # Override individual settings
    python scripts/submit.py resources=multi_gpu resources.walltime=16:00:00 \\
        cluster.launcher=mpiexec job.name=bbll_test data=bbllv08

    # Dry run: print the submit command and job environment without submitting
    python scripts/submit.py --dry-run resources=multi_gpu data=bbllv08

Cluster configs live in configs/cluster/, job sizes in configs/resources/.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
src_path = REPO_ROOT / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from particleman.config import TrainConfig
from particleman.scheduling import build_job_env, check_launcher, get_scheduler
from particleman.training import OutputManager

CONFIG_DIR = REPO_ROOT / "configs"
JOB_SCRIPT = REPO_ROOT / "scripts" / "jobs" / "launch.sh"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

# Top-level keys that belong to the submit config rather than the training config
SUBMIT_KEYS = ("cluster", "resources", "job")


def split_overrides(args: List[str]) -> Tuple[List[str], List[str]]:
    """Separate submit-config overrides from training overrides."""
    submit, train = [], []
    for arg in args:
        key = arg.lstrip("+~").split("=", 1)[0].split(".", 1)[0]
        (submit if key in SUBMIT_KEYS else train).append(arg)
    return submit, train


def compose_submit_config(submit_overrides: List[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="submit_config", overrides=submit_overrides)


def compose_train_config(train_overrides: List[str], config_name: str = "config") -> DictConfig:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainConfig)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=train_overrides)


def format_env(env: Dict[str, str]) -> str:
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a ParticleMan training job to PBS or Slurm.",
        add_help=False,
    )
    parser.add_argument(
        "--config-name", "-cn",
        default="config",
        metavar="NAME",
        help="Top-level training config name in configs/ (default: config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the submit command and job environment without submitting",
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show this help message and exit",
    )

    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        print("\nAll other arguments are Hydra overrides, for example:")
        print("  cluster=slurm_example      Select a site (configs/cluster/)")
        print("  resources=multi_gpu        Select a job size (configs/resources/)")
        print("  resources.walltime=8:00:00 Override a single resource field")
        print("  cluster.launcher=mpiexec   Override the launcher")
        print("  job.name=my_run            Override the job name")
        print("  data=bbllv08               Training config group override")
        print("  training.epochs=80         Training field override")
        print("\nTraining configs: configs/<name>.yaml  (selected with --config-name/-cn)")
        sys.exit(0)

    submit_overrides, train_overrides = split_overrides(remaining)

    submit_cfg = OmegaConf.to_container(compose_submit_config(submit_overrides), resolve=True)
    cluster, resources, job = submit_cfg["cluster"], submit_cfg["resources"], submit_cfg["job"]

    try:
        scheduler = get_scheduler(cluster["scheduler"])
        check_launcher(cluster["scheduler"], cluster["launcher"])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("Validating training config...", flush=True)
    train_config = compose_train_config(train_overrides, config_name=args.config_name)

    # Set a timestamp value to coordinate logging with submission
    timestamp_element = train_config.output.get("timestamp", None)
    if not timestamp_element:
        timestamp = str(round(time.time()))
        train_overrides += [f"output.timestamp={timestamp}"]
    else:
        timestamp = str(timestamp_element)

    hydra_args = ([f"--config-name={args.config_name}"] if args.config_name != "config" else []) + train_overrides
    job_env = build_job_env(
        cluster=cluster,
        resources=resources,
        job=job,
        output=OmegaConf.to_container(train_config.output, resolve=True),
        repo_root=str(REPO_ROOT),
        train_script=str(TRAIN_SCRIPT),
        hydra_args=hydra_args,
        timestamp=timestamp,
    )
    cmd = scheduler.build_submit_command(
        cluster, resources, job, str(JOB_SCRIPT), list(job_env)
    )

    print(f"Scheduler: {scheduler.name}  Launcher: {cluster['launcher']}")
    print(
        f"  Nodes: {resources['nnodes']}  GPUs/node: {resources['gpus_per_node']}"
        f"  Walltime: {resources['walltime']}"
    )
    print("  Training overrides:", " ".join(train_overrides) or "(none — using default config)")
    print()
    print("Submit command:")
    print(" ", shlex.join(cmd))
    print()
    print("Job environment:")
    print(format_env(job_env))
    print()

    if args.dry_run:
        print("Dry run: not submitting.")
        return

    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **job_env})
    if result.returncode != 0:
        print(f"ERROR: submission failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr or result.stdout, file=sys.stderr)
        sys.exit(result.returncode)

    job_id = scheduler.parse_job_id(result.stdout)
    print(f"Submitted job {job_id}")

    # Pre-create the run directory the job will use, and record how it was submitted
    output_manager = OutputManager(
        train_config.output.experiment_name,
        run_name=train_config.output.get("run_name") or f"{timestamp}_{scheduler.name}{job_id}",
        base_dir=train_config.output.get("output_dir") or "outputs",
    )
    submit_file = output_manager.run_dir / "submit_command.sh"
    submit_file.write_text(
        "#!/bin/bash\n# Submitted from " + os.getcwd() + "\n"
        + "\n".join(f"export {line}" for line in format_env(job_env).splitlines())
        + "\n" + shlex.join(cmd) + "\n"
    )
    print(f"Run directory: {output_manager.run_dir}")


if __name__ == "__main__":
    main()
