#!/usr/bin/env python3
"""
PBS batch submission wrapper for ParticleMan training.

Accepts Hydra overrides for both PBS resources (pbs.*) and training
configuration (data.*, model.*, training.*, etc.).  Training overrides
are validated against the training config schema before submission and
then forwarded to the PBS job via HYDRA_ARGS.

PBS profiles are selected with the pbs= config group override.
Individual PBS fields can be overridden with pbs.<field>=<value>.

Usage:
    # Submit with default PBS resources (single GPU)
    python scripts/submit.py data=bbllv08 training.epochs=80

    # Use a top-level training config (instead of the default configs/config.yaml)
    python scripts/submit.py --config-name bbllv08

    # Use a different PBS resource profile
    python scripts/submit.py pbs=multi_gpu data=bbllv08 model.d_model=256

    # Multi-node run
    python scripts/submit.py pbs=multi_node data=bbllv08 training.epochs=200

    # Override individual PBS settings
    python scripts/submit.py pbs=multi_gpu pbs.walltime=16:00:00 data=bbllv08

    # Override the PBS job name
    python scripts/submit.py pbs.job_name=bbll_test data=bbllv08 training=quick_test

    # Dry run: print the qsub command without submitting
    python scripts/submit.py --dry-run pbs=multi_gpu data=bbllv08 training.epochs=80

PBS profiles live in configs/pbs/<name>.yaml.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
PBS_SCRIPT = REPO_ROOT / "scripts" / "pbs" / "submit_training.pbs"

# Prefixes that belong to the submit (PBS) config rather than the training config
_SUBMIT_PREFIXES = ("pbs=", "pbs.", "+pbs", "~pbs", "++pbs")


def split_overrides(args: list[str]) -> tuple[list[str], list[str]]:
    """Separate PBS-level overrides from training overrides."""
    submit, train = [], []
    for arg in args:
        if any(arg.startswith(p) for p in _SUBMIT_PREFIXES):
            submit.append(arg)
        else:
            train.append(arg)
    return submit, train


def compose_submit_config(submit_overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="submit_config", overrides=submit_overrides)


def compose_train_config(train_overrides: list[str], config_name: str = "config"):
    # Register TrainConfig so Hydra validates against the structured schema
    src_path = REPO_ROOT / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from particleman.config import TrainConfig
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainConfig)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=train_overrides)


def build_select_string(pbs) -> str:
    out = ""
    if pbs.get("nnodes", None):
        out += f"select={pbs.nnodes}:"
    if pbs.get("ncpus", None):
        out += f"ncpus={pbs.ncpus}:"
    if pbs.get("ngpus", None):
        out += f"ngpus={pbs.ngpus}:"
    if pbs.get("mem", None):
        out += f"mem={pbs.mem}:"
    out = out[:-1]
    return out


def build_qsub_command(pbs, train_overrides: list[str], config_name: str = "config") -> list[str]:
    cn_prefix = [f"--config-name {config_name}"] if config_name != "config" else []
    hydra_args_str = " ".join(cn_prefix + train_overrides)

    v_vars = ",".join([
        f"HYDRA_ARGS={hydra_args_str}",
        f"NGPUS_PER_NODE={pbs.ngpus}",
        f"MASTER_PORT={pbs.master_port}",
        f"OMP_NUM_THREADS={pbs.omp_num_threads}",
        f"NCCL_DEBUG={pbs.nccl_debug}",
        f"LAUNCHER={pbs.launcher}",
    ])

    return [
        "qsub",
        "-A", pbs.account,
        "-N", pbs.job_name,
        "-q", pbs.queue,
        "-l", f"walltime={pbs.walltime}",
        "-l", build_select_string(pbs),
        #"-j", "oe",
        "-v", v_vars,
        str(PBS_SCRIPT),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Submit a ParticleMan training job to PBS.",
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
        help="Print the qsub command without submitting",
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
        print("  pbs=multi_gpu              Select a PBS resource profile")
        print("  pbs.walltime=8:00:00       Override a single PBS field")
        print("  pbs.job_name=my_run        Override the PBS job name")
        print("  data=bbllv08               Training config group override")
        print("  training.epochs=80         Training field override")
        print("\nPBS profiles: configs/pbs/<name>.yaml")
        print("Training configs: configs/<name>.yaml  (selected with --config-name/-cn)")
        sys.exit(0)

    submit_overrides, train_overrides = split_overrides(remaining)

    # Compose and validate PBS config via Hydra
    submit_cfg = compose_submit_config(submit_overrides)
    pbs = submit_cfg.pbs

    # Derive the profile name for display (first bare pbs=NAME arg, or "default")
    pbs_profile = next(
        (o.split("=", 1)[1] for o in submit_overrides if o.startswith("pbs=") and "." not in o.split("=")[0]),
        "default",
    )

    # Compose and validate training config via Hydra
    print("Validating training config...", flush=True)
    compose_train_config(train_overrides, config_name=args.config_name)

    cmd = build_qsub_command(pbs, train_overrides, config_name=args.config_name)

    print(f"Submitting with PBS profile: {pbs_profile}")
    print(f"  Nodes: {pbs.nnodes}  GPUs/node: {pbs.ngpus}  Walltime: {pbs.walltime}")
    if train_overrides:
        print("  Training overrides:", " ".join(train_overrides))
    else:
        print("  Training overrides: (none — using default config)")
    print()
    print("qsub command:")
    print(" ", " ".join(cmd))
    print()

    if args.dry_run:
        print("Dry run: not submitting.")
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
