#!/usr/bin/env python3
"""
PBS batch submission wrapper for ParticleMan training.

Accepts the same Hydra overrides as scripts/train.py and submits them as a
PBS job, with cluster resources specified by a PBS profile config.

Usage:
    # Submit with default PBS resources (single GPU)
    python scripts/submit.py data=bbllv08 training.epochs=80

    # Use a different PBS resource profile
    python scripts/submit.py --pbs multi_gpu data=bbllv08 model.d_model=256

    # Multi-node run
    python scripts/submit.py --pbs multi_node data=bbllv08 training.epochs=200

    # Dry run: print the qsub command without submitting
    python scripts/submit.py --dry-run --pbs multi_gpu data=bbllv08 training.epochs=80

    # Override the PBS job name
    python scripts/submit.py --job-name bbll_test data=bbllv08 training=quick_test

PBS profiles live in configs/pbs/<name>.yaml.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
PBS_CONFIG_DIR = REPO_ROOT / "configs" / "pbs"
PBS_SCRIPT = REPO_ROOT / "scripts" / "pbs" / "submit_training.pbs"


def load_pbs_config(name: str) -> dict:
    config_path = PBS_CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in PBS_CONFIG_DIR.glob("*.yaml"))
        print(f"ERROR: PBS config '{name}' not found.", file=sys.stderr)
        print(f"Available configs: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_select_string(cfg: dict) -> str:
    return (
        f"select={cfg['nnodes']}"
        f":ncpus={cfg['ncpus']}"
        f":ngpus={cfg['ngpus']}"
        f":mem={cfg['mem']}"
    )


def build_qsub_command(cfg: dict, hydra_args: list, job_name_override: str | None) -> list:
    job_name = job_name_override or cfg.get("job_name", "particleman_train")
    hydra_args_str = " ".join(hydra_args)

    v_vars = ",".join([
        f"HYDRA_ARGS={hydra_args_str}",
        f"NGPUS_PER_NODE={cfg['ngpus']}",
        f"MASTER_PORT={cfg.get('master_port', 29500)}",
        f"OMP_NUM_THREADS={cfg.get('omp_num_threads', 8)}",
        f"NCCL_DEBUG={cfg.get('nccl_debug', 'WARN')}",
    ])

    return [
        "qsub",
        "-A", cfg["account"],
        "-N", job_name,
        "-q", cfg["queue"],
        "-l", f"walltime={cfg['walltime']}",
        "-l", build_select_string(cfg),
        "-j", "oe",
        "-v", v_vars,
        str(PBS_SCRIPT),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Submit a ParticleMan training job to PBS.",
    )
    parser.add_argument(
        "--pbs",
        default="default",
        metavar="NAME",
        help="PBS resource profile from configs/pbs/<NAME>.yaml (default: 'default')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the qsub command without submitting",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        metavar="NAME",
        help="Override the PBS job name (default: from PBS config yaml)",
    )

    args, hydra_args = parser.parse_known_args()

    cfg = load_pbs_config(args.pbs)
    cmd = build_qsub_command(cfg, hydra_args, args.job_name)

    # Print a readable version of the command
    print("Submitting with PBS profile:", args.pbs)
    print(f"  Nodes: {cfg['nnodes']}  GPUs/node: {cfg['ngpus']}  Walltime: {cfg['walltime']}")
    if hydra_args:
        print("  Hydra overrides:", " ".join(hydra_args))
    else:
        print("  Hydra overrides: (none — using default config)")
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
