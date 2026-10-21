#!/usr/bin/env python3
"""
Train ParticleMan in this process (or in each worker of a distributed launch).

This is the barebones trainer. scripts/run.py is the usual entry point: it
submits a batch job or starts workers with scripts/jobs/launch.sh when
needed, and the workers run this script. Configuration is managed via YAML
files and command-line overrides.

Usage:
    # Use default config (configs/config.yaml)
    python scripts/train.py

    # Override values via CLI (note: key=value syntax)
    python scripts/train.py training.epochs=20 model.d_model=256 model.d_ff=512

    # Use different sub-configs
    python scripts/train.py data=physlite model=default

    # Override data settings
    python scripts/train.py data.max_events=5000 data.preprocessing.pt_cut=10.0

    # Print resolved config without running
    python scripts/train.py --cfg job

    # Multi-GPU with torchrun
    torchrun --standalone --nproc_per_node=4 scripts/train.py training.batch_size=64

    # Multi-node with torchrun
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<master_ip> \\
        scripts/train.py
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from particleman.config import register_configs
from particleman.training import run_training

# Setup basic console logging (file logging added later by OutputManager)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    run_training(cfg)


if __name__ == "__main__":
    main()
