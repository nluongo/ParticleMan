#!/usr/bin/env python3
"""
Report the number of input files and events available from a data config.

Uses the same Hydra configs as training, so any root config or ``data=`` group
works. Only file metadata is read (ROOT tree entries / HDF5 dataset shapes).

Usage:
    python scripts/count_events.py data=mc20_h5
    python scripts/count_events.py --config-name bbllv08 ++per_file=true
    python scripts/count_events.py data=mc20 'data.source.files=[/path/*.root]'

With ``data.max_events`` set, also reports how many events training would use
and the resulting train/val/test split sizes.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from particleman.config import register_configs
from particleman.data import count_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    per_file = bool(OmegaConf.select(cfg, "per_file", default=False))

    counts = count_events(cfg.data)

    if per_file:
        width = max((len(str(f.path)) for f in counts.per_file), default=4)
        logger.info(f"{'File':<{width}}  {'Events':>12}  Label")
        for f in counts.per_file:
            logger.info(f"{str(f.path):<{width}}  {f.n_events:>12,}  {f.label}")

    logger.info(f"Input files: {counts.n_files}")
    logger.info(f"Available events: {counts.total:,}")

    labels = sorted({f.label for f in counts.per_file})
    if labels != [-1]:
        for label in labels:
            n = sum(f.n_events for f in counts.per_file if f.label == label)
            logger.info(f"  label {label}: {n:,} events")

    max_events = cfg.data.get("max_events", None)
    n_used = counts.n_used(max_events)
    if n_used < counts.total:
        logger.info(f"Used with data.max_events={max_events}: {n_used:,}")

    # Same arithmetic as create_split_indices, without building the index list
    split = cfg.data.get("split", {})
    n_train = int(n_used * split.get("train", 0.8))
    n_val = int(n_used * split.get("val", 0.1))
    n_test = n_used - n_train - n_val
    logger.info(f"Split: {n_train:,} train, {n_val:,} val, {n_test:,} test")


if __name__ == "__main__":
    main()
