#!/usr/bin/env python3
"""
Convert ROOT ntuples to padded HDF5 files using uproot (no ROOT install needed).

Uses the selected data config (``data=<name>``) for the input files, tree name,
particle collections, particle ID mapping and preprocessing (pt_scale, pt/eta
cuts, max_particles). Each input ``<name>.root`` is written to
``<output_dir>/<name>.h5`` so ``file_event_label_map`` patterns still match.
Particles are not shuffled; enable shuffling when training on the output.

Usage:
    python scripts/convert_root_to_h5.py data=bbllv08 ++output_dir=/path/to/h5

    # Optional overrides (data.max_events applies per input file):
    python scripts/convert_root_to_h5.py data=bbllv08 ++output_dir=h5 \\
        ++step_size=10000 ++overwrite=true ++skip_empty=false \\
        data.max_events=1000 data.preprocessing.max_particles=100 \\
        data.source.files='[a.root,b.root]'

Training on the output:
    python scripts/run.py data=preprocessed_h5 \\
        data.source.files='[/path/to/h5/*.h5]' \\
        data.preprocessing.max_particles=<same as conversion> \\
        data.preprocessing.shuffle_particles=true

Output datasets (per file):
    pt, eta, phi   (n_events, max_particles) float32, zero-padded, pt in GeV
    particle_id    (n_events, max_particles) int32, padded with default_particle_id
    n_particles    (n_events,) int32, particles after cuts, before truncation
    event_id       (n_events,) int64, entry index in the source tree
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
from particleman.data.conversion import convert_file, resolve_input_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = OmegaConf.select(cfg, "output_dir", default="")
    step_size = int(OmegaConf.select(cfg, "step_size", default=10000))
    overwrite = bool(OmegaConf.select(cfg, "overwrite", default=False))
    skip_empty = bool(OmegaConf.select(cfg, "skip_empty", default=False))

    if not output_dir:
        logger.error("output_dir must be specified, e.g. ++output_dir=path/to/h5")
        sys.exit(1)

    data_cfg = cfg.data
    if data_cfg.source.type != "root":
        logger.error(
            f"data.source.type is '{data_cfg.source.type}'; pick a ROOT data config"
        )
        sys.exit(1)

    inputs = resolve_input_files(data_cfg.source)
    if not inputs:
        logger.error("No input files found from data.source.files")
        sys.exit(1)

    output_dir = Path(output_dir)
    max_events = data_cfg.get("max_events", None)
    logger.info(f"Converting {len(inputs)} file(s) into {output_dir}")

    total = 0
    for input_path in inputs:
        output_path = output_dir / f"{input_path.stem}.h5".replace(".root", "").replace(".pool", "")
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping {output_path} (exists; use ++overwrite=true)")
            continue
        total += convert_file(
            input_path,
            output_path,
            data_cfg,
            step_size=step_size,
            max_events=max_events,
            skip_empty=skip_empty,
        )

    logger.info(f"Done: {total} events written")


if __name__ == "__main__":
    main()
