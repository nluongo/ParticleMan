"""Tests for counting input files and available events."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from particleman.data import EventCount, FileEventCount, count_events
from particleman.data.hdf5_loader import HDF5ParticleLoader


def _write_h5(path: Path, n_events: int, max_particles: int = 5) -> Path:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        shape = (n_events, max_particles)
        f.create_dataset("pt", data=rng.exponential(20, shape).astype(np.float32))
        f.create_dataset("eta", data=rng.normal(0, 1, shape).astype(np.float32))
        f.create_dataset("phi", data=rng.uniform(-3, 3, shape).astype(np.float32))
        f.create_dataset("particle_id", data=np.zeros(shape, dtype=np.int32))
    return path


def _config(files: list[str], label_map: dict[str, int] | None = None) -> DictConfig:
    cfg: dict[str, Any] = {
        "source": {"type": "hdf5", "files": files},
        "collections": {
            "particles": {
                "enabled": True,
                "columns": {
                    "pt": "pt",
                    "eta": "eta",
                    "phi": "phi",
                    "particle_id": "particle_id",
                },
                "is_vector": False,
            }
        },
        "file_event_label_map": label_map or {},
        "preprocessing": {"max_particles": 5},
    }
    return OmegaConf.create(cfg)


@pytest.fixture
def h5_dir(tmp_path: Path) -> Path:
    _write_h5(tmp_path / "ttbar_a.h5", 30)
    _write_h5(tmp_path / "ttbar_b.h5", 20)
    _write_h5(tmp_path / "zjets.h5", 7)
    return tmp_path


def test_total_matches_loader(h5_dir: Path) -> None:
    cfg = _config([str(h5_dir / "zjets.h5")])
    counts = count_events(cfg)
    assert counts.total == len(HDF5ParticleLoader(cfg)) == 7
    assert counts.n_files == 1


def test_per_file_with_glob_and_labels(h5_dir: Path) -> None:
    cfg = _config(
        [str(h5_dir / "ttbar_*.h5"), str(h5_dir / "zjets.h5")],
        label_map={"ttbar*": 1},
    )
    counts = count_events(cfg)
    assert counts.total == 57
    assert counts.n_files == 3
    assert [(f.path.name, f.n_events, f.label) for f in counts.per_file] == [
        ("ttbar_a.h5", 30, 1),
        ("ttbar_b.h5", 20, 1),
        ("zjets.h5", 7, -1),
    ]


def test_n_used() -> None:
    counts = EventCount(total=100, per_file=[FileEventCount(Path("x.h5"), 100)])
    assert counts.n_used() == 100
    assert counts.n_used(None) == 100
    assert counts.n_used(40) == 40
    assert counts.n_used(1000) == 100
