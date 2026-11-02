"""
Tests for the uproot-based ROOT -> HDF5 conversion.
"""

import copy
import pickle
from pathlib import Path
from typing import Any, Dict

import awkward as ak
import h5py
import numpy as np
import pytest
import uproot
from torch.utils.data import DataLoader

from particleman.data.conversion import (
    convert_file,
    map_particle_ids,
    resolve_input_files,
)
from particleman.data.dataset import ParticleDataset
from particleman.data.hdf5_loader import HDF5ParticleLoader
from particleman.data.root_loader import ROOTParticleLoader

N_EVENTS = 25
MAX_PARTICLES = 4


@pytest.fixture
def root_file(tmp_path: Path) -> Path:
    """Write a small ROOT ntuple with jagged and scalar branches."""
    rng = np.random.default_rng(0)
    counts_jet = rng.integers(0, 6, N_EVENTS)
    counts_el = rng.integers(0, 3, N_EVENTS)
    n_jet, n_el = int(counts_jet.sum()), int(counts_el.sum())

    def jagged(values: np.ndarray, counts: np.ndarray) -> Any:
        return ak.unflatten(values, counts)

    path = tmp_path / "ttbar_sample.root"
    with uproot.recreate(path) as f:
        f["AnalysisMiniTree"] = {
            "jet_pt": jagged(
                rng.uniform(0, 80000, n_jet).astype(np.float32), counts_jet
            ),
            "jet_eta": jagged(rng.uniform(-5, 5, n_jet).astype(np.float32), counts_jet),
            "jet_phi": jagged(rng.uniform(-3, 3, n_jet).astype(np.float32), counts_jet),
            "jet_label": jagged(
                rng.choice([0, 4, 5, 21, 99], n_jet).astype(np.int32), counts_jet
            ),
            "el_pt": jagged(rng.uniform(0, 50000, n_el).astype(np.float32), counts_el),
            "el_eta": jagged(rng.uniform(-3, 3, n_el).astype(np.float32), counts_el),
            "el_phi": jagged(rng.uniform(-3, 3, n_el).astype(np.float32), counts_el),
            "lep1_pt": rng.uniform(1000, 90000, N_EVENTS).astype(np.float32),
            "lep1_eta": rng.uniform(-2.5, 2.5, N_EVENTS).astype(np.float32),
            "lep1_phi": rng.uniform(-3, 3, N_EVENTS).astype(np.float32),
            "lep1_pdgid": rng.choice([-13, 11, 13], N_EVENTS).astype(np.int32),
        }
    return path


@pytest.fixture
def data_cfg(root_file: Path) -> Dict[str, Any]:
    """Data config mirroring configs/data/bbllv08.yaml for the test file."""
    return {
        "source": {
            "type": "root",
            "files": [str(root_file)],
            "tree_name": "AnalysisMiniTree",
        },
        "collections": {
            "jets": {
                "enabled": True,
                "columns": {
                    "pt": "jet_pt",
                    "eta": "jet_eta",
                    "phi": "jet_phi",
                    "particle_id": "jet_label",
                },
                "is_vector": True,
            },
            "electrons": {
                "enabled": True,
                "columns": {"pt": "el_pt", "eta": "el_eta", "phi": "el_phi"},
                "fixed_particle_id": 0,
                "is_vector": True,
            },
            "lepton1": {
                "enabled": True,
                "columns": {
                    "pt": "lep1_pt",
                    "eta": "lep1_eta",
                    "phi": "lep1_phi",
                    "particle_id": "lep1_pdgid",
                },
                "is_vector": False,
            },
            "disabled": {
                "enabled": False,
                "columns": {"pt": "nope", "eta": "nope", "phi": "nope"},
                "fixed_particle_id": 3,
            },
        },
        "particle_id_map": {0: 7, 4: 14, 5: 13, 21: 7, 11: 0, -13: 1, 13: 1},
        "default_particle_id": 15,
        "preprocessing": {
            "pt_cut": 5.0,
            "eta_cut": 4.5,
            "max_particles": MAX_PARTICLES,
            "pt_scale": 1000.0,
            "shuffle_particles": False,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
    }


def test_map_particle_ids() -> None:
    out = map_particle_ids(np.array([5, -13, 99, 0]), {0: 7, 5: 13, -13: 1}, 15)
    np.testing.assert_array_equal(out, [13, 1, 15, 7])
    assert out.dtype == np.int32


def test_resolve_input_files_glob(root_file: Path) -> None:
    files = resolve_input_files({"files": [str(root_file.parent / "*.root")]})
    assert files == [root_file]


def test_output_layout(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "out" / "ttbar_sample.h5"
    n = convert_file(root_file, out, data_cfg, step_size=7, show_progress=False)
    assert n == N_EVENTS
    with h5py.File(out, "r") as f:
        for name in ("pt", "eta", "phi", "particle_id"):
            assert f[name].shape == (N_EVENTS, MAX_PARTICLES)
        assert f["pt"].dtype == np.float32
        assert f["particle_id"].dtype == np.int32
        np.testing.assert_array_equal(f["event_id"][:], np.arange(N_EVENTS))
        assert f.attrs["n_events"] == N_EVENTS
        assert f.attrs["max_particles"] == MAX_PARTICLES
        assert list(f.attrs["collections"]) == ["jets", "electrons", "lepton1"]


def test_matches_root_loader(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    """Converted events equal ROOTParticleLoader.get_event (no shuffling)."""
    out = tmp_path / "ttbar_sample.h5"
    convert_file(root_file, out, data_cfg, step_size=7, show_progress=False)
    loader = ROOTParticleLoader(data_cfg)
    truncated = 0
    with h5py.File(out, "r") as f:
        for i in range(N_EVENTS):
            ev = loader.get_event(i)
            np.testing.assert_array_equal(f["pt"][i], ev["pt"])
            np.testing.assert_array_equal(f["eta"][i], ev["eta"])
            np.testing.assert_array_equal(f["phi"][i], ev["phi"])
            np.testing.assert_array_equal(f["particle_id"][i], ev["particle_id"])
            assert f["n_particles"][i] == ev["n_particles"]
            truncated += int(ev["n_particles"] > MAX_PARTICLES)
    assert truncated > 0, "fixture should exercise truncation"


def test_max_events_and_skip_empty(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "a.h5"
    assert (
        convert_file(root_file, out, data_cfg, max_events=10, show_progress=False) == 10
    )

    cfg = copy.deepcopy(data_cfg)
    cfg["preprocessing"]["pt_cut"] = 1e9  # everything cut away
    out = tmp_path / "b.h5"
    assert convert_file(root_file, out, cfg, skip_empty=True, show_progress=False) == 0


def test_missing_branch_raises(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    cfg = copy.deepcopy(data_cfg)
    cfg["collections"]["electrons"]["columns"]["pt"] = "does_not_exist"
    with pytest.raises(KeyError, match="does_not_exist"):
        convert_file(root_file, tmp_path / "x.h5", cfg, show_progress=False)


def _h5_cfg(path: Path) -> Dict[str, Any]:
    """preprocessed_h5-style config for a converted file."""
    return {
        "source": {"type": "hdf5", "files": [str(path)]},
        "collections": {
            "particles": {
                "enabled": True,
                "columns": {
                    "pt": "pt",
                    "eta": "eta",
                    "phi": "phi",
                    "particle_id": "particle_id",
                },
            }
        },
        "particle_id_map": {i: i for i in range(16)},
        "default_particle_id": 15,
        "preprocessing": {
            "pt_cut": 0.0,
            "eta_cut": 999.0,
            "max_particles": MAX_PARTICLES,
            "pt_scale": 1.0,
            "shuffle_particles": False,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
        "file_event_label_map": {"ttbar*": 0},
    }


def test_roundtrip_hdf5_loader(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    """Output loads with a preprocessed_h5-style config and gives the same events."""
    out = tmp_path / "ttbar_sample.h5"
    convert_file(root_file, out, data_cfg, show_progress=False)
    h5_loader = HDF5ParticleLoader(_h5_cfg(out))
    root_loader = ROOTParticleLoader(data_cfg)
    assert len(h5_loader) == N_EVENTS
    for i in range(N_EVENTS):
        a, b = h5_loader.get_event(i), root_loader.get_event(i)
        for key in ("pt", "eta", "phi", "particle_id", "mask"):
            np.testing.assert_array_equal(a[key], b[key])
        assert a["event_label"] == 0


def test_hdf5_loader_pickles(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    """The loader pickles (no open h5py handles) and reopens files on demand."""
    out = tmp_path / "ttbar_sample.h5"
    convert_file(root_file, out, data_cfg, show_progress=False)
    loader = HDF5ParticleLoader(_h5_cfg(out))
    loader.get_event(0)  # opens a handle in this process
    copy_ = pickle.loads(pickle.dumps(loader))
    for i in range(N_EVENTS):
        np.testing.assert_array_equal(
            copy_.get_event(i)["pt"], loader.get_event(i)["pt"]
        )
    loader.close()
    assert loader.get_event(1)["n_particles"] == copy_.get_event(1)["n_particles"]


def test_hdf5_dataloader_spawn_workers(
    root_file: Path, data_cfg: Dict[str, Any], tmp_path: Path
) -> None:
    """DataLoader workers started with spawn (macOS default) can read HDF5."""
    out = tmp_path / "ttbar_sample.h5"
    convert_file(root_file, out, data_cfg, show_progress=False)
    dataset = ParticleDataset(_h5_cfg(out))
    loader = DataLoader(
        dataset, batch_size=8, num_workers=2, multiprocessing_context="spawn"
    )
    batches = list(loader)
    assert sum(len(b["pt"]) for b in batches) == N_EVENTS
