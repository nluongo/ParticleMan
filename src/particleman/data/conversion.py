"""
Convert ROOT ntuples to padded HDF5 files using uproot.

The conversion is driven by the same data config that ``ROOTParticleLoader``
uses (``source``, ``collections``, ``particle_id_map``, ``preprocessing``), so
the output matches what the loader would produce event by event (without
particle shuffling). The resulting files can be read with ``HDF5ParticleLoader``
via ``configs/data/preprocessed_h5.yaml``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import awkward as ak
    import uproot

    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

from .base_loader import ConfigType, get_enabled_collections

logger = logging.getLogger(__name__)

COMMON_TREE_NAMES = ["CollectionTree", "Events", "tree", "ntuple"]
FEATURES = ("pt", "eta", "phi")


def resolve_input_files(source_cfg: ConfigType) -> List[Path]:
    """
    Expand the ``source.files`` list (with glob patterns) into file paths.

    Args:
        source_cfg: The ``data.source`` config section.

    Returns:
        Sorted list of input file paths, in config order.

    Raises:
        FileNotFoundError: If a non-glob path does not exist.
    """
    paths: List[Path] = []
    for filepath in source_cfg.get("files", []):
        path = Path(filepath)
        if "*" in str(path):
            matches = sorted(path.parent.glob(path.name))
            if not matches:
                logger.warning(f"No files matched pattern: {filepath}")
            paths.extend(matches)
        else:
            if not path.exists():
                raise FileNotFoundError(f"ROOT file not found: {filepath}")
            paths.append(path)
    return paths


def _open_tree(f: Any, tree_name: str, filepath: Path) -> Tuple[Any, str]:
    """Return ``(tree, name)``, falling back to common tree names."""
    if tree_name in f:
        return f[tree_name], tree_name
    for name in COMMON_TREE_NAMES:
        if name in f:
            logger.info(f"Using tree '{name}' instead of '{tree_name}'")
            return f[name], name
    available = [k.split(";")[0] for k in f.keys()]
    raise ValueError(
        f"Tree '{tree_name}' not found in {filepath}. "
        f"Available keys: {available[:10]}"
    )


def _as_jagged(arr: Any) -> Any:
    """Turn a branch array into a 2-level jagged array (one list per event)."""
    if arr.ndim == 1:
        # Scalar branch: one value per event
        return ak.unflatten(arr, 1)
    while arr.ndim > 2:
        # Nested vectors: merge inner lists, as the loader's flatten does
        arr = ak.flatten(arr, axis=2)
    return arr


def map_particle_ids(
    raw_ids: np.ndarray, particle_id_map: Dict[int, int], default_id: int
) -> np.ndarray:
    """
    Vectorized version of ``BaseParticleLoader._map_particle_ids``.

    Args:
        raw_ids: Raw IDs (e.g. PDG codes or jet truth labels).
        particle_id_map: Mapping of raw ID to categorical ID.
        default_id: ID for raw values not in the map.

    Returns:
        int32 array of categorical IDs.
    """
    raw = raw_ids.astype(np.int64)
    if not particle_id_map:
        return np.full(raw.shape, default_id, dtype=np.int32)
    keys = np.array(sorted(particle_id_map), dtype=np.int64)
    values = np.array([particle_id_map[k] for k in keys], dtype=np.int32)
    pos = np.clip(np.searchsorted(keys, raw), 0, len(keys) - 1)
    found = keys[pos] == raw
    return np.where(found, values[pos], default_id).astype(np.int32)


def _collection_branches(coll_cfg: ConfigType) -> List[str]:
    """Return the branch names a collection reads."""
    columns = coll_cfg.get("columns", {})
    branches = [columns.get(feat) for feat in FEATURES]
    if columns.get("particle_id") is not None:
        branches.append(columns.get("particle_id"))
    return [str(b) for b in branches]


def process_chunk(chunk: Any, data_cfg: ConfigType) -> Dict[str, np.ndarray]:
    """
    Combine, map, cut and pad one chunk of events.

    Args:
        chunk: Awkward record array of branches from ``tree.iterate``.
        data_cfg: Data config.

    Returns:
        Dict with padded ``pt``, ``eta``, ``phi`` (float32), ``particle_id``
        (int32), each of shape ``(n_events, max_particles)``, and
        ``n_particles`` (int32, count after cuts and before truncation).
    """
    preproc = data_cfg.get("preprocessing", {})
    pt_scale = float(preproc.get("pt_scale", 1.0))
    pt_cut = float(preproc.get("pt_cut", 0.5))
    eta_cut = float(preproc.get("eta_cut", 5.0))
    max_p = int(preproc.get("max_particles", 200))
    default_id = int(data_cfg.get("default_particle_id", 15))
    id_map = {int(k): int(v) for k, v in data_cfg.get("particle_id_map", {}).items()}

    parts: Dict[str, List[Any]] = {"pt": [], "eta": [], "phi": [], "particle_id": []}
    for coll_cfg in get_enabled_collections(data_cfg).values():
        columns = coll_cfg.get("columns", {})
        feats = {
            feat: ak.values_astype(_as_jagged(chunk[columns.get(feat)]), np.float32)
            for feat in FEATURES
        }
        pid_col = columns.get("particle_id")
        if pid_col is not None:
            raw = _as_jagged(chunk[pid_col])
            mapped = map_particle_ids(ak.to_numpy(ak.flatten(raw)), id_map, default_id)
            pid = ak.unflatten(mapped, ak.num(raw))
        else:
            fixed = coll_cfg.get("fixed_particle_id")
            if fixed is None:
                raise ValueError(
                    f"Collection needs columns.particle_id or fixed_particle_id: "
                    f"{dict(columns)}"
                )
            pid = ak.values_astype(ak.zeros_like(feats["pt"]) + fixed, np.int32)
        for feat in FEATURES:
            parts[feat].append(feats[feat])
        parts["particle_id"].append(pid)

    n_events = len(chunk)
    if not parts["pt"]:
        raise ValueError("No enabled collections in data config")

    combined = {k: ak.concatenate(v, axis=1) for k, v in parts.items()}
    combined["pt"] = combined["pt"] / np.float32(pt_scale)
    keep = (combined["pt"] >= pt_cut) & (np.abs(combined["eta"]) <= eta_cut)
    combined = {k: v[keep] for k, v in combined.items()}

    out: Dict[str, np.ndarray] = {
        "n_particles": ak.to_numpy(ak.num(combined["pt"])).astype(np.int32)
    }
    for key, arr in combined.items():
        fill = default_id if key == "particle_id" else 0.0
        dtype = np.int32 if key == "particle_id" else np.float32
        padded = ak.fill_none(ak.pad_none(arr, max_p, axis=1, clip=True), fill)
        out[key] = ak.to_numpy(padded).astype(dtype).reshape(n_events, max_p)
    return out


def _create_datasets(h5f: h5py.File, max_p: int) -> None:
    """Create empty resizable datasets in an output file."""
    chunk_rows = 1024
    specs = {
        "pt": (np.float32, (max_p,)),
        "eta": (np.float32, (max_p,)),
        "phi": (np.float32, (max_p,)),
        "particle_id": (np.int32, (max_p,)),
        "n_particles": (np.int32, ()),
        "event_id": (np.int64, ()),
    }
    for name, (dtype, tail) in specs.items():
        h5f.create_dataset(
            name,
            shape=(0, *tail),
            maxshape=(None, *tail),
            dtype=dtype,
            chunks=(chunk_rows, *tail),
            compression="gzip",
        )
    h5f["pt"].attrs["description"] = "Transverse momentum in GeV (0 = padding)"
    h5f["eta"].attrs["description"] = "Pseudorapidity"
    h5f["phi"].attrs["description"] = "Azimuthal angle in radians"
    h5f["particle_id"].attrs["description"] = "Categorical particle type ID"
    h5f["n_particles"].attrs["description"] = "Real particles before truncation"
    h5f["event_id"].attrs["description"] = "Entry index in the source ROOT tree"


def _append(h5f: h5py.File, name: str, data: np.ndarray) -> None:
    """Append rows to a resizable dataset."""
    dset = h5f[name]
    start = dset.shape[0]
    dset.resize(start + len(data), axis=0)
    dset[start:] = data


def convert_file(
    input_path: Path,
    output_path: Path,
    data_cfg: ConfigType,
    step_size: int = 10000,
    max_events: Optional[int] = None,
    skip_empty: bool = False,
    show_progress: bool = True,
) -> int:
    """
    Convert one ROOT file to a padded HDF5 file.

    Args:
        input_path: Input ROOT file.
        output_path: Output HDF5 file (overwritten if it exists).
        data_cfg: Data config describing tree, collections and preprocessing.
        step_size: Number of entries read per uproot chunk.
        max_events: Maximum number of entries to read (None for all).
        skip_empty: Drop events with no particles after cuts.
        show_progress: Show a tqdm progress bar.

    Returns:
        Number of events written.
    """
    if not HAS_UPROOT:
        raise ImportError(
            "uproot is required for ROOT conversion. "
            "Install with: pip install uproot awkward"
        )

    source = data_cfg.get("source", {})
    preproc = data_cfg.get("preprocessing", {})
    max_p = int(preproc.get("max_particles", 200))
    collections = get_enabled_collections(data_cfg)
    branches = sorted(
        {b for c in collections.values() for b in _collection_branches(c)}
    )

    with uproot.open(input_path) as f:
        tree, tree_name = _open_tree(
            f, str(source.get("tree_name", "CollectionTree")), input_path
        )
        missing = [b for b in branches if b not in tree]
        if missing:
            raise KeyError(f"Branches not found in {input_path}: {missing}")

        n_entries = int(tree.num_entries)
        if max_events is not None:
            n_entries = min(n_entries, int(max_events))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        n_written = 0
        with h5py.File(output_path, "w") as h5f:
            _create_datasets(h5f, max_p)
            pbar = None
            if show_progress:
                from tqdm import tqdm

                pbar = tqdm(total=n_entries, desc=input_path.name, unit="evt")

            for chunk, report in tree.iterate(
                branches,
                step_size=step_size,
                entry_stop=n_entries,
                library="ak",
                report=True,
            ):
                out = process_chunk(chunk, data_cfg)
                out["event_id"] = np.arange(report.start, report.stop, dtype=np.int64)
                if skip_empty:
                    nonempty = out["n_particles"] > 0
                    out = {k: v[nonempty] for k, v in out.items()}
                for name, data in out.items():
                    _append(h5f, name, data)
                n_written += len(out["event_id"])
                if pbar is not None:
                    pbar.update(report.stop - report.start)

            if pbar is not None:
                pbar.close()

            h5f.attrs["n_events"] = n_written
            h5f.attrs["source_file"] = str(input_path)
            h5f.attrs["tree_name"] = tree_name
            h5f.attrs["max_particles"] = max_p
            h5f.attrs["pt_scale_applied"] = float(preproc.get("pt_scale", 1.0))
            h5f.attrs["pt_cut"] = float(preproc.get("pt_cut", 0.5))
            h5f.attrs["eta_cut"] = float(preproc.get("eta_cut", 5.0))
            h5f.attrs["particle_id_map"] = str(
                dict(data_cfg.get("particle_id_map", {}))
            )
            h5f.attrs["default_particle_id"] = int(
                data_cfg.get("default_particle_id", 15)
            )
            h5f.attrs["collections"] = list(collections)

    logger.info(f"Wrote {n_written} events to {output_path}")
    return n_written
