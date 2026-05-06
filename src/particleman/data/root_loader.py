"""
ROOT file loader for particle data.

This module provides a loader for reading particle data from ROOT flat ntuples
using uproot (pure Python, no ROOT installation required).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import uproot
    import awkward as ak

    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

from .base_loader import BaseParticleLoader, ConfigType

logger = logging.getLogger(__name__)


class ROOTParticleLoader(BaseParticleLoader):
    """
    Load particle data from ROOT flat ntuples using uproot.

    This loader supports:
        - Multiple ROOT files
        - Multiple particle collections per event
        - Both scalar and vector (jagged) branches
        - Configurable column mappings

    Example:
        >>> loader = ROOTParticleLoader(config)
        >>> event = loader.get_event(0)
        >>> print(event['pt'].shape)  # (max_particles,)
    """

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize the ROOT loader.

        Args:
            config: Data configuration (Hydra DictConfig or plain dict).

        Raises:
            ImportError: If uproot is not installed.
        """
        if not HAS_UPROOT:
            raise ImportError(
                "uproot is required for ROOT file loading. "
                "Install with: pip install uproot awkward"
            )
        super().__init__(config)

    def _open_files(self) -> None:
        """
        Open ROOT files and build event index.

        This method:
            1. Opens each ROOT file
            2. Accesses the specified tree
            3. Builds an index mapping global event index to (file_idx, local_idx)
        """
        self._files: List[uproot.ReadOnlyDirectory] = []
        self._trees: List[uproot.TTree] = []
        self._event_offsets: List[int] = [0]
        self._total_events = 0
        self._file_paths: List[Path] = []

        source = self.config.get("source", {})
        tree_name = source.get("tree_name", "CollectionTree")
        files = source.get("files", [])

        for filepath in files:
            path = Path(filepath)

            # Handle glob patterns
            if "*" in str(path):
                matching_files = sorted(path.parent.glob(path.name))
                if not matching_files:
                    logger.warning(f"No files matched pattern: {filepath}")
                    continue
                for match in matching_files:
                    self._add_file(match, tree_name)
            else:
                if not path.exists():
                    raise FileNotFoundError(f"ROOT file not found: {filepath}")
                self._add_file(path, tree_name)

        if self._total_events == 0:
            raise ValueError("No events found in any of the specified files")

        logger.info(
            f"Loaded {len(self._files)} file(s) with {self._total_events} total events"
        )

    def _add_file(self, filepath: Path, tree_name: str) -> None:
        """
        Add a single ROOT file to the loader.

        Args:
            filepath: Path to ROOT file.
            tree_name: Name of the TTree to read.
        """
        try:
            f = uproot.open(filepath)

            # Try to find the tree
            if tree_name in f:
                tree = f[tree_name]
            else:
                # Try common tree names if specified one not found
                common_names = ["CollectionTree", "Events", "tree", "ntuple"]
                tree = None
                for name in common_names:
                    if name in f:
                        tree = f[name]
                        logger.info(f"Using tree '{name}' instead of '{tree_name}'")
                        break

                if tree is None:
                    available = [k for k in f.keys() if not k.endswith(";1")]
                    raise ValueError(
                        f"Tree '{tree_name}' not found in {filepath}. "
                        f"Available keys: {available[:10]}"
                    )

            n_events = tree.num_entries
            self._files.append(f)
            self._trees.append(tree)
            self._file_paths.append(filepath)
            self._event_offsets.append(self._event_offsets[-1] + n_events)
            self._total_events += n_events

            logger.debug(f"Added {filepath} with {n_events} events")

        except Exception as e:
            logger.error(f"Failed to open {filepath}: {e}")
            raise

    def __len__(self) -> int:
        """Return total number of events."""
        return self._total_events

    def _get_file_idx(self, idx: int) -> int:
        """Return the file index for global event index idx."""
        file_idx, _ = self._get_file_and_local_idx(idx)
        return file_idx

    def _get_file_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global event index to (file_index, local_index).

        Args:
            global_idx: Global event index across all files.

        Returns:
            Tuple of (file_index, local_event_index).
        """
        for file_idx in range(len(self._files)):
            if global_idx < self._event_offsets[file_idx + 1]:
                local_idx = global_idx - self._event_offsets[file_idx]
                return file_idx, local_idx
        raise IndexError(f"Event index {global_idx} out of range")

    def _load_collection(
        self,
        idx: int,
        collection_name: str,
        collection_config: ConfigType,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single collection for one event.

        Args:
            idx: Global event index.
            collection_name: Name of the collection (for logging).
            collection_config: Configuration for this collection (dict or DictConfig).

        Returns:
            Tuple of (pt, eta, phi, particle_id) numpy arrays.
        """
        file_idx, local_idx = self._get_file_and_local_idx(idx)
        tree = self._trees[file_idx]
        
        # Access columns config
        columns = collection_config.get("columns", {})
        pt_col = columns.get("pt")
        eta_col = columns.get("eta")
        phi_col = columns.get("phi")
        pid_col = columns.get("particle_id")
        is_vector = collection_config.get("is_vector", False)
        fixed_particle_id = collection_config.get("fixed_particle_id")

        try:
            # Determine which branches to read
            branches_to_read = [pt_col, eta_col, phi_col]
            if pid_col is not None:
                branches_to_read.append(pid_col)

            # Read the branches for this event
            # Using library="ak" for awkward arrays which handle jagged data well
            data = tree.arrays(
                branches_to_read,
                entry_start=local_idx,
                entry_stop=local_idx + 1,
                library="ak",
            )

            # Extract arrays, handling both scalar and vector branches
            pt = self._extract_array(data[pt_col], is_vector)
            eta = self._extract_array(data[eta_col], is_vector)
            phi = self._extract_array(data[phi_col], is_vector)

            # Handle particle ID
            if pid_col is not None:
                raw_pid = self._extract_array(data[pid_col], is_vector)
                particle_id = self._map_particle_ids(raw_pid)
            else:
                # Use fixed particle ID for all particles in this collection
                particle_id = np.full(
                    len(pt),
                    fixed_particle_id,
                    dtype=np.int32,
                )

            return (
                pt.astype(np.float32),
                eta.astype(np.float32),
                phi.astype(np.float32),
                particle_id.astype(np.int32),
            )

        except Exception as e:
            logger.warning(
                f"Failed to load collection '{collection_name}' for event {idx}: {e}"
            )
            # Return empty arrays on failure
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
            )

    def _extract_array(
        self, arr: Any, is_vector: bool
    ) -> np.ndarray:
        """
        Extract a numpy array from uproot/awkward array.

        Handles both scalar branches (single value per event) and
        vector branches (variable-length arrays per event).

        Args:
            arr: Awkward array from uproot.
            is_vector: Whether this is a vector branch.

        Returns:
            1D numpy array of values.
        """
        # Convert awkward array to numpy
        if hasattr(arr, "to_numpy"):
            # Simple case: already flat
            return arr.to_numpy().flatten()
        elif hasattr(arr, "tolist"):
            # Awkward array - need to flatten
            flat = ak.flatten(arr, axis=None)
            return ak.to_numpy(flat)
        else:
            # Try direct numpy conversion
            return np.asarray(arr).flatten()

    def get_branch_names(self) -> List[str]:
        """
        Get list of all branch names in the first tree.

        Useful for debugging and discovering available columns.

        Returns:
            List of branch names.
        """
        if not self._trees:
            return []
        return list(self._trees[0].keys())

    def get_branch_info(self, branch_name: str) -> Dict[str, Any]:
        """
        Get information about a specific branch.

        Args:
            branch_name: Name of the branch.

        Returns:
            Dictionary with branch information.
        """
        if not self._trees:
            return {}

        tree = self._trees[0]
        if branch_name not in tree:
            return {"error": f"Branch '{branch_name}' not found"}

        branch = tree[branch_name]
        return {
            "name": branch_name,
            "typename": str(branch.typename),
            "interpretation": str(branch.interpretation),
            "num_entries": branch.num_entries,
        }

    def preview_event(
        self, idx: int = 0, max_particles: int = 10
    ) -> Dict[str, Any]:
        """
        Preview an event for debugging purposes.

        Args:
            idx: Event index.
            max_particles: Maximum number of particles to show.

        Returns:
            Dictionary with event data preview.
        """
        event = self.get_event(idx)
        n_real = event["n_particles"]

        return {
            "event_index": idx,
            "n_particles": n_real,
            "pt": event["pt"][:max_particles].tolist(),
            "eta": event["eta"][:max_particles].tolist(),
            "phi": event["phi"][:max_particles].tolist(),
            "particle_id": event["particle_id"][:max_particles].tolist(),
            "mask": event["mask"][:max_particles].tolist(),
        }
