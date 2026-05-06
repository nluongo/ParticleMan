"""
HDF5 file loader for particle data.

This module provides a loader for reading particle data from HDF5 files,
which is a common format for preprocessed physics data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from .base_loader import BaseParticleLoader, ConfigType

logger = logging.getLogger(__name__)


class HDF5ParticleLoader(BaseParticleLoader):
    """
    Load particle data from HDF5 files.

    This loader supports:
        - Multiple HDF5 files
        - Multiple particle collections per event
        - Both fixed-size and variable-length datasets
        - Configurable column mappings

    HDF5 files are expected to have datasets that are either:
        - 2D arrays of shape (n_events, n_particles) for fixed-size data
        - 1D arrays with a separate index dataset for variable-length data

    Example:
        >>> loader = HDF5ParticleLoader(config)
        >>> event = loader.get_event(0)
        >>> print(event['pt'].shape)  # (max_particles,)
    """
    def __init__(self, config: ConfigType) -> None:
        """
        Initialize the HDF5 loader.

            config: Data configuration.
        """
        super().__init__(config)

    def _open_files(self) -> None:
        """
        Open HDF5 files and build event index.

        This method:
            1. Opens each HDF5 file
            2. Determines the number of events per file
            3. Builds an index mapping global event index to (file_idx, local_idx)
        """
        self._files: List[h5py.File] = []
        self._event_offsets: List[int] = [0]
        self._total_events = 0
        self._file_paths: List[Path] = []  # Keep track of paths for reopening

        source = self.config.get("source", {})
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
                    self._add_file(match)
            else:
                if not path.exists():
                    raise FileNotFoundError(f"HDF5 file not found: {filepath}")
                self._add_file(path)

        if self._total_events == 0:
            raise ValueError("No events found in any of the specified files")

        logger.info(
            f"Loaded {len(self._files)} file(s) with {self._total_events} total events"
        )

    def _add_file(self, filepath: Path) -> None:
        """
        Add a single HDF5 file to the loader.

        Args:
            filepath: Path to HDF5 file.
        """
        try:
            f = h5py.File(filepath, "r")

            # Determine number of events
            # Try to find a dataset to get the event count
            n_events = self._get_num_events(f)

            if n_events == 0:
                logger.warning(f"No events found in {filepath}")
                f.close()
                return

            self._files.append(f)
            self._file_paths.append(filepath)
            self._event_offsets.append(self._event_offsets[-1] + n_events)
            self._total_events += n_events

            logger.debug(f"Added {filepath} with {n_events} events")

        except Exception as e:
            logger.error(f"Failed to open {filepath}: {e}")
            raise

    def _get_num_events(self, f: h5py.File) -> int:
        """
        Determine the number of events in an HDF5 file.

        Tries multiple strategies:
            1. Check for 'n_events' attribute
            2. Look at first dimension of first collection's pt dataset
            3. Check common dataset names like 'pt', 'event_id'

        Args:
            f: Open HDF5 file.

        Returns:
            Number of events in the file.
        """
        # Strategy 1: Check file attributes
        if "n_events" in f.attrs:
            return int(f.attrs["n_events"])

        # Strategy 2: Check first enabled collection
        for name, coll_cfg in self.enabled_collections.items():
            pt_col = coll_cfg.columns.pt
            if pt_col in f:
                dataset = f[pt_col]
                if len(dataset.shape) >= 1:
                    return dataset.shape[0]

        # Strategy 3: Check common dataset names
        for common_name in ["pt", "event_id", "events", "n_particles"]:
            if common_name in f:
                dataset = f[common_name]
                if len(dataset.shape) >= 1:
                    return dataset.shape[0]

        # Strategy 4: Check any dataset
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                if len(f[key].shape) >= 1:
                    return f[key].shape[0]

        return 0

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
        f = self._files[file_idx]
        
        # Access columns config
        columns = collection_config.get("columns", {})
        pt_col = columns.get("pt")
        eta_col = columns.get("eta")
        phi_col = columns.get("phi")
        pid_col = columns.get("particle_id")
        is_vector = collection_config.get("is_vector", False)
        fixed_particle_id = collection_config.get("fixed_particle_id")

        try:
            # Read the datasets for this event
            pt = self._read_dataset(f, pt_col, local_idx, is_vector)
            eta = self._read_dataset(f, eta_col, local_idx, is_vector)
            phi = self._read_dataset(f, phi_col, local_idx, is_vector)

            # Handle particle ID
            if pid_col is not None:
                raw_pid = self._read_dataset(f, pid_col, local_idx, is_vector)
                particle_id = self._map_particle_ids(raw_pid)
            else:
                # Use fixed particle ID for all particles in this collection
                particle_id = np.full(
                    len(pt),
                    fixed_particle_id,
                    dtype=np.int32,
                )

            # Filter out padding (zero pt values) if present
            valid_mask = pt > 0
            if not np.all(valid_mask):
                pt = pt[valid_mask]
                eta = eta[valid_mask]
                phi = phi[valid_mask]
                particle_id = particle_id[valid_mask]

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

    def _read_dataset(
        self,
        f: h5py.File,
        dataset_name: str,
        event_idx: int,
        is_vector: bool,
    ) -> np.ndarray:
        """
        Read a dataset for a single event.

        Handles both 2D fixed-size arrays and variable-length data.

        Args:
            f: Open HDF5 file.
            dataset_name: Name of the dataset.
            event_idx: Local event index within the file.
            is_vector: Whether this is variable-length data.

        Returns:
            1D numpy array of values for this event.
        """
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in HDF5 file")

        dataset = f[dataset_name]

        # Handle different dataset shapes
        if len(dataset.shape) == 1:
            # 1D dataset - could be one value per event (scalar)
            # or flattened variable-length with separate index
            return np.array([dataset[event_idx]])

        elif len(dataset.shape) == 2:
            # 2D dataset - (n_events, n_particles)
            return np.array(dataset[event_idx])

        else:
            # Higher dimensional - just take the slice
            return np.array(dataset[event_idx]).flatten()

    def get_dataset_names(self) -> List[str]:
        """
        Get list of all dataset names in the first file.

        Useful for debugging and discovering available columns.

        Returns:
            List of dataset names.
        """
        if not self._files:
            return []

        def collect_names(group: h5py.Group, prefix: str = "") -> List[str]:
            """Recursively collect dataset names."""
            names = []
            for key in group.keys():
                full_name = f"{prefix}/{key}" if prefix else key
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    names.append(full_name)
                elif isinstance(item, h5py.Group):
                    names.extend(collect_names(item, full_name))
            return names

        return collect_names(self._files[0])

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Dictionary with dataset information.
        """
        if not self._files:
            return {}

        f = self._files[0]
        if dataset_name not in f:
            return {"error": f"Dataset '{dataset_name}' not found"}

        dataset = f[dataset_name]
        info = {
            "name": dataset_name,
            "shape": dataset.shape,
            "dtype": str(dataset.dtype),
            "size": dataset.size,
        }

        # Add any attributes
        if dataset.attrs:
            info["attrs"] = dict(dataset.attrs)

        return info

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

    def close(self) -> None:
        """Close all open HDF5 files."""
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = []

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
