"""
Abstract base class for particle data loaders.

This module defines the interface that all particle data loaders must implement,
along with common preprocessing logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Tuple, Union

import numpy as np

# Support both OmegaConf DictConfig and plain dicts
try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = dict  # type: ignore

# Type alias for config - accepts Hydra DictConfig or plain dict
ConfigType = Union[DictConfig, Mapping[str, Any]]


def get_enabled_collections(config: ConfigType) -> Dict[str, Any]:
    """Get enabled collections from config.
    
    Args:
        config: Data configuration (DictConfig or dict).
        
    Returns:
        Dictionary of enabled collection names to their configs.
    """
    collections = config.get("collections", {})
    return {
        name: coll for name, coll in collections.items()
        if coll.get("enabled", True)
    }


class BaseParticleLoader(ABC):
    """
    Abstract base class for particle data loaders.

    Subclasses must implement:
        - _load_collection(): Load a single collection for one event
        - __len__(): Return total number of events
        - _open_files(): Open data files (called in __init__)

    The base class handles:
        - Combining multiple collections into a single event
        - Preprocessing (cuts, scaling, shuffling)
        - Padding/truncation to fixed length
        - Particle ID mapping
    """

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize the loader.

        Args:
            config: Data configuration (Hydra DictConfig or plain dict) specifying
                   sources, columns, and preprocessing.
        """
        self.config = config
        self.enabled_collections = get_enabled_collections(config)
        self._rng = np.random.default_rng(seed=config.get("split", {}).get("seed", 42))
        self._open_files()

    @abstractmethod
    def _open_files(self) -> None:
        """
        Open data files and prepare for reading.

        This method should set up any file handles, indices, or caches needed
        for efficient data access.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of events across all files."""
        pass

    @abstractmethod
    def _load_collection(
        self,
        idx: int,
        collection_name: str,
        collection_config: ConfigType,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single collection for one event.

        Args:
            idx: Event index.
            collection_name: Name of the collection (for debugging/logging).
            collection_config: Configuration for this collection.

        Returns:
            Tuple of (pt, eta, phi, particle_id) numpy arrays.
            Arrays may be empty if no particles in this collection for this event.
            The particle_id should already be mapped to categorical IDs.
        """
        pass

    def get_event(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single event by index, combining all enabled collections.

        Args:
            idx: Event index.

        Returns:
            Dict with keys:
                - 'pt': float32 array of shape (max_particles,)
                - 'eta': float32 array of shape (max_particles,)
                - 'phi': float32 array of shape (max_particles,)
                - 'particle_id': int32 array of shape (max_particles,)
                - 'mask': bool array of shape (max_particles,) - True for real particles
                - 'n_particles': int - number of real particles before padding
        """
        all_pt: List[np.ndarray] = []
        all_eta: List[np.ndarray] = []
        all_phi: List[np.ndarray] = []
        all_particle_id: List[np.ndarray] = []

        # Load each enabled collection
        for name, collection_cfg in self.enabled_collections.items():
            pt, eta, phi, pid = self._load_collection(idx, name, collection_cfg)

            if len(pt) > 0:
                all_pt.append(pt)
                all_eta.append(eta)
                all_phi.append(phi)
                all_particle_id.append(pid)

        # Combine all collections
        if all_pt:
            pt = np.concatenate(all_pt)
            eta = np.concatenate(all_eta)
            phi = np.concatenate(all_phi)
            particle_id = np.concatenate(all_particle_id)
        else:
            # Empty event
            pt = np.array([], dtype=np.float32)
            eta = np.array([], dtype=np.float32)
            phi = np.array([], dtype=np.float32)
            particle_id = np.array([], dtype=np.int32)

        # Apply preprocessing (cuts, scaling, shuffling)
        pt, eta, phi, particle_id = self._preprocess(pt, eta, phi, particle_id)

        # Track number of real particles before padding
        n_particles = len(pt)

        # Pad or truncate to max_particles
        pt, eta, phi, particle_id, mask = self._pad_or_truncate(pt, eta, phi, particle_id)

        return {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "particle_id": particle_id,
            "mask": mask,
            "n_particles": n_particles,
        }

    def _preprocess(
        self,
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        particle_id: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply preprocessing: scaling, cuts, and optional shuffling.

        Args:
            pt, eta, phi, particle_id: Raw particle arrays.

        Returns:
            Preprocessed arrays after cuts and scaling.
        """
        if len(pt) == 0:
            return pt, eta, phi, particle_id

        # Get preprocessing config with defaults
        preproc = self.config.get("preprocessing", {})
        pt_scale = preproc.get("pt_scale", 1.0)
        pt_cut = preproc.get("pt_cut", 0.5)
        eta_cut = preproc.get("eta_cut", 5.0)
        shuffle_particles = preproc.get("shuffle_particles", True)

        # Scale pT (e.g., MeV → GeV)
        pt = pt / pt_scale

        # Apply kinematic cuts
        mask = (pt >= pt_cut) & (np.abs(eta) <= eta_cut)
        pt = pt[mask]
        eta = eta[mask]
        phi = phi[mask]
        particle_id = particle_id[mask]

        # Shuffle particles within event if requested
        if shuffle_particles and len(pt) > 0:
            indices = self._rng.permutation(len(pt))
            pt = pt[indices]
            eta = eta[indices]
            phi = phi[indices]
            particle_id = particle_id[indices]

        return pt, eta, phi, particle_id

    def _map_particle_id(self, raw_id: int) -> int:
        """
        Map a single PDG ID to categorical ID.

        Args:
            raw_id: Raw PDG particle ID.

        Returns:
            Categorical particle ID for the model.
        """
        particle_id_map = self.config.get("particle_id_map", {})
        default_id = self.config.get("default_particle_id", 15)
        return particle_id_map.get(raw_id, default_id)

    def _map_particle_ids(self, raw_ids: np.ndarray) -> np.ndarray:
        """
        Map an array of PDG IDs to categorical IDs.

        Args:
            raw_ids: Array of raw PDG particle IDs.

        Returns:
            Array of categorical particle IDs for the model.
        """
        mapped = np.array(
            [self._map_particle_id(int(pid)) for pid in raw_ids],
            dtype=np.int32,
        )
        return mapped

    def _pad_or_truncate(
        self,
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        particle_id: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pad or truncate arrays to max_particles length.

        Args:
            pt, eta, phi, particle_id: Particle arrays.

        Returns:
            Tuple of (pt, eta, phi, particle_id, mask) arrays, all of length max_particles.
            mask is True for real particles, False for padding.
        """
        preproc = self.config.get("preprocessing", {})
        max_p = preproc.get("max_particles", 200)
        default_id = self.config.get("default_particle_id", 15)
        n = len(pt)

        # Create mask for real particles
        if n > max_p:
            # Truncate
            pt = pt[:max_p]
            eta = eta[:max_p]
            phi = phi[:max_p]
            particle_id = particle_id[:max_p]
            mask = np.ones(max_p, dtype=bool)
        elif n < max_p:
            # Pad
            pad_size = max_p - n
            pt = np.pad(pt, (0, pad_size), constant_values=0.0)
            eta = np.pad(eta, (0, pad_size), constant_values=0.0)
            phi = np.pad(phi, (0, pad_size), constant_values=0.0)
            particle_id = np.pad(
                particle_id,
                (0, pad_size),
                constant_values=default_id,
            )
            mask = np.zeros(max_p, dtype=bool)
            mask[:n] = True
        else:
            # Exact match
            mask = np.ones(max_p, dtype=bool)

        return (
            pt.astype(np.float32),
            eta.astype(np.float32),
            phi.astype(np.float32),
            particle_id.astype(np.int32),
            mask,
        )

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over all events."""
        for idx in range(len(self)):
            yield self.get_event(idx)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get event by index (supports negative indexing)."""
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Event index {idx} out of range [0, {len(self)})")
        return self.get_event(idx)