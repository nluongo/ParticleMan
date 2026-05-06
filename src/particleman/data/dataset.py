"""
PyTorch Dataset and DataLoader utilities for particle data.

This module provides PyTorch-compatible Dataset classes and factory functions
for creating DataLoaders from various file formats.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .base_loader import BaseParticleLoader, ConfigType
from .root_loader import ROOTParticleLoader
from .hdf5_loader import HDF5ParticleLoader

logger = logging.getLogger(__name__)


class ParticleDataset(Dataset):
    """
    PyTorch Dataset for particle physics data.

    This dataset wraps the particle data loaders and provides a PyTorch-compatible
    interface for training and evaluation.

    Example:
        >>> dataset = ParticleDataset(config)
        >>> sample = dataset[0]
        >>> print(sample['pt'].shape)  # torch.Size([max_particles])

        >>> # With DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # batch['pt'].shape: (32, max_particles)
        ...     pass
    """

    def __init__(
        self,
        config: ConfigType,
        transform: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            config: Data configuration (Hydra DictConfig or plain dict).
            transform: Optional transform function applied to each sample.
            indices: Optional list of indices to use (for train/val/test splits).
        """
        self.config = config
        self.transform = transform
        self.indices = indices

        # Create appropriate loader based on source type
        source = config.get("source", {})
        source_type = source.get("type", "root").lower()
        if source_type == "root":
            self.loader: BaseParticleLoader = ROOTParticleLoader(config)
        elif source_type == "hdf5":
            self.loader = HDF5ParticleLoader(config)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Set up index mapping
        if indices is not None:
            self._indices = indices
        else:
            self._indices = list(range(len(self.loader)))

        logger.info(f"Created ParticleDataset with {len(self)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with torch tensors:
                - 'pt': float32 tensor of shape (max_particles,)
                - 'eta': float32 tensor of shape (max_particles,)
                - 'phi': float32 tensor of shape (max_particles,)
                - 'particle_id': long tensor of shape (max_particles,)
                - 'mask': bool tensor of shape (max_particles,)
                - 'n_particles': int tensor (scalar)
        """
        # Map to actual index
        actual_idx = self._indices[idx]

        # Get event from loader
        event = self.loader.get_event(actual_idx)

        # Apply transform if provided
        if self.transform is not None:
            event = self.transform(event)

        # Convert to torch tensors
        return {
            "pt": torch.tensor(event["pt"], dtype=torch.float32),
            "eta": torch.tensor(event["eta"], dtype=torch.float32),
            "phi": torch.tensor(event["phi"], dtype=torch.float32),
            "particle_id": torch.tensor(event["particle_id"], dtype=torch.long),
            "mask": torch.tensor(event["mask"], dtype=torch.bool),
            "n_particles": torch.tensor(event["n_particles"], dtype=torch.long),
            "event_label": torch.tensor(event["event_label"], dtype=torch.long),
        }

    @property
    def max_particles(self) -> int:
        """Return the maximum number of particles per event."""
        preproc = self.config.get("preprocessing", {})
        return preproc.get("max_particles", 200)

    def get_raw_event(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a raw event (numpy arrays) without transformation.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with numpy arrays.
        """
        actual_idx = self._indices[idx]
        return self.loader.get_event(actual_idx)


def create_split_indices(
    n_samples: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test split indices.

    Args:
        n_samples: Total number of samples.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.
        test_frac: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    # Validate fractions
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    # Create shuffled indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples).tolist()

    # Calculate split points
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)

    # Split
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return train_indices, val_indices, test_indices


def create_dataloaders(
    config: ConfigType,
    max_events: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from a config.

    This is a convenience function that:
        1. Creates the dataset
        2. Splits into train/val/test
        3. Creates DataLoaders for each split

    Args:
        config: Data configuration (Hydra DictConfig or plain dict).
        max_events: Maximum number of events to use (None for all).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for CUDA transfers.
        transform: Optional transform function applied to each sample.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     config,
        ...     batch_size=64,
        ...     num_workers=4,
        ... )
        >>> for batch in train_loader:
        ...     # Training step
        ...     pass
    """
    # Create base dataset to get total size
    base_dataset = ParticleDataset(config, transform=transform)
    n_samples = len(base_dataset.loader)
    if max_events:
        n_samples = min(n_samples, max_events)

    # Get split config with defaults
    split = config.get("split", {})
    train_frac = split.get("train", 0.8)
    val_frac = split.get("val", 0.1)
    test_frac = split.get("test", 0.1)
    seed = split.get("seed", 42)

    # Create split indices
    train_indices, val_indices, test_indices = create_split_indices(
        n_samples,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    logger.info(
        f"Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
    )

    # Create datasets for each split
    train_dataset = ParticleDataset(config, transform=transform, indices=train_indices)
    val_dataset = ParticleDataset(config, transform=transform, indices=val_indices)
    test_dataset = ParticleDataset(config, transform=transform, indices=test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_datasets(
    config: ConfigType,
    max_events: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Tuple[ParticleDataset, ParticleDataset, ParticleDataset]:
    """
    Create train, validation, and test datasets from a config.

    This function handles splitting but does not create DataLoaders,
    making it suitable for distributed training where you need to
    wrap datasets with DistributedSampler.

    Args:
        config: Data configuration (Hydra DictConfig or plain dict).
        max_events: Maximum number of events to use (None for all).
        transform: Optional transform function applied to each sample.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Create base dataset to get total size
    base_dataset = ParticleDataset(config, transform=transform)
    n_samples = len(base_dataset.loader)
    if max_events:
        n_samples = min(n_samples, max_events)

    # Get split config with defaults
    split = config.get("split", {})
    train_frac = split.get("train", 0.8)
    val_frac = split.get("val", 0.1)
    test_frac = split.get("test", 0.1)
    seed = split.get("seed", 42)

    # Create split indices
    train_indices, val_indices, test_indices = create_split_indices(
        n_samples,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    logger.info(
        f"Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
    )

    # Create datasets for each split
    train_dataset = ParticleDataset(config, transform=transform, indices=train_indices)
    val_dataset = ParticleDataset(config, transform=transform, indices=val_indices)
    test_dataset = ParticleDataset(config, transform=transform, indices=test_indices)

    return train_dataset, val_dataset, test_dataset


def create_single_dataloader(
    config: ConfigType,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a single DataLoader for all data (no splitting).

    Useful for inference or when splits are handled externally.

    Args:
        config: Data configuration (Hydra DictConfig or plain dict).
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for CUDA transfers.
        transform: Optional transform function applied to each sample.

    Returns:
        DataLoader for all data.
    """
    dataset = ParticleDataset(config, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
