"""
Data loading module for ParticleMan.

This module provides utilities for loading particle physics data from
various file formats (ROOT, HDF5) with configurable column mappings.

Configuration is handled via Hydra - pass DictConfig or plain dict to
the loaders and dataset classes.
"""

from .dataset import (
    ParticleDataset,
    create_dataloaders,
    create_datasets,
    create_single_dataloader,
)
from .base_loader import BaseParticleLoader, ConfigType
from .root_loader import ROOTParticleLoader
from .hdf5_loader import HDF5ParticleLoader

__all__ = [
    # Dataset
    "ParticleDataset",
    "create_dataloaders",
    "create_datasets",
    "create_single_dataloader",
    # Loaders
    "BaseParticleLoader",
    "ROOTParticleLoader",
    "HDF5ParticleLoader",
    # Type alias
    "ConfigType",
]