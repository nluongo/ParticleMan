"""
ParticleMan: A framework for pre-training foundation models for particle physics tasks.

This package provides tools and utilities for building, training, and evaluating
foundation models specifically designed for particle physics applications.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Version information
VERSION = __version__

# Import main components (when dependencies are available)
try:
    from .models import ParticleTransformer, ParticleConfig
    from .training import ParticleTrainer
    from .data import (
        ParticleDataset,
        create_dataloaders,
        create_datasets,
    )
    from .config import (
        TrainConfig,
        DataSourceConfig,
        ModelConfig,
        TrainingConfig,
        OutputConfig,
        DistributedConfig,
    )
    
    __all__ = [
        "__version__",
        "VERSION",
        # Models
        "ParticleTransformer",
        "ParticleConfig", 
        # Training
        "ParticleTrainer",
        # Data
        "ParticleDataset",
        "create_dataloaders",
        "create_datasets",
        # Hydra configs
        "TrainConfig",
        "DataSourceConfig",
        "ModelConfig",
        "TrainingConfig",
        "OutputConfig",
        "DistributedConfig",
    ]
except ImportError:
    # Dependencies not available (e.g., during installation)
    __all__ = [
        "__version__",
        "VERSION",
    ] 