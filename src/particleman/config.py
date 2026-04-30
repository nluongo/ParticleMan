"""Hydra structured configs for ParticleMan training.

This module defines dataclasses for Hydra configuration. The data source
configuration (collections, preprocessing, etc.) is loaded via Hydra's
config composition from configs/data/*.yaml files.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# Data Source Configuration (mirrors particleman.data.config structure)
# =============================================================================


@dataclass
class SourceConfig:
    """Data source configuration."""

    type: str = "root"  # "root" or "hdf5"
    files: List[str] = field(default_factory=list)
    tree_name: str = "CollectionTree"


@dataclass
class CollectionColumnsConfig:
    """Column names for a single particle collection."""

    pt: str = ""
    eta: str = ""
    phi: str = ""
    particle_id: Optional[str] = None


@dataclass
class CollectionConfig:
    """Configuration for a single particle collection."""

    enabled: bool = True
    columns: CollectionColumnsConfig = field(default_factory=CollectionColumnsConfig)
    fixed_particle_id: Optional[int] = None
    is_vector: bool = False


@dataclass
class PreprocessingConfig:
    """Preprocessing parameters."""

    pt_cut: float = 0.5
    eta_cut: float = 5.0
    max_particles: int = 200
    pt_scale: float = 1.0
    shuffle_particles: bool = True


@dataclass
class SplitConfig:
    """Data splitting configuration."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42


@dataclass
class DataSourceConfig:
    """Complete data source configuration.
    
    This mirrors the structure in configs/data/*.yaml and is loaded
    via Hydra's config composition.
    """

    source: SourceConfig = field(default_factory=SourceConfig)
    collections: Dict[str, Any] = field(default_factory=dict)
    particle_id_map: Dict[int, int] = field(default_factory=dict)
    default_particle_id: int = 15
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    # Training-specific override for max events (None = use all)
    max_events: Optional[int] = None


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    embedding: str = "joint"  # "joint" or "concat"
    phi_encoding: str = "raw"  # "raw", "sincos", or "none"
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.1
    angular_attention_bias: bool = False  # Use AttentionBiasTransformer
    bias_hidden_dim: int = 32  # Hidden dim for angular attention bias MLP


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.01
    mask_prob: float = 0.15
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    profile: bool = False


# =============================================================================
# Output Configuration
# =============================================================================


@dataclass
class OutputConfig:
    """Output and logging configuration."""

    output_dir: str = "outputs"
    experiment_name: str = "ParticleMan_mc20_ttbar"
    run_name: Optional[str] = None
    log_interval: int = 1
    experiment_logger: str = "mlflow"  # "mlflow", "comet", "noop", "none"


# =============================================================================
# Distributed Configuration
# =============================================================================


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    backend: str = "nccl"  # "nccl" or "gloo"


# =============================================================================
# Root Configuration
# =============================================================================


@dataclass
class TrainConfig:
    """Root configuration combining all sub-configs.
    
    The 'data' section is loaded from configs/data/*.yaml via Hydra's
    defaults system, ensuring consistent configuration loading.
    """

    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
