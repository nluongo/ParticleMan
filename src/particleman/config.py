"""Hydra structured configs for ParticleMan training.

This module defines dataclasses for Hydra configuration. The data source
configuration (collections, preprocessing, etc.) is loaded via Hydra's
config composition from configs/data/*.yaml files.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


# =============================================================================
# Data Source Configuration (mirrors particleman.data.config structure)
# =============================================================================

def resolve_glob(pattern: str) -> List[str]:
    # Expands patterns like "datasets/*.csv" or "images/img_*.png"
    return glob.glob(pattern)

# Register before the config store handles the schema
OmegaConf.register_new_resolver("glob", resolve_glob)

def expand_paths(inputs) -> list[str]:
    # Handle both a single string or a list of items
    if isinstance(inputs, str):
        inputs = [inputs]
        
    resolved_files = []
    for path_item in inputs:
        # Convert relative paths properly based on the project root
        abs_pattern = to_absolute_path(str(path_item))
        # Expand wildcards or leave literal paths as-is
        matched_files = glob.glob(abs_pattern)
        resolved_files.extend(matched_files)
        
    # Deduplicate and sort the final list
    return sorted(list(set(resolved_files)))

OmegaConf.register_new_resolver("resolve_paths", expand_paths)

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
    # Event label mapping: glob pattern on file basename -> integer label
    file_event_label_map: Dict[str, int] = field(default_factory=dict)
    n_event_types: int = 0
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
    d_ff: int = 256
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
    event_label_mask_prob: float = 0.15
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    profile: bool = False
    pretrained_checkpoint: Optional[str] = None
    mode: str = 'pretrain'  # 'pretrain' or 'classify'


# =============================================================================
# Output Configuration
# =============================================================================


@dataclass
class OutputConfig:
    """Output and logging configuration."""

    output_dir: str = "outputs"
    experiment_name: str = "ParticleMan_mc20_ttbar"
    timestamp: Optional[str] = None
    run_name: Optional[str] = None
    log_interval: int = 1
    experiment_logger: str = "mlflow"  # "mlflow", "comet", "noop", "none"


# =============================================================================
# Cluster Configuration
# =============================================================================

class Launcher(str, Enum):
    srun = "srun"
    torchrun = "torchrun"

    @classmethod
    def _missing_(cls, value):
        # Maps lowercase string inputs back to the uppercase enum keys
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value.lower() or member.value.lower() == value.lower():
                    return member
        return None

@dataclass
class ClusterConfig:
    """Cluster configuration."""

    scheduler: str = "slurm"
    account: str = "nluongo"
    # Slurm partition (resources.queue overrides this)
    queue: str = "gpu"
    qos: Optional[str] = None
    # torchrun or srun (one task per GPU)
    launcher: Launcher = Launcher.srun
    # Sourced at job start (conda activate, module load, ...); null to skip
    setup_script: Optional[str] = None
    master_port: int = 29500
    # Extra environment variables exported into the job
    env: Dict[str, Any] = field(default_factory=dict)
    # Raw extra sbatch arguments, e.g. ["--constraint=a100"]
    extra_args: List[str] = field(default_factory=list)


# =============================================================================
# Resource Configuration
# =============================================================================

@dataclass
class ResourceConfig:
    """Resource configuration."""

    nnodes: int = 1
    gpus_per_node: int = 1
    cpus_per_node: int = 8
    mem: str = "32gb"
    walltime: str = "4:00:00"
    omp_num_threads: int = 8
    # Queue/partition override (null = use cluster.queue)
    queue: Optional[str] = None

# =============================================================================
# Distributed Configuration
# =============================================================================


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    backend: str = "nccl"  # "nccl" or "gloo"
    timeout_s: int = 60  # Process group initialization timeout (seconds)


# =============================================================================
# Root Configuration
# =============================================================================


@dataclass
class TrainConfig:
    """Root configuration combining all sub-configs.
    
    The 'data' section is loaded from configs/data/*.yaml via Hydra's
    defaults system, ensuring consistent configuration loading.
    """

    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        {"data": "DataSourceConfig"},
        {"model": "ModelConfig"},
        {"training": "TrainingConfig"},
        {"output": "OutputConfig"},
        {"cluster": "ClusterConfig"},
        {"resource": "ResourceConfig"},
    ])

    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

def glob_directory(dir_path: str) -> list:
    path = pathlib.Path(dir_path)
    # This gathers all files in the directory. Alter the string to '*.csv' if needed.
    return [str(p) for p in path.glob('*') if p.is_file()]

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainConfig)
    cs.store(group="data", name="DataSourceConfig", node=DataSourceConfig)
    cs.store(group="model", name="ModelConfig", node=ModelConfig)
    cs.store(group="training", name="TrainingConfig", node=TrainingConfig)
    cs.store(group="output", name="OutputConfig", node=OutputConfig)
    cs.store(group="cluster", name="ClusterConfig", node=ClusterConfig)
    cs.store(group="resource", name="ResourceConfig", node=ResourceConfig)
    cs.store(group="distributed", name="DistributedConfig", node=DistributedConfig)
