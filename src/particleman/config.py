"""Hydra structured config schemas for ParticleMan.

The dataclasses here are schemas: they define which keys exist and their
types. The values come from the YAML files in configs/. ``register_configs``
stores each group schema as ``base_<group>`` (and the root as
``base_config``). Each YAML file puts its schema first in its defaults list,
so Hydra merges the YAML onto the dataclass and rejects unknown keys or
wrong types. The composition order is: schema, group YAML, root YAML, CLI.

When you add a config key, add it to both the YAML and the dataclass.
"""

import glob
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


# =============================================================================
# Data Source Configuration (mirrors particleman.data.config structure)
# =============================================================================

def resolve_glob(pattern: str) -> List[str]:
    """Expand a glob pattern such as ``datasets/*.h5`` (``${glob:...}``)."""
    return glob.glob(pattern)


def expand_paths(inputs: Union[str, Sequence[str]]) -> List[str]:
    """Expand paths and glob patterns into a sorted, de-duplicated file list.

    Relative paths are taken from the directory Hydra was started in
    (``${resolve_paths:...}``).
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    resolved_files: List[str] = []
    for path_item in inputs:
        resolved_files.extend(glob.glob(to_absolute_path(str(path_item))))
    return sorted(set(resolved_files))


# Register before the config store handles the schema. replace=True allows
# this module to be imported more than once (e.g. by tests).
OmegaConf.register_new_resolver("glob", resolve_glob, replace=True)
OmegaConf.register_new_resolver("resolve_paths", expand_paths, replace=True)

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
    # Keep DataLoader workers alive between epochs (ignored when num_workers=0)
    persistent_workers: bool = False
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
    mlflow_experiment_prefix: Optional[str] = None  # e.g. "/Users/<email>"


# =============================================================================
# Cluster Configuration
# =============================================================================

class Launcher(str, Enum):
    """How launch.sh starts the training processes.

    The scheduler/launcher pairing is checked by
    ``particleman.scheduling.check_launcher``.
    """

    torchrun = "torchrun"
    srun = "srun"
    mpiexec = "mpiexec"


@dataclass
class ClusterConfig:
    """Cluster configuration."""

    # Machine name recorded with each run (e.g. "Alpha", "EmpireMacbook")
    machine_name: str = "local"
    # "none" (run locally), "pbs" or "slurm"
    scheduler: str = "none"
    account: Optional[str] = None
    # Queue/partition (resources.queue overrides this)
    queue: Optional[str] = None
    qos: Optional[str] = None
    # PBS "-l filesystems=..." value (e.g. home:eagle on Polaris); null to skip
    filesystems: Optional[str] = None
    # torchrun, srun (Slurm, one task per GPU) or mpiexec (PBS)
    launcher: Launcher = Launcher.torchrun
    # Sourced at job start (conda activate, module load, ...); null to skip
    setup_script: Optional[str] = None
    master_port: int = 29500
    # Extra environment variables exported into the job
    env: Dict[str, Any] = field(default_factory=dict)
    # Raw extra sbatch arguments, e.g. ["--constraint=a100"]
    extra_args: List[str] = field(default_factory=list)


# =============================================================================
# Resources Configuration
# =============================================================================

@dataclass
class ResourcesConfig:
    """Resources configuration. Null fields are not requested from the scheduler."""

    nnodes: int = 1
    # Null: use the GPUs visible on the node (local runs and launch.sh)
    gpus_per_node: Optional[int] = 1
    # Slurm GPU type (e.g. "rtx_pro_6000_blackwell"); requests
    # --gres=gpu:<type>:<n> instead of --gpus-per-node=<n>. Null: any GPU
    gpu_type: Optional[str] = None
    cpus_per_node: Optional[int] = 8
    mem: Optional[str] = "32gb"
    walltime: Optional[str] = "4:00:00"
    omp_num_threads: Optional[int] = 8
    # Queue/partition override (null = use cluster.queue)
    queue: Optional[str] = None

# =============================================================================
# Job Configuration
# =============================================================================


@dataclass
class JobConfig:
    """How scripts/run.py runs the job, plus batch job settings."""

    # Scheduler job name (also names the <name>.o<job_id> output file)
    name: str = "particleman_train"
    nccl_debug: str = "WARN"
    # "auto", "train" (in this process), "launch" (scripts/jobs/launch.sh)
    # or "submit" (batch job)
    mode: str = "auto"
    # Submit mode: print the submit command and job environment only
    dry_run: bool = False


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
    """Root schema combining all group schemas.

    Root YAML files (configs/config.yaml, bbllv08.yaml, ...) list
    ``base_config`` first in their defaults list to pick it up.
    """

    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    job: JobConfig = field(default_factory=JobConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def register_configs() -> None:
    """Store the schemas in Hydra's ConfigStore.

    Call this before composing any config (before ``@hydra.main`` runs or
    ``compose()`` is called). The YAML files reference these names in their
    defaults lists.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
    cs.store(group="data", name="base_data", node=DataSourceConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="output", name="base_output", node=OutputConfig)
    cs.store(group="cluster", name="base_cluster", node=ClusterConfig)
    cs.store(group="resources", name="base_resources", node=ResourcesConfig)
    cs.store(group="job", name="base_job", node=JobConfig)
    cs.store(group="distributed", name="base_distributed", node=DistributedConfig)
