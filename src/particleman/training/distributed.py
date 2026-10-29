"""
Distributed training utilities for multi-GPU training.

This module provides utilities for training across multiple GPUs using
PyTorch's DistributedDataParallel (DDP).
"""

import datetime
import logging
import os
from typing import Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


# Env vars that launchers set, in priority order: (rank, world_size, local_rank).
# torchrun comes first so torchrun-inside-srun uses torchrun's per-GPU ranks
# rather than srun's per-node task IDs.
_RANK_ENV_SOURCES = (
    ("torchrun", "RANK", "WORLD_SIZE", ("LOCAL_RANK",)),
    ("slurm", "SLURM_PROCID", "SLURM_NTASKS", ("SLURM_LOCALID",)),
    ("mpich", "PMI_RANK", "PMI_SIZE", ("PALS_LOCAL_RANKID", "MPI_LOCALRANKID")),
    (
        "openmpi",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        ("OMPI_COMM_WORLD_LOCAL_RANK",),
    ),
)


def detect_rank_info(
    env: Mapping[str, str], gpus_per_node: Optional[int] = None
) -> Tuple[int, int, int, str]:
    """
    Detect process rank info from launcher environment variables.

    Supports torchrun, Slurm srun, MPICH/PALS mpiexec and OpenMPI mpirun.

    Args:
        env: Environment mapping (usually os.environ)
        gpus_per_node: Used to derive local_rank when the launcher doesn't set
            one (falls back to GPUS_PER_NODE in env, then 1)

    Returns:
        Tuple of (rank, world_size, local_rank, source), where source names the
        launcher detected ("none" for a plain single process)
    """
    for source, rank_var, size_var, local_vars in _RANK_ENV_SOURCES:
        if rank_var in env and size_var in env:
            rank = int(env[rank_var])
            world_size = int(env[size_var])
            local_rank = next(
                (int(env[v]) for v in local_vars if v in env), None
            )
            if local_rank is None:
                if gpus_per_node is None:
                    gpus_per_node = int(env.get("GPUS_PER_NODE", 1))
                local_rank = rank % max(gpus_per_node, 1)
            return rank, world_size, local_rank, source
    return 0, 1, 0, "none"


def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl",
    timeout_s: int = 60,
) -> Tuple[int, int, torch.device]:
    """
    Initialize distributed training environment.
    
    This function handles both:
    - Manual setup (rank/world_size provided)
    - Environment-based setup (torchrun, srun or mpiexec; see detect_rank_info)

    MASTER_ADDR/MASTER_PORT must be set by the launcher (scripts/jobs/launch.sh
    exports them); they default to localhost:29500.
    
    Args:
        rank: Process rank (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        timeout_s: Process group initialization timeout in seconds
    
    Returns:
        Tuple of (rank, world_size, device)
    """
    gpus_per_node = torch.cuda.device_count() or None
    env_rank, env_world_size, local_rank, source = detect_rank_info(
        os.environ, gpus_per_node=gpus_per_node
    )
    if rank is None:
        rank = env_rank
    if world_size is None:
        world_size = env_world_size

    # Check if we're actually doing distributed training
    if world_size <= 1:
        logger.info("Single GPU/CPU training (no distributed setup needed)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    logger.info(f"Detected launcher environment: {source}")

    # Required by torch.distributed with init_method="env://"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Set the device
    if torch.cuda.is_available():
        # If the launcher binds one GPU per rank only one device is visible;
        # otherwise all node GPUs are visible and local_rank picks one.
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
        backend = "gloo"  # NCCL doesn't support CPU
    
    # Initialize process group
    if not dist.is_initialized():
        # Set master address and port if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        dist.init_process_group(backend=backend, init_method="env://", timeout=datetime.timedelta(seconds=timeout_s))
    
    logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, device={device}")
    
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap a model with DistributedDataParallel.
    
    Args:
        model: The model to wrap
        device: Device the model is on
        find_unused_parameters: Whether to find unused parameters (slower but safer)
    
    Returns:
        DDP-wrapped model (or original model if not distributed)
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return model.to(device)
    
    model = model.to(device)
    model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )
    
    return model


def create_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    drop_last: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with DistributedSampler for multi-GPU training.
    
    Args:
        dataset: The dataset to load from
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle (handled by sampler in distributed mode)
        num_workers: Number of data loading workers
        persistent_workers: Keep workers alive between epochs (only applies
            when num_workers > 0).
        pin_memory: Whether to pin memory for CUDA transfers
        drop_last: Whether to drop the last incomplete batch
        rank: Process rank
        world_size: Total number of processes
        seed: Random seed for sampler
    
    Returns:
        DataLoader configured for distributed training
    """
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return dataloader


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce a tensor across all processes (average).
    
    Args:
        tensor: Tensor to reduce
        world_size: Number of processes
    
    Returns:
        Reduced tensor (averaged across processes)
    """
    if world_size <= 1 or not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    
    return rt


def sync_across_processes() -> None:
    """Synchronize all processes (barrier)."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
