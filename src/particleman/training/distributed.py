"""
Distributed training utilities for multi-GPU training.

This module provides utilities for training across multiple GPUs using
PyTorch's DistributedDataParallel (DDP).
"""

import datetime
import logging
import socket
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def _rank_info_from_mpi() -> Tuple[int, int, int]:
    """
    Discover rank info via mpi4py and export it for torch.distributed.

    Used for mpiexec launches, where torchrun-style env vars are not set.
    Rank 0's hostname is broadcast as MASTER_ADDR.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    try:
        from mpi4py import MPI
    except ImportError as e:
        raise ImportError(
            "distributed.use_mpi=true requires mpi4py. Install it in the environment "
            "(preferably built against the system MPI) or set distributed.use_mpi=false."
        ) from e

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Required by torch.distributed with init_method="env://"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = comm.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "2345"

    # Polaris: 4 GPUs/node, ranks packed by node with --ppn 4
    local_rank = rank % 4

    return rank, world_size, local_rank


def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl",
    use_mpi: bool = False,
) -> Tuple[int, int, torch.device]:
    """
    Initialize distributed training environment.
    
    This function handles both:
    - Manual setup (rank/world_size provided)
    - Environment-based setup (from SLURM, PBS, or torchrun)
    
    Args:
        rank: Process rank (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        use_mpi: Discover rank/world_size via mpi4py (for mpiexec launches).
            mpi4py is only imported when this is True.
    
    Returns:
        Tuple of (rank, world_size, device)
    """
    if use_mpi:
        rank, world_size, local_rank = _rank_info_from_mpi()
    else:
        if rank is None:
            rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", 0)))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))
        n_gpus = max(torch.cuda.device_count(), 1)
        local_rank = int(os.environ.get("LOCAL_RANK", rank % n_gpus))

    # Check if we're actually doing distributed training
    if world_size <= 1:
        logger.info("Single GPU/CPU training (no distributed setup needed)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    os.environ["LOCAL_RANK"] = str(local_rank)
    if use_mpi:
        # Strong per-rank binding
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    # Set the device
    if torch.cuda.is_available():
        # With per-rank binding (MPI, or CUDA_VISIBLE_DEVICES set by the launcher)
        # only one device is visible; under torchrun all node GPUs are visible.
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
        
        dist.init_process_group(backend=backend, init_method="env://", timeout=datetime.timedelta(seconds=60))
    
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
