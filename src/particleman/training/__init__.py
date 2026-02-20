"""
Training module for ParticleMan.

This module contains training utilities for particle physics models.
"""

from .trainer import ParticleTrainer
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    wrap_model_ddp,
    create_distributed_dataloader,
    reduce_tensor,
    sync_across_processes,
)
from .output_manager import OutputManager

__all__ = [
    "ParticleTrainer",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "wrap_model_ddp",
    "create_distributed_dataloader",
    "reduce_tensor",
    "sync_across_processes",
    "OutputManager",
] 