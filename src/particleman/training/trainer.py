"""
Trainer for particle physics pre-training.

This module provides a unified trainer class for pre-training particle transformer
models using masked prediction. Supports both single-GPU and distributed multi-GPU
training.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.particle_transformer import ParticleTransformer, ParticleConfig
from ..loggers.base import BaseLogger, NoOpLogger
from .distributed import (
    is_main_process,
    wrap_model_ddp,
    reduce_tensor,
    sync_across_processes,
)

logger = logging.getLogger(__name__)


class ParticleTrainer:
    """
    Unified trainer for particle transformer pre-training.
    
    Supports both single-GPU and distributed multi-GPU training. When world_size > 1,
    the model is wrapped with DistributedDataParallel and gradient synchronization
    is performed automatically.
    """
    
    def __init__(
        self,
        model: ParticleTransformer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        rank: int = 0,
        world_size: int = 1,
        log_interval: int = 100,
        save_dir: Optional[Path] = None,
        experiment_logger: Optional[BaseLogger] = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: The particle transformer model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of epochs to train (used for learning rate scheduler)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            device: Device to train on (auto-detect if None)
            rank: Process rank for distributed training (0 = main process)
            world_size: Total number of processes (1 = single GPU)
            log_interval: How often to log training stats
            save_dir: Directory to save checkpoints
            experiment_logger: Logger for experiment tracking (MLflow, Comet, etc.)
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.num_epochs = num_epochs
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)
        self.is_distributed = world_size > 1
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else None
        self.experiment_logger = experiment_logger or NoOpLogger()
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                if self.is_distributed:
                    local_rank = rank % torch.cuda.device_count()
                    self.device = torch.device(f"cuda:{local_rank}")
                else:
                    self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        
        # Store raw model reference and optionally wrap with DDP
        self.raw_model = model
        if self.is_distributed:
            self.model = wrap_model_ddp(model, self.device)
        else:
            self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler using actual num_epochs
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create save directory (only on main process)
        if self.save_dir and self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_model_for_inference(self) -> ParticleTransformer:
        """Get the underlying model (unwrapped from DDP if necessary)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of particle data with keys:
                - pt, eta, phi, particle_id: Particle features
                - mask: Boolean mask where True = real particle, False = padding
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move batch to device
        pt = batch['pt'].to(self.device)
        eta = batch['eta'].to(self.device)
        phi = batch['phi'].to(self.device)
        particle_id = batch['particle_id'].to(self.device)
        
        # Get padding mask from batch
        # batch['mask'] is True for real particles, we need True for padding
        real_particle_mask = batch['mask'].to(self.device)
        padding_mask = ~real_particle_mask  # Invert: True = padding
        
        # Get the raw model for create_masks (DDP wrapper doesn't have this method)
        raw_model = self._get_model_for_inference()
        
        # Create masks for prediction (only mask real particles)
        masked_inputs, mask_targets = raw_model.create_masks(
            pt, eta, phi, particle_id, 
            padding_mask=padding_mask
        )
        
        # Forward pass with masked inputs and padding mask
        predictions = self.model(
            masked_inputs['pt'],
            masked_inputs['eta'],
            masked_inputs['phi'],
            masked_inputs['particle_id'],
            padding_mask=padding_mask,
        )
        
        # Compute loss (only on masked real particles)
        losses = raw_model.compute_loss(predictions, mask_targets)
        
        # Scale loss for gradient accumulation
        loss = losses['total_loss'] / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Convert to float for logging
        return {k: v.item() for k, v in losses.items()}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a given dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            
        Returns:
            Dictionary of loss values (averaged across processes if distributed)
        """
        if dataloader is None:
            return {}
        
        self.model.eval()
        raw_model = self._get_model_for_inference()
        
        total_losses = {
            'total_loss': 0.0,
            'pt_loss': 0.0,
            'eta_loss': 0.0,
            'phi_loss': 0.0,
            'particle_id_loss': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                pt = batch['pt'].to(self.device)
                eta = batch['eta'].to(self.device)
                phi = batch['phi'].to(self.device)
                particle_id = batch['particle_id'].to(self.device)
                
                # Get padding mask
                real_particle_mask = batch['mask'].to(self.device)
                padding_mask = ~real_particle_mask
                
                # Create masks
                masked_inputs, mask_targets = raw_model.create_masks(
                    pt, eta, phi, particle_id,
                    padding_mask=padding_mask
                )
                
                # Forward pass
                predictions = self.model(
                    masked_inputs['pt'],
                    masked_inputs['eta'],
                    masked_inputs['phi'],
                    masked_inputs['particle_id'],
                    padding_mask=padding_mask,
                )
                
                # Compute loss
                losses = raw_model.compute_loss(predictions, mask_targets)
                
                # Accumulate losses
                for k, v in losses.items():
                    total_losses[k] += v.item()
                
                num_batches += 1
        
        # Average losses locally
        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches
        
        # Reduce across all processes if distributed
        if self.is_distributed:
            for k in total_losses:
                tensor = torch.tensor(total_losses[k], device=self.device)
                total_losses[k] = reduce_tensor(tensor, self.world_size).item()
        
        return total_losses

    def validate(self) -> Dict[str, float]:
        """Perform validation on the validation set."""
        return self.evaluate(self.val_dataloader)

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint (only on main process)."""
        if not self.is_main:
            return
            
        if self.save_dir is None:
            logger.warning("No save directory specified, skipping checkpoint save")
            return
        
        # Save the unwrapped model
        raw_model = self._get_model_for_inference()
        
        checkpoint = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'config': raw_model.config,
            'world_size': self.world_size,
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        raw_model = self._get_model_for_inference()
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        if self.is_main:
            logger.info(f"Loaded checkpoint from {filepath}")
    
    def train(self) -> None:
        """Train the model for the number of epochs specified in the constructor."""
        num_epochs = self.num_epochs
        
        if self.is_main:
            logger.info(f"Starting training for {num_epochs} epochs")
            if self.is_distributed:
                logger.info(f"Distributed training with world_size={self.world_size}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Do initial loss logging for both train and val (only on main process)
        if self.is_main:
            logger.info("Performing initial loss calculation over training dataset")
        init_train_losses = self.evaluate(self.train_dataloader)
        if init_train_losses and self.is_main:
            logger.info(f"Epoch 0 - Initial Training Loss: {init_train_losses['total_loss']:.4f}")
            self.experiment_logger.log_metrics({f"train_{k}": v for k, v in init_train_losses.items()}, step=self.step)

        if self.is_main:
            logger.info("Performing initial loss calculation over validation dataset")
        init_val_losses = self.evaluate(self.val_dataloader)
        if init_val_losses and self.is_main:
            logger.info(f"Epoch 0 - Initial Validation Loss: {init_val_losses['total_loss']:.4f}")
            self.experiment_logger.log_metrics({f"val_{k}": v for k, v in init_val_losses.items()}, step=self.step)

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if hasattr(self.train_dataloader, 'sampler') and \
               hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Training loop with progress bar (only on main process)
            if self.is_main:
                pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = self.train_dataloader
            
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # Gradient accumulation: step optimizer every N batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.step += 1
                    
                    # Log training stats (only on main process)
                    if self.is_main and self.step % self.log_interval == 0 and len(epoch_losses) >= self.log_interval:
                        avg_loss = sum(l['total_loss'] for l in epoch_losses[-self.log_interval:]) / self.log_interval
                        lr = self.scheduler.get_last_lr()[0]
                        pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{lr:.2e}"})
                        
                        metrics_to_log = losses.copy()
                        metrics_to_log['avg_loss'] = avg_loss
                        metrics_to_log['lr'] = lr
                        metrics_to_log = {f"train_{k}": v for k, v in metrics_to_log.items()}
                        self.experiment_logger.log_metrics(metrics_to_log, step=self.step)
            
            # Synchronize before validation (if distributed)
            if self.is_distributed:
                sync_across_processes()
            
            # Validation
            val_losses = self.validate()
            if val_losses and self.is_main:
                val_loss = val_losses['total_loss']
                logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
                self.experiment_logger.log_metrics({f"val_{k}": v for k, v in val_losses.items()}, step=self.step)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Synchronize before next epoch (if distributed)
            if self.is_distributed:
                sync_across_processes()
        
        if self.is_main:
            logger.info("Training completed!")
        self.save_checkpoint("final_model.pt")
