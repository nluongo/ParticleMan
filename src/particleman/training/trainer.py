"""
Trainer for particle physics pre-training.

This module provides a trainer class for pre-training particle transformer models
using masked prediction.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.particle_transformer import ParticleTransformer, ParticleConfig

logger = logging.getLogger(__name__)


class ParticleTrainer:
    """Trainer for particle transformer pre-training."""
    
    def __init__(
        self,
        model: ParticleTransformer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        log_interval: int = 100,
        save_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: The particle transformer model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            device: Device to train on (auto-detect if None)
            log_interval: How often to log training stats
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=len(train_dataloader) * 10  # Assume 10 epochs
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create save directory
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Create masks for prediction (only mask real particles)
        masked_inputs, mask_targets = self.model.create_masks(
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
        losses = self.model.compute_loss(predictions, mask_targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert to float for logging
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self) -> Dict[str, float]:
        """
        Perform validation.
        
        Returns:
            Dictionary of validation loss values
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_losses = {
            'total_loss': 0.0,
            'pt_loss': 0.0,
            'eta_loss': 0.0,
            'phi_loss': 0.0,
            'particle_id_loss': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                pt = batch['pt'].to(self.device)
                eta = batch['eta'].to(self.device)
                phi = batch['phi'].to(self.device)
                particle_id = batch['particle_id'].to(self.device)
                
                # Get padding mask
                real_particle_mask = batch['mask'].to(self.device)
                padding_mask = ~real_particle_mask
                
                # Create masks
                masked_inputs, mask_targets = self.model.create_masks(
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
                losses = self.model.compute_loss(predictions, mask_targets)
                
                # Accumulate losses
                for k, v in losses.items():
                    total_losses[k] += v.item()
                
                num_batches += 1
        
        # Average losses
        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches
        
        return total_losses
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.save_dir is None:
            logger.warning("No save directory specified, skipping checkpoint save")
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'config': self.model.config
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def train(self, num_epochs: int) -> None:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                self.step += 1
                
                # Log training stats
                if self.step % self.log_interval == 0:
                    avg_loss = sum(l['total_loss'] for l in epoch_losses[-self.log_interval:]) / min(len(epoch_losses), self.log_interval)
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})
            
            # Validation
            val_losses = self.validate()
            if val_losses:
                val_loss = val_losses['total_loss']
                logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        logger.info("Training completed!")
        self.save_checkpoint("final_model.pt") 