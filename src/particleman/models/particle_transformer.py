"""Particle Transformer model for pre-training on particle physics data."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ParticleConfig:
    """Configuration for the Particle Transformer model."""
    
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_particles: int = 100
    n_particle_types: int = 13
    mask_prob: float = 0.15
    mask_continuous_std: float = 0.1
    pt_range: Tuple[float, float] = (0.0, 1000.0)  # pt is always positive (normalized to [0, 1])
    eta_range: Tuple[float, float] = (-5.0, 5.0)   # eta can be +/- (normalized to [-1, 1])
    phi_range: Tuple[float, float] = (-3.14159, 3.14159)  # phi in [-π, π] (normalized to [-1, 1])


class ParticleTransformer(nn.Module):
    """Transformer model for particle physics pre-training."""
    
    def __init__(self, config: ParticleConfig) -> None:
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.pt_proj = nn.Linear(1, config.d_model // 4)
        self.eta_proj = nn.Linear(1, config.d_model // 4)
        self.phi_proj = nn.Linear(1, config.d_model // 4)
        self.particle_id_embedding = nn.Embedding(config.n_particle_types, config.d_model // 4)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Prediction heads
        self.pt_head = nn.Linear(config.d_model, 1)
        self.eta_head = nn.Linear(config.d_model, 1)
        self.phi_head = nn.Linear(config.d_model, 1)
        self.particle_id_head = nn.Linear(config.d_model, config.n_particle_types)
    
    def forward(
        self, 
        pt: Tensor, 
        eta: Tensor, 
        phi: Tensor, 
        particle_id: Tensor
    ) -> Dict[str, Tensor]:
        """Forward pass of the model."""
        # Normalize features
        pt_norm = (pt - self.config.pt_range[0]) / (self.config.pt_range[1] - self.config.pt_range[0])  # [0, 1]
        eta_norm = 2 * (eta - self.config.eta_range[0]) / (self.config.eta_range[1] - self.config.eta_range[0]) - 1  # [-1, 1]
        phi_norm = 2 * (phi - self.config.phi_range[0]) / (self.config.phi_range[1] - self.config.phi_range[0]) - 1  # [-1, 1]
        
        # Create embeddings
        pt_emb = self.pt_proj(pt_norm.unsqueeze(-1))
        eta_emb = self.eta_proj(eta_norm.unsqueeze(-1))
        phi_emb = self.phi_proj(phi_norm.unsqueeze(-1))
        pid_emb = self.particle_id_embedding(particle_id)
        
        # Combine embeddings
        embeddings = torch.cat([pt_emb, eta_emb, phi_emb, pid_emb], dim=-1)
        
        # Apply transformer
        hidden_states = self.transformer(embeddings)
        
        # Generate predictions
        return {
            'pt': self.pt_head(hidden_states).squeeze(-1),
            'eta': self.eta_head(hidden_states).squeeze(-1),
            'phi': self.phi_head(hidden_states).squeeze(-1),
            'particle_id': self.particle_id_head(hidden_states),
            'hidden_states': hidden_states
        }
    
    def create_masks(
        self, 
        pt: Tensor, 
        eta: Tensor, 
        phi: Tensor, 
        particle_id: Tensor,
        mask_prob: Optional[float] = None
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Create random masks for pre-training."""
        if mask_prob is None:
            mask_prob = self.config.mask_prob
        
        batch_size, seq_len = pt.shape
        device = pt.device
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Create masked inputs
        masked_pt = pt.clone()
        masked_eta = eta.clone() 
        masked_phi = phi.clone()
        masked_particle_id = particle_id.clone()
        
        if mask.sum() > 0:
            # Add noise to continuous features
            masked_pt[mask] = torch.normal(mean=masked_pt[mask], std=self.config.mask_continuous_std)
            masked_eta[mask] = torch.normal(mean=masked_eta[mask], std=self.config.mask_continuous_std)
            masked_phi[mask] = torch.normal(mean=masked_phi[mask], std=self.config.mask_continuous_std)
            
            # Mask particle IDs
            masked_particle_id[mask] = self.config.n_particle_types - 1
        
        masked_inputs = {
            'pt': masked_pt,
            'eta': masked_eta,
            'phi': masked_phi,
            'particle_id': masked_particle_id
        }
        
        mask_targets = {
            'pt': pt,
            'eta': eta,
            'phi': phi,
            'particle_id': particle_id,
            'mask': mask
        }
        
        return masked_inputs, mask_targets
    
    def compute_loss(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Compute pre-training loss."""
        mask = targets['mask']
        
        pt_loss = F.mse_loss(predictions['pt'][mask], targets['pt'][mask]) if mask.sum() > 0 else torch.tensor(0.0)
        eta_loss = F.mse_loss(predictions['eta'][mask], targets['eta'][mask]) if mask.sum() > 0 else torch.tensor(0.0)
        phi_loss = F.mse_loss(predictions['phi'][mask], targets['phi'][mask]) if mask.sum() > 0 else torch.tensor(0.0)
        particle_id_loss = F.cross_entropy(predictions['particle_id'][mask], targets['particle_id'][mask]) if mask.sum() > 0 else torch.tensor(0.0)
        
        total_loss = pt_loss + eta_loss + phi_loss + particle_id_loss
        
        return {
            'total_loss': total_loss,
            'pt_loss': pt_loss,
            'eta_loss': eta_loss,
            'phi_loss': phi_loss,
            'particle_id_loss': particle_id_loss
        }
