"""Particle Transformer model for pre-training on particle physics data."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbeddingType(Enum):
    """Type of particle embedding strategy."""
    CONCAT = "concat"  # Independent projections, then concatenate
    JOINT = "joint"    # Joint MLP over all features


class PhiEncoding(Enum):
    """How to encode the phi (azimuthal angle) feature."""
    RAW = "raw"        # Use phi directly (normalized to [-1, 1])
    SINCOS = "sincos"  # Use sin(phi) and cos(phi) separately


@dataclass
class ParticleConfig:
    """Configuration for the Particle Transformer model."""
    
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_particles: int = 100
    n_particle_types: int = 16  # 0-14 for particles, 15 for unknown/padding
    mask_prob: float = 0.15
    mask_continuous_std: float = 0.1
    pt_range: Tuple[float, float] = (0.0, 1000.0)  # pt is always positive (normalized to [0, 1])
    eta_range: Tuple[float, float] = (-5.0, 5.0)   # eta can be +/- (normalized to [-1, 1])
    phi_range: Tuple[float, float] = (-3.14159, 3.14159)  # phi in [-π, π] (normalized to [-1, 1])
    # Embedding configuration
    embedding_type: EmbeddingType = EmbeddingType.JOINT
    phi_encoding: PhiEncoding = PhiEncoding.RAW  # How to encode phi: "raw" or "sincos"
    id_embed_dim: int = 32  # Dimension for particle ID embedding (used in JOINT mode)
    embed_hidden_dim: int = 128  # Hidden dimension for embedding MLP (used in JOINT mode)


class ConcatEmbedding(nn.Module):
    """
    Embed particle features by projecting each feature independently, then concatenating.
    
    Each feature (pt, eta, phi, id) is projected to d_model/4 dimensions independently,
    then concatenated to form the final d_model-dimensional embedding.
    
    Architecture (RAW phi):
        pt  → Linear(1, d_model//4)  ─┐
        eta → Linear(1, d_model//4)  ─┼─→ concat → (d_model,)
        phi → Linear(1, d_model//4)  ─┤
        id  → Embedding(d_model//4)  ─┘
    
    Architecture (SINCOS phi):
        pt      → Linear(1, d_model//4)  ─┐
        eta     → Linear(1, d_model//4)  ─┼─→ concat → (d_model,)
        sin/cos → Linear(2, d_model//4)  ─┤
        id      → Embedding(d_model//4)  ─┘
    """
    
    def __init__(self, config: ParticleConfig) -> None:
        super().__init__()
        self.config = config
        feature_dim = config.d_model // 4
        
        self.pt_proj = nn.Linear(1, feature_dim)
        self.eta_proj = nn.Linear(1, feature_dim)
        
        # Phi projection depends on encoding type
        if config.phi_encoding == PhiEncoding.SINCOS:
            self.phi_proj = nn.Linear(2, feature_dim)  # sin(phi), cos(phi)
        else:
            self.phi_proj = nn.Linear(1, feature_dim)  # raw phi
            
        self.particle_id_embedding = nn.Embedding(config.n_particle_types, feature_dim)
    
    def forward(
        self, pt_norm: Tensor, eta_norm: Tensor, phi_processed: Tensor, particle_id: Tensor
    ) -> Tensor:
        """
        Embed particles by projecting features independently and concatenating.
        
        Args:
            pt_norm: Normalized pT in [0, 1] (batch, seq)
            eta_norm: Normalized eta in [-1, 1] (batch, seq)
            phi_processed: For RAW encoding: normalized phi in [-1, 1].
                          For SINCOS encoding: raw phi in radians.
            particle_id: Particle type IDs (batch, seq)
        
        Returns:
            Particle embeddings of shape (batch, seq, d_model)
        """
        pt_emb = self.pt_proj(pt_norm.unsqueeze(-1))
        eta_emb = self.eta_proj(eta_norm.unsqueeze(-1))
        
        # Encode phi based on config
        if self.config.phi_encoding == PhiEncoding.SINCOS:
            # Compute sin(phi) and cos(phi) from raw phi
            sin_phi = torch.sin(phi_processed)
            cos_phi = torch.cos(phi_processed)
            phi_features = torch.stack([sin_phi, cos_phi], dim=-1)  # (batch, seq, 2)
            phi_emb = self.phi_proj(phi_features)
        else:
            # phi_processed is already normalized to [-1, 1]
            phi_emb = self.phi_proj(phi_processed.unsqueeze(-1))
        
        pid_emb = self.particle_id_embedding(particle_id)
        
        return torch.cat([pt_emb, eta_emb, phi_emb, pid_emb], dim=-1)


class JointEmbedding(nn.Module):
    """
    Embed particle features jointly through an MLP (Particle Transformer style).
    
    All features are fed together into an MLP, allowing the network to learn
    cross-feature interactions during the embedding stage.
    
    Architecture (RAW phi):
        [pt, eta, phi, id_emb] → Linear → GELU → Linear → (d_model,)
        Input dim: 3 + id_embed_dim
    
    Architecture (SINCOS phi):
        [pt, eta, sin(phi), cos(phi), id_emb] → Linear → GELU → Linear → (d_model,)
        Input dim: 4 + id_embed_dim
    
    Reference:
        Qu & Gouskos, "Particle Transformer for Jet Tagging" (2022)
    """
    
    def __init__(self, config: ParticleConfig) -> None:
        super().__init__()
        self.config = config
        
        self.particle_id_embedding = nn.Embedding(
            config.n_particle_types, config.id_embed_dim
        )
        
        # Input dimension depends on phi encoding
        # RAW: pt, eta, phi (3 features)
        # SINCOS: pt, eta, sin(phi), cos(phi) (4 features)
        if config.phi_encoding == PhiEncoding.SINCOS:
            n_continuous = 4  # pt, eta, sin(phi), cos(phi)
        else:
            n_continuous = 3  # pt, eta, phi
        
        input_dim = n_continuous + config.id_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.embed_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )
    
    def forward(
        self, pt_norm: Tensor, eta_norm: Tensor, phi_processed: Tensor, particle_id: Tensor
    ) -> Tensor:
        """
        Embed particles by jointly processing all features through an MLP.
        
        Args:
            pt_norm: Normalized pT in [0, 1] (batch, seq)
            eta_norm: Normalized eta in [-1, 1] (batch, seq)
            phi_processed: For RAW encoding: normalized phi in [-1, 1].
                          For SINCOS encoding: raw phi in radians.
            particle_id: Particle type IDs (batch, seq)
        
        Returns:
            Particle embeddings of shape (batch, seq, d_model)
        """
        # Build continuous features based on phi encoding
        if self.config.phi_encoding == PhiEncoding.SINCOS:
            # Compute sin(phi) and cos(phi) from raw phi
            sin_phi = torch.sin(phi_processed)
            cos_phi = torch.cos(phi_processed)
            continuous = torch.stack([pt_norm, eta_norm, sin_phi, cos_phi], dim=-1)
        else:
            # phi_processed is already normalized to [-1, 1]
            continuous = torch.stack([pt_norm, eta_norm, phi_processed], dim=-1)
        
        # Get particle ID embedding: (batch, seq, id_embed_dim)
        id_emb = self.particle_id_embedding(particle_id)
        
        # Concatenate and project: (batch, seq, d_model)
        x = torch.cat([continuous, id_emb], dim=-1)
        return self.mlp(x)


class ParticleTransformer(nn.Module):
    """Transformer model for particle physics pre-training."""
    
    def __init__(self, config: ParticleConfig) -> None:
        super().__init__()
        self.config = config
        
        # Select embedding strategy
        if config.embedding_type == EmbeddingType.CONCAT:
            self.embedding = ConcatEmbedding(config)
        else:  # JOINT (default)
            self.embedding = JointEmbedding(config)
        
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
    
    def _normalize_features(
        self, pt: Tensor, eta: Tensor, phi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normalize continuous features to standard ranges.
        
        Args:
            pt: Raw pT values (batch, seq)
            eta: Raw eta values (batch, seq)  
            phi: Raw phi values in radians (batch, seq)
        
        Returns:
            Tuple of (pt_norm, eta_norm, phi_processed):
            - pt_norm: pT normalized to [0, 1]
            - eta_norm: eta normalized to [-1, 1]
            - phi_processed: For RAW encoding, phi normalized to [-1, 1].
                            For SINCOS encoding, raw phi (embedding computes sin/cos).
        """
        import math
        
        pt_norm = (pt - self.config.pt_range[0]) / (self.config.pt_range[1] - self.config.pt_range[0])
        eta_norm = 2 * (eta - self.config.eta_range[0]) / (self.config.eta_range[1] - self.config.eta_range[0]) - 1
        
        # Wrap phi to [-π, π] to handle values outside the standard range
        # This ensures phi and phi + 2πn produce the same result for both encodings
        phi_wrapped = torch.remainder(phi + math.pi, 2 * math.pi) - math.pi
        
        if self.config.phi_encoding == PhiEncoding.RAW:
            # Normalize wrapped phi to [-1, 1] for RAW encoding
            phi_processed = 2 * (phi_wrapped - self.config.phi_range[0]) / (self.config.phi_range[1] - self.config.phi_range[0]) - 1
        else:
            # For SINCOS, pass wrapped phi - embedding layer computes sin/cos
            phi_processed = phi_wrapped
        
        return pt_norm, eta_norm, phi_processed
    
    def forward(
        self, 
        pt: Tensor, 
        eta: Tensor, 
        phi: Tensor, 
        particle_id: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the model.
        
        Args:
            pt: Transverse momentum (batch, seq)
            eta: Pseudorapidity (batch, seq)
            phi: Azimuthal angle (batch, seq)
            particle_id: Particle type IDs (batch, seq)
            padding_mask: Boolean mask where True indicates padding (batch, seq).
                         If None, no masking is applied.
        
        Returns:
            Dictionary with predictions and hidden states.
        """
        # Normalize features
        pt_norm, eta_norm, phi_norm = self._normalize_features(pt, eta, phi)
        
        # Create embeddings
        embeddings = self.embedding(pt_norm, eta_norm, phi_norm, particle_id)
        
        # Apply transformer with optional padding mask
        # PyTorch transformer expects src_key_padding_mask where True = ignore
        hidden_states = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        
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
        padding_mask: Optional[Tensor] = None,
        mask_prob: Optional[float] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Create random masks for pre-training.
        
        Only real particles (non-padding) are masked. Padded positions are never
        masked and never contribute to the loss.
        
        Args:
            pt, eta, phi, particle_id: Input tensors (batch, seq)
            padding_mask: Boolean mask where True indicates padding positions.
                         If None, all positions are treated as real particles.
            mask_prob: Probability of masking each real particle.
        
        Returns:
            Tuple of (masked_inputs, mask_targets) dictionaries.
        """
        if mask_prob is None:
            mask_prob = self.config.mask_prob
        
        batch_size, seq_len = pt.shape
        device = pt.device
        
        # Create random mask for prediction targets
        random_mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Only mask real particles, not padding
        if padding_mask is not None:
            # padding_mask: True = padding, False = real particle
            # We want to mask only real particles
            prediction_mask = random_mask & (~padding_mask)
        else:
            prediction_mask = random_mask
        
        # Create masked inputs
        masked_pt = pt.clone()
        masked_eta = eta.clone() 
        masked_phi = phi.clone()
        masked_particle_id = particle_id.clone()
        
        if prediction_mask.sum() > 0:
            # Add noise to continuous features for masked positions
            masked_pt[prediction_mask] = torch.normal(
                mean=masked_pt[prediction_mask], 
                std=self.config.mask_continuous_std
            )
            masked_eta[prediction_mask] = torch.normal(
                mean=masked_eta[prediction_mask], 
                std=self.config.mask_continuous_std
            )
            masked_phi[prediction_mask] = torch.normal(
                mean=masked_phi[prediction_mask], 
                std=self.config.mask_continuous_std
            )
            
            # Replace particle IDs with mask token (last ID)
            masked_particle_id[prediction_mask] = self.config.n_particle_types - 1
        
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
            'mask': prediction_mask  # Only True for real particles that are masked
        }
        
        return masked_inputs, mask_targets
    
    def compute_loss(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Compute pre-training loss.
        
        Loss is only computed on masked positions (real particles that were masked).
        
        Args:
            predictions: Model predictions dictionary.
            targets: Target values dictionary with 'mask' indicating positions to predict.
        
        Returns:
            Dictionary of loss values.
        """
        mask = targets['mask']
        device = predictions['pt'].device
        
        if mask.sum() > 0:
            pt_loss = F.mse_loss(predictions['pt'][mask], targets['pt'][mask])
            eta_loss = F.mse_loss(predictions['eta'][mask], targets['eta'][mask])
            phi_loss = F.mse_loss(predictions['phi'][mask], targets['phi'][mask])
            particle_id_loss = F.cross_entropy(
                predictions['particle_id'][mask], 
                targets['particle_id'][mask]
            )
        else:
            pt_loss = torch.tensor(0.0, device=device)
            eta_loss = torch.tensor(0.0, device=device)
            phi_loss = torch.tensor(0.0, device=device)
            particle_id_loss = torch.tensor(0.0, device=device)
        
        total_loss = pt_loss + eta_loss + phi_loss + particle_id_loss
        
        return {
            'total_loss': total_loss,
            'pt_loss': pt_loss,
            'eta_loss': eta_loss,
            'phi_loss': phi_loss,
            'particle_id_loss': particle_id_loss
        }
