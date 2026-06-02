"""
Models module for ParticleMan.

This module contains neural network architectures for particle physics tasks.
"""

from .particle_transformer import (
    ParticleTransformer,
    AttentionBiasTransformer,
    AngularAttentionBias,
    ParticleConfig,
    EmbeddingType,
    PhiEncoding,
    ConcatEmbedding,
    JointEmbedding,
)
from .vae import VAEParticleModel, VAEConfig, compute_vae_loss

__all__ = [
    "ParticleTransformer",
    "AttentionBiasTransformer",
    "AngularAttentionBias",
    "ParticleConfig",
    "EmbeddingType",
    "PhiEncoding",
    "ConcatEmbedding",
    "JointEmbedding",
    "VAEParticleModel",
    "VAEConfig",
    "compute_vae_loss",
]
