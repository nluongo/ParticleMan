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

__all__ = [
    "ParticleTransformer",
    "AttentionBiasTransformer",
    "AngularAttentionBias",
    "ParticleConfig",
    "EmbeddingType",
    "PhiEncoding",
    "ConcatEmbedding",
    "JointEmbedding",
]
