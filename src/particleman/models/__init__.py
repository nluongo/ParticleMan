"""
Models module for ParticleMan.

This module contains neural network architectures for particle physics tasks.
"""

from .particle_transformer import (
    ParticleTransformer,
    ParticleConfig,
    EmbeddingType,
    ConcatEmbedding,
    JointEmbedding,
)

__all__ = [
    "ParticleTransformer",
    "ParticleConfig",
    "EmbeddingType",
    "ConcatEmbedding",
    "JointEmbedding",
]
