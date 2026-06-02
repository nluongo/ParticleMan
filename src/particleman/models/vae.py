"""Variational Autoencoder for particle physics event generation."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .particle_transformer import (
    AttentionBiasTransformer,
    EmbeddingType,
    ParticleConfig,
    ParticleTransformer,
    PhiEncoding,
)


@dataclass
class VAEConfig:
    """Configuration for VAEParticleModel.

    All encoder fields mirror ParticleConfig; VAE-specific fields follow.
    """

    # --- Encoder fields (forwarded to ParticleConfig) ---
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_particles: int = 100
    n_particle_types: int = 16
    mask_prob: float = 0.15
    mask_continuous_std: float = 0.1
    pt_range: Tuple[float, float] = (0.0, 1000.0)
    eta_range: Tuple[float, float] = (-5.0, 5.0)
    phi_range: Tuple[float, float] = (-3.14159, 3.14159)
    embedding_type: EmbeddingType = EmbeddingType.JOINT
    phi_encoding: PhiEncoding = PhiEncoding.RAW
    id_embed_dim: int = 32
    embed_hidden_dim: int = 128
    angular_attention_bias: bool = False
    bias_hidden_dim: int = 32
    n_event_types: int = 0
    event_label_mask_prob: float = 0.15

    # --- VAE-specific ---
    latent_dim: int = 64
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_ff_dim: int = 512
    # 0 → use max_particles; supports future override for conditional generation
    n_decoder_queries: int = 0
    kl_weight: float = 1.0
    kl_warmup_steps: int = 0
    existence_loss_weight: float = 1.0

    def to_particle_config(self) -> ParticleConfig:
        """Build a ParticleConfig from the shared encoder fields."""
        return ParticleConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_particles=self.max_particles,
            n_particle_types=self.n_particle_types,
            mask_prob=self.mask_prob,
            mask_continuous_std=self.mask_continuous_std,
            pt_range=self.pt_range,
            eta_range=self.eta_range,
            phi_range=self.phi_range,
            embedding_type=self.embedding_type,
            phi_encoding=self.phi_encoding,
            id_embed_dim=self.id_embed_dim,
            embed_hidden_dim=self.embed_hidden_dim,
            angular_attention_bias=self.angular_attention_bias,
            bias_hidden_dim=self.bias_hidden_dim,
            n_event_types=self.n_event_types,
            event_label_mask_prob=self.event_label_mask_prob,
        )


class VAEParticleModel(nn.Module):
    """Variational Autoencoder built on the ParticleTransformer encoder.

    Encoder: ParticleTransformer → mean-pool → mu/logvar → z
    Decoder: learned query tokens cross-attend to z (as a single memory token)
             → per-slot reconstruction of (pt, eta, phi, particle_id) + existence logit

    The memory token is a single projection of z, making it straightforward to extend
    for conditional generation: observed particle hidden states can be concatenated
    to form a richer memory sequence.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        particle_config = config.to_particle_config()
        if config.angular_attention_bias:
            self.encoder: ParticleTransformer = AttentionBiasTransformer(particle_config)
        else:
            self.encoder = ParticleTransformer(particle_config)

        d = config.d_model
        self.mu_head = nn.Linear(d, config.latent_dim)
        self.logvar_head = nn.Linear(d, config.latent_dim)

        # Project latent z into a single decoder memory token
        self.latent_to_decoder = nn.Linear(config.latent_dim, d)

        n_queries = config.n_decoder_queries if config.n_decoder_queries > 0 else config.max_particles
        self._n_queries = n_queries
        # Scale init like Transformer embedding: N(0, d_model^-0.5)
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_queries, d) * math.sqrt(1.0 / d)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

        # Reconstruction heads (output in normalized feature space)
        self.recon_pt_head = nn.Linear(d, 1)
        self.recon_eta_head = nn.Linear(d, 1)
        self.recon_phi_head = nn.Linear(d, 1)
        self.recon_particle_id_head = nn.Linear(d, config.n_particle_types)
        # Existence: raw logit (real=1, padding=0); BCE loss in compute_vae_loss
        self.existence_head = nn.Linear(d, 1)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z from q(z|x) = N(mu, exp(logvar)) via reparameterization trick."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(
        self,
        pt: Tensor,
        eta: Tensor,
        phi: Tensor,
        particle_id: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encode event to latent z and decode to particle slots.

        Args:
            pt: (B, S) transverse momentum
            eta: (B, S) pseudorapidity
            phi: (B, S) azimuthal angle
            particle_id: (B, S) long, categorical IDs
            padding_mask: (B, S) bool, True = padding (matches PyTorch convention)

        Returns dict with keys:
            mu, logvar, z: (B, latent_dim)
            recon_pt, recon_eta, recon_phi: (B, N) normalized
            recon_particle_id: (B, N, n_particle_types) logits
            existence_logit: (B, N) raw logit
        """
        # Encode
        enc_out = self.encoder(pt, eta, phi, particle_id, padding_mask=padding_mask)
        hidden_states = enc_out["hidden_states"]  # (B, S, d_model)

        # Mean-pool over real (non-padding) particles
        if padding_mask is not None:
            real_mask_f = (~padding_mask).unsqueeze(-1).float()  # (B, S, 1)
        else:
            B_size, S_size = hidden_states.shape[:2]
            real_mask_f = torch.ones(
                B_size, S_size, 1, dtype=torch.float, device=hidden_states.device
            )
        event_emb = (hidden_states * real_mask_f).sum(dim=1) / real_mask_f.sum(dim=1).clamp(min=1)
        # event_emb: (B, d_model)

        mu = self.mu_head(event_emb)          # (B, latent_dim)
        logvar = self.logvar_head(event_emb)  # (B, latent_dim)
        z = self._reparameterize(mu, logvar)  # (B, latent_dim)

        # Decode: project z to single memory token
        memory = self.latent_to_decoder(z).unsqueeze(1)  # (B, 1, d_model)

        batch_size = pt.shape[0]
        tgt = self.query_tokens.expand(batch_size, -1, -1)  # (B, N, d_model)
        dec_out = self.decoder(tgt, memory)                  # (B, N, d_model)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "recon_pt": self.recon_pt_head(dec_out).squeeze(-1),           # (B, N)
            "recon_eta": self.recon_eta_head(dec_out).squeeze(-1),         # (B, N)
            "recon_phi": self.recon_phi_head(dec_out).squeeze(-1),         # (B, N)
            "recon_particle_id": self.recon_particle_id_head(dec_out),     # (B, N, n_types)
            "existence_logit": self.existence_head(dec_out).squeeze(-1),   # (B, N)
        }


def compute_vae_loss(
    recon: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    encoder: ParticleTransformer,
    kl_weight: float = 1.0,
    existence_loss_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Compute VAE ELBO loss.

    Args:
        recon: output dict from VAEParticleModel.forward()
        targets: dict with keys 'pt', 'eta', 'phi', 'particle_id', 'mask'
                 where mask is True for real particles (batch['mask'] convention)
        encoder: ParticleTransformer instance — used to normalize continuous targets
                 to the same scale as the decoder's output heads
        kl_weight: coefficient on KL term (caller handles warmup schedule)
        existence_loss_weight: coefficient on existence BCE loss

    Returns dict with:
        total_loss, recon_pt_loss, recon_eta_loss, recon_phi_loss,
        recon_particle_id_loss, existence_loss, kl_loss
    """
    mu = recon["mu"]
    logvar = recon["logvar"]
    real_mask = targets["mask"]   # (B, S), True = real particle
    device = mu.device

    # Normalize continuous targets to match decoder output scale
    pt_norm, eta_norm, phi_norm = encoder._normalize_features(
        targets["pt"], targets["eta"], targets["phi"]
    )

    if real_mask.sum() > 0:
        recon_pt_loss = F.mse_loss(recon["recon_pt"][real_mask], pt_norm[real_mask])
        recon_eta_loss = F.mse_loss(recon["recon_eta"][real_mask], eta_norm[real_mask])
        recon_phi_loss = F.mse_loss(recon["recon_phi"][real_mask], phi_norm[real_mask])
        recon_particle_id_loss = F.cross_entropy(
            recon["recon_particle_id"][real_mask],
            targets["particle_id"][real_mask],
        )
    else:
        recon_pt_loss = torch.tensor(0.0, device=device)
        recon_eta_loss = torch.tensor(0.0, device=device)
        recon_phi_loss = torch.tensor(0.0, device=device)
        recon_particle_id_loss = torch.tensor(0.0, device=device)

    # Existence loss over all N slots (real particles should output 1, padding 0)
    existence_loss = F.binary_cross_entropy_with_logits(
        recon["existence_logit"], real_mask.float()
    )

    # KL divergence: closed-form KL(q(z|x) || N(0,I))
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    total_loss = (
        recon_pt_loss
        + recon_eta_loss
        + recon_phi_loss
        + recon_particle_id_loss
        + existence_loss_weight * existence_loss
        + kl_weight * kl_loss
    )

    return {
        "total_loss": total_loss,
        "recon_pt_loss": recon_pt_loss,
        "recon_eta_loss": recon_eta_loss,
        "recon_phi_loss": recon_phi_loss,
        "recon_particle_id_loss": recon_particle_id_loss,
        "existence_loss": existence_loss,
        "kl_loss": kl_loss,
    }
