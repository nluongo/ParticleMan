"""Variational Autoencoder for particle physics event generation."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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

    def _denormalize_features(
        self, pt_norm: Tensor, eta_norm: Tensor, phi_norm: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Invert _normalize_features to recover physical units."""
        pt_lo, pt_hi = self.config.pt_range
        eta_lo, eta_hi = self.config.eta_range
        phi_lo, phi_hi = self.config.phi_range

        pt  = pt_norm  * (pt_hi  - pt_lo)  + pt_lo
        eta = (eta_norm + 1) / 2 * (eta_hi - eta_lo) + eta_lo
        phi = (phi_norm + 1) / 2 * (phi_hi - phi_lo) + phi_lo
        return pt, eta, phi

    def generate(
        self,
        n_events: int,
        device: Optional[torch.device] = None,
        existence_threshold: float = 0.5,
    ) -> Dict[str, Tensor]:
        """Generate new particle events by sampling from the prior.

        Args:
            n_events: Number of events to generate.
            device: Device to run on; defaults to the model's current device.
            existence_threshold: Sigmoid threshold for the existence logit above
                which a slot is treated as a real particle.

        Returns dict with keys:
            pt, eta, phi: (B, N) in physical units
            particle_id: (B, N) long, categorical
            existence: (B, N) sigmoid probabilities
            mask: (B, N) bool, True = real particle (existence > threshold)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            z = torch.randn(n_events, self.config.latent_dim, device=device)
            memory = self.latent_to_decoder(z).unsqueeze(1)          # (B, 1, d)
            tgt = self.query_tokens.expand(n_events, -1, -1)          # (B, N, d)
            dec_out = self.decoder(tgt, memory)                        # (B, N, d)

            pt_norm  = self.recon_pt_head(dec_out).squeeze(-1)         # (B, N)
            eta_norm = self.recon_eta_head(dec_out).squeeze(-1)        # (B, N)
            phi_norm = self.recon_phi_head(dec_out).squeeze(-1)        # (B, N)
            particle_id = self.recon_particle_id_head(dec_out).argmax(-1)  # (B, N)
            existence = self.existence_head(dec_out).squeeze(-1).sigmoid()  # (B, N)

            pt, eta, phi = self._denormalize_features(pt_norm, eta_norm, phi_norm)

        return {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "particle_id": particle_id,
            "existence": existence,
            "mask": existence > existence_threshold,
        }

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


def _build_cost_matrix(
    recon_pt: Tensor,
    recon_eta: Tensor,
    recon_phi: Tensor,
    recon_id: Tensor,
    exist: Tensor,
    pt_norm_j: Tensor,
    eta_norm_j: Tensor,
    phi_norm_j: Tensor,
    pid_j: Tensor,
    existence_loss_weight: float,
) -> np.ndarray:
    """Build the (N, N) Hungarian cost matrix for one event (no gradients).

    The first M columns correspond to real particles; the remaining N-M columns
    correspond to null slots. linear_sum_assignment minimises the total cost,
    so assigning a slot to null is cheap when its existence logit is low.

    Args:
        recon_pt/eta/phi: (N,) normalized decoder outputs for this event
        recon_id:         (N, n_types) particle-ID logits
        exist:            (N,) raw existence logits
        pt_norm_j/eta_norm_j/phi_norm_j: (M,) normalized real-particle targets
        pid_j:            (M,) long, real-particle IDs
        existence_loss_weight: scalar weight applied to existence BCE inside cost

    Returns:
        cost: (N, N) float32 numpy array
    """
    N = recon_pt.shape[0]
    M = pt_norm_j.shape[0]

    # Work entirely with detached CPU tensors; no gradients needed for assignment.
    with torch.no_grad():
        rpt = recon_pt.detach().cpu()
        reta = recon_eta.detach().cpu()
        rphi = recon_phi.detach().cpu()
        rid = recon_id.detach().cpu()
        ex = exist.detach().cpu()
        pt_j = pt_norm_j.detach().cpu()
        eta_j = eta_norm_j.detach().cpu()
        phi_j = phi_norm_j.detach().cpu()
        pid = pid_j.detach().cpu()

        # Pairwise squared differences: (N, M)
        pt_cost  = (rpt.unsqueeze(1) - pt_j.unsqueeze(0)).pow(2)
        eta_cost = (reta.unsqueeze(1) - eta_j.unsqueeze(0)).pow(2)
        phi_cost = (rphi.unsqueeze(1) - phi_j.unsqueeze(0)).pow(2)

        # Cross-entropy for each (slot, particle) pair via negative log-prob: (N, M)
        log_probs = F.log_softmax(rid, dim=-1)   # (N, n_types)
        id_cost = -log_probs[:, pid]              # (N, M)

        # Existence BCE for matching to a real particle (target=1): (N,) → broadcast
        exist_real_cost = F.binary_cross_entropy_with_logits(
            ex, torch.ones_like(ex), reduction="none"
        )  # (N,)

        # Existence BCE for matching to null (target=0): (N,)
        exist_null_cost = F.binary_cross_entropy_with_logits(
            ex, torch.zeros_like(ex), reduction="none"
        )  # (N,)

        # Real-particle columns: recon cost + existence cost (target=1)
        real_cols = (
            pt_cost + eta_cost + phi_cost + id_cost
            + existence_loss_weight * exist_real_cost.unsqueeze(1)
        )  # (N, M)

        # Null columns: existence cost (target=0), replicated N-M times
        null_cols = (
            existence_loss_weight * exist_null_cost.unsqueeze(1)
        ).expand(N, N - M)  # (N, N-M)

        cost = torch.cat([real_cols, null_cols], dim=1)  # (N, N)

    return cost.numpy().astype(np.float32)


def compute_vae_loss(
    recon: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    encoder: ParticleTransformer,
    kl_weight: float = 1.0,
    existence_loss_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Compute VAE ELBO loss with Hungarian matching.

    For each event, finds the optimal bijection between N decoder output slots
    and M real particles + (N-M) null slots via the Hungarian algorithm.
    This makes the loss permutation-invariant: swapping two particles in the
    reconstruction incurs zero additional cost if all features are equal.

    Slots matched to a real particle pay reconstruction loss + existence(target=1).
    Slots matched to null pay existence(target=0).

    Args:
        recon: output dict from VAEParticleModel.forward()
        targets: dict with keys 'pt', 'eta', 'phi', 'particle_id', 'mask'
                 where mask is True for real particles (batch['mask'] convention)
        encoder: ParticleTransformer — used to normalize continuous targets to
                 the same scale as the decoder's output heads
        kl_weight: coefficient on KL term (caller handles warmup schedule)
        existence_loss_weight: weight on existence BCE, both in the matching cost
                               and in the final loss

    Returns dict with:
        total_loss, recon_pt_loss, recon_eta_loss, recon_phi_loss,
        recon_particle_id_loss, existence_loss, kl_loss
    """
    mu = recon["mu"]
    logvar = recon["logvar"]
    real_mask = targets["mask"]   # (B, S), True = real particle
    device = mu.device
    B = mu.shape[0]

    # Normalize continuous targets to match decoder output scale
    pt_norm, eta_norm, phi_norm = encoder._normalize_features(
        targets["pt"], targets["eta"], targets["phi"]
    )

    recon_pt_acc = torch.tensor(0.0, device=device)
    recon_eta_acc = torch.tensor(0.0, device=device)
    recon_phi_acc = torch.tensor(0.0, device=device)
    recon_id_acc = torch.tensor(0.0, device=device)
    exist_acc = torch.tensor(0.0, device=device)

    for b in range(B):
        M = int(real_mask[b].sum().item())

        exist_b = recon["existence_logit"][b]   # (N,)
        ones = torch.ones(M, device=device)
        null_count = exist_b.shape[0] - M
        zeros = torch.zeros(null_count, device=device)

        if M == 0:
            # All slots should predict "no particle"
            exist_acc = exist_acc + F.binary_cross_entropy_with_logits(
                exist_b, torch.zeros_like(exist_b)
            )
            continue

        # Build cost matrix and run Hungarian (no gradient)
        cost = _build_cost_matrix(
            recon["recon_pt"][b],
            recon["recon_eta"][b],
            recon["recon_phi"][b],
            recon["recon_particle_id"][b],
            exist_b,
            pt_norm[b, real_mask[b]],
            eta_norm[b, real_mask[b]],
            phi_norm[b, real_mask[b]],
            targets["particle_id"][b, real_mask[b]],
            existence_loss_weight,
        )
        row_ind, col_ind = linear_sum_assignment(cost)

        # Partition into real-particle matches and null matches
        real_sel = col_ind < M
        real_rows = row_ind[real_sel]          # output slot indices → real particles
        real_cols = col_ind[real_sel]          # which real particle each slot maps to
        null_rows = row_ind[~real_sel]         # output slot indices → null

        # Reconstruction losses on matched real pairs (with gradients)
        real_pt_targets = pt_norm[b, real_mask[b]][real_cols]
        real_eta_targets = eta_norm[b, real_mask[b]][real_cols]
        real_phi_targets = phi_norm[b, real_mask[b]][real_cols]
        real_pid_targets = targets["particle_id"][b, real_mask[b]][real_cols]

        recon_pt_acc = recon_pt_acc + F.mse_loss(
            recon["recon_pt"][b][real_rows], real_pt_targets
        )
        recon_eta_acc = recon_eta_acc + F.mse_loss(
            recon["recon_eta"][b][real_rows], real_eta_targets
        )
        recon_phi_acc = recon_phi_acc + F.mse_loss(
            recon["recon_phi"][b][real_rows], real_phi_targets
        )
        recon_id_acc = recon_id_acc + F.cross_entropy(
            recon["recon_particle_id"][b][real_rows], real_pid_targets
        )

        # Existence losses on matched slots (with gradients)
        exist_real = F.binary_cross_entropy_with_logits(exist_b[real_rows], ones)
        if null_rows.size > 0:
            exist_null = F.binary_cross_entropy_with_logits(exist_b[null_rows], zeros)
        else:
            exist_null = torch.tensor(0.0, device=device)
        exist_acc = exist_acc + (exist_real + exist_null) / 2

    # Average over batch
    recon_pt_loss = recon_pt_acc / B
    recon_eta_loss = recon_eta_acc / B
    recon_phi_loss = recon_phi_acc / B
    recon_particle_id_loss = recon_id_acc / B
    existence_loss = exist_acc / B

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
