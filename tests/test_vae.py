"""Tests for the VAE particle model."""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from particleman.models.vae import VAEConfig, VAEParticleModel, compute_vae_loss
from particleman.models.particle_transformer import EmbeddingType, PhiEncoding
from particleman.training.trainer import ParticleTrainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_vae_dataloader(batch_size: int = 2, seq_len: int = 20, n_particle_types: int = 5):
    """Build a minimal DataLoader compatible with ParticleTrainer VAE mode."""
    B, S = 4, seq_len
    # last slot is always padding so we exercise the masking logic
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[:, -5:] = False

    ds = TensorDataset(
        torch.rand(B, S) * 100,
        torch.randn(B, S) * 2,
        torch.rand(B, S) * 2 * 3.14159 - 3.14159,
        torch.randint(0, n_particle_types - 1, (B, S)),
        mask,
        torch.randint(0, 3, (B,)),
    )

    def collate(items):
        pt, eta, phi, pid, mask_, lbl = zip(*items)
        return {
            'pt': torch.stack(pt),
            'eta': torch.stack(eta),
            'phi': torch.stack(phi),
            'particle_id': torch.stack(pid),
            'mask': torch.stack(mask_),
            'event_label': torch.stack(lbl),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ---------------------------------------------------------------------------
# VAEConfig tests
# ---------------------------------------------------------------------------

class TestVAEConfig:
    def test_default_config(self) -> None:
        cfg = VAEConfig()
        assert cfg.d_model == 256
        assert cfg.n_heads == 8
        assert cfg.latent_dim == 64
        assert cfg.decoder_layers == 2
        assert cfg.decoder_heads == 4
        assert cfg.n_decoder_queries == 0
        assert cfg.kl_weight == 1.0
        assert cfg.kl_warmup_steps == 0
        assert cfg.existence_loss_weight == 1.0

    def test_to_particle_config_fields(self) -> None:
        cfg = VAEConfig(d_model=128, n_heads=4, n_layers=3, n_particle_types=10)
        pc = cfg.to_particle_config()
        assert pc.d_model == 128
        assert pc.n_heads == 4
        assert pc.n_layers == 3
        assert pc.n_particle_types == 10
        # VAE-specific fields must not appear on ParticleConfig
        assert not hasattr(pc, 'latent_dim')

    def test_custom_vae_fields(self) -> None:
        cfg = VAEConfig(latent_dim=32, decoder_layers=4, kl_weight=0.5)
        assert cfg.latent_dim == 32
        assert cfg.decoder_layers == 4
        assert cfg.kl_weight == 0.5


# ---------------------------------------------------------------------------
# VAEParticleModel tests
# ---------------------------------------------------------------------------

class TestVAEParticleModel:
    @pytest.fixture
    def cfg(self) -> VAEConfig:
        return VAEConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_particles=20,
            n_particle_types=5,
            id_embed_dim=16,
            embed_hidden_dim=64,
            latent_dim=8,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ff_dim=128,
            kl_weight=1.0,
        )

    @pytest.fixture
    def model(self, cfg: VAEConfig) -> VAEParticleModel:
        return VAEParticleModel(cfg)

    @pytest.fixture
    def batch(self, cfg: VAEConfig) -> dict:
        B, S = 4, cfg.max_particles
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[:, -5:] = False
        return {
            'pt': torch.rand(B, S) * 100,
            'eta': torch.randn(B, S) * 2,
            'phi': torch.rand(B, S) * 2 * 3.14159 - 3.14159,
            'particle_id': torch.randint(0, cfg.n_particle_types - 1, (B, S)),
            'mask': mask,
        }

    def test_initialization(self, model: VAEParticleModel, cfg: VAEConfig) -> None:
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'mu_head')
        assert hasattr(model, 'logvar_head')
        assert hasattr(model, 'latent_to_decoder')
        assert hasattr(model, 'query_tokens')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'recon_pt_head')
        assert hasattr(model, 'recon_eta_head')
        assert hasattr(model, 'recon_phi_head')
        assert hasattr(model, 'recon_particle_id_head')
        assert hasattr(model, 'existence_head')
        # query_tokens shape: (1, N, d_model)
        N = cfg.max_particles
        assert model.query_tokens.shape == (1, N, cfg.d_model)

    def test_forward_output_keys(self, model: VAEParticleModel, batch: dict) -> None:
        model.eval()
        with torch.no_grad():
            out = model(
                batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                padding_mask=~batch['mask'],
            )
        expected = {'mu', 'logvar', 'z', 'recon_pt', 'recon_eta', 'recon_phi',
                    'recon_particle_id', 'existence_logit'}
        assert expected == set(out.keys())

    def test_forward_output_shapes(self, model: VAEParticleModel, batch: dict, cfg: VAEConfig) -> None:
        B = batch['pt'].shape[0]
        N = cfg.max_particles
        model.eval()
        with torch.no_grad():
            out = model(
                batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                padding_mask=~batch['mask'],
            )
        assert out['mu'].shape == (B, cfg.latent_dim)
        assert out['logvar'].shape == (B, cfg.latent_dim)
        assert out['z'].shape == (B, cfg.latent_dim)
        assert out['recon_pt'].shape == (B, N)
        assert out['recon_eta'].shape == (B, N)
        assert out['recon_phi'].shape == (B, N)
        assert out['recon_particle_id'].shape == (B, N, cfg.n_particle_types)
        assert out['existence_logit'].shape == (B, N)

    def test_forward_without_padding_mask(self, model: VAEParticleModel, batch: dict, cfg: VAEConfig) -> None:
        B = batch['pt'].shape[0]
        N = cfg.max_particles
        model.eval()
        with torch.no_grad():
            out = model(batch['pt'], batch['eta'], batch['phi'], batch['particle_id'])
        assert out['recon_pt'].shape == (B, N)

    def test_eval_mode_deterministic(self, model: VAEParticleModel, batch: dict) -> None:
        model.eval()
        with torch.no_grad():
            out1 = model(batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                         padding_mask=~batch['mask'])
            out2 = model(batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                         padding_mask=~batch['mask'])
        assert torch.equal(out1['z'], out2['z']), "eval mode must be deterministic"

    def test_train_mode_stochastic(self, model: VAEParticleModel, batch: dict) -> None:
        model.train()
        out1 = model(batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                     padding_mask=~batch['mask'])
        out2 = model(batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                     padding_mask=~batch['mask'])
        # With overwhelmingly high probability two independent samples differ
        assert not torch.equal(out1['z'], out2['z']), "training mode should sample stochastically"

    def test_n_decoder_queries_override(self, cfg: VAEConfig) -> None:
        cfg2 = VAEConfig(**{**cfg.__dict__, 'n_decoder_queries': 10})
        m = VAEParticleModel(cfg2)
        assert m.query_tokens.shape == (1, 10, cfg2.d_model)
        B, S = 2, cfg2.max_particles
        with torch.no_grad():
            out = m(
                torch.rand(B, S) * 100,
                torch.randn(B, S),
                torch.zeros(B, S),
                torch.zeros(B, S, dtype=torch.long),
            )
        assert out['recon_pt'].shape == (B, 10)


# ---------------------------------------------------------------------------
# compute_vae_loss tests
# ---------------------------------------------------------------------------

class TestComputeVAELoss:
    @pytest.fixture
    def cfg(self) -> VAEConfig:
        return VAEConfig(
            d_model=64, n_heads=4, n_layers=1, d_ff=128,
            max_particles=20, n_particle_types=5,
            id_embed_dim=16, embed_hidden_dim=64,
            latent_dim=8, decoder_layers=1, decoder_heads=4, decoder_ff_dim=128,
        )

    @pytest.fixture
    def model_and_batch(self, cfg: VAEConfig):
        model = VAEParticleModel(cfg)
        B, S = 4, cfg.max_particles
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[:, -5:] = False
        batch = {
            'pt': torch.rand(B, S) * 100,
            'eta': torch.randn(B, S) * 2,
            'phi': torch.rand(B, S) * 2 * 3.14159 - 3.14159,
            'particle_id': torch.randint(0, cfg.n_particle_types - 1, (B, S)),
            'mask': mask,
        }
        model.eval()
        with torch.no_grad():
            recon = model(batch['pt'], batch['eta'], batch['phi'], batch['particle_id'],
                          padding_mask=~mask)
        return model, recon, batch

    def test_loss_keys(self, model_and_batch) -> None:
        model, recon, batch = model_and_batch
        losses = compute_vae_loss(recon, batch, model.encoder)
        expected = {'total_loss', 'recon_pt_loss', 'recon_eta_loss', 'recon_phi_loss',
                    'recon_particle_id_loss', 'existence_loss', 'kl_loss'}
        assert expected == set(losses.keys())

    def test_losses_are_scalars(self, model_and_batch) -> None:
        model, recon, batch = model_and_batch
        losses = compute_vae_loss(recon, batch, model.encoder)
        for k, v in losses.items():
            assert v.shape == (), f"{k} should be a scalar tensor"

    def test_losses_nonnegative(self, model_and_batch) -> None:
        model, recon, batch = model_and_batch
        losses = compute_vae_loss(recon, batch, model.encoder)
        for k, v in losses.items():
            assert v.item() >= 0.0, f"{k} should be non-negative"

    def test_kl_weight_zero_excludes_kl(self, model_and_batch) -> None:
        model, recon, batch = model_and_batch
        losses_kl = compute_vae_loss(recon, batch, model.encoder, kl_weight=1.0)
        losses_no_kl = compute_vae_loss(recon, batch, model.encoder, kl_weight=0.0)
        # total with kl_weight=0 should be total minus KL contribution
        expected = losses_kl['total_loss'] - losses_kl['kl_loss']
        assert torch.isclose(losses_no_kl['total_loss'], expected, atol=1e-5)

    def test_kl_loss_zero_for_standard_normal(self, model_and_batch) -> None:
        model, recon, batch = model_and_batch
        # KL(N(0,I) || N(0,I)) == 0
        zero_recon = dict(recon)
        B = batch['pt'].shape[0]
        latent_dim = model.config.latent_dim
        zero_recon['mu'] = torch.zeros(B, latent_dim)
        zero_recon['logvar'] = torch.zeros(B, latent_dim)
        losses = compute_vae_loss(zero_recon, batch, model.encoder)
        assert torch.isclose(losses['kl_loss'], torch.tensor(0.0), atol=1e-5)

    def test_existence_loss_finite(self, model_and_batch) -> None:
        model, recon, batch = model_and_batch
        losses = compute_vae_loss(recon, batch, model.encoder)
        assert torch.isfinite(losses['existence_loss'])

    def test_permutation_invariance(self, model_and_batch) -> None:
        """Permuting real particles in the targets should not change the loss."""
        model, recon, batch = model_and_batch
        B, S = batch['pt'].shape
        # All events have the same mask; find M (real particle count per event)
        M = int(batch['mask'][0].sum().item())

        perm = torch.randperm(M)
        shuffled = {k: v.clone() for k, v in batch.items()}
        shuffled['pt'][:, :M]           = batch['pt'][:, perm]
        shuffled['eta'][:, :M]          = batch['eta'][:, perm]
        shuffled['phi'][:, :M]          = batch['phi'][:, perm]
        shuffled['particle_id'][:, :M]  = batch['particle_id'][:, perm]
        # mask stays the same (same number of real particles, same positions)

        loss_orig = compute_vae_loss(recon, batch, model.encoder, kl_weight=0.0)
        loss_perm = compute_vae_loss(recon, shuffled, model.encoder, kl_weight=0.0)
        assert torch.isclose(
            loss_orig['total_loss'], loss_perm['total_loss'], atol=1e-4
        ), f"Loss changed under permutation: {loss_orig['total_loss'].item():.6f} vs {loss_perm['total_loss'].item():.6f}"

    def test_existence_false_positive_penalized(self, cfg: VAEConfig, model_and_batch) -> None:
        """Slots matched to null should be penalized when existence logit is high.

        With M=15 real and 5 null slots:
          loud (+10 everywhere): null slots pay BCE(+10, 0)≈10 each → 5 false positives penalized
          perfect (real=+10, null=-10): no false positives, no false negatives → lower loss
        """
        model, recon, batch = model_and_batch
        B, N = recon['existence_logit'].shape
        M = int(batch['mask'][0].sum().item())  # real particles per event

        # Perfect existence prediction: real slots +10, null slots -10
        perfect = torch.full((B, N), -10.0)
        perfect[:, :M] = 10.0
        perfect_recon = dict(recon)
        perfect_recon['existence_logit'] = perfect
        losses_perfect = compute_vae_loss(perfect_recon, batch, model.encoder, kl_weight=0.0)

        # False-positive prediction: all slots predict +10 (all "real")
        loud_recon = dict(recon)
        loud_recon['existence_logit'] = torch.full_like(recon['existence_logit'], 10.0)
        losses_loud = compute_vae_loss(loud_recon, batch, model.encoder, kl_weight=0.0)

        # False positives (null slots predicting "real") inflate existence loss
        assert losses_loud['existence_loss'].item() > losses_perfect['existence_loss'].item()

    def test_existence_false_negative_penalized(self, model_and_batch) -> None:
        """Slots matched to real particles should be penalized when existence logit is low."""
        model, recon, batch = model_and_batch
        quiet_recon = dict(recon)
        quiet_recon['existence_logit'] = torch.full_like(recon['existence_logit'], -10.0)
        losses = compute_vae_loss(quiet_recon, batch, model.encoder, kl_weight=0.0)
        # All real slots predict 'no particle' → existence loss should be large
        assert losses['existence_loss'].item() > 0.5


# ---------------------------------------------------------------------------
# Trainer VAE mode tests
# ---------------------------------------------------------------------------

class TestVAETrainerStep:
    @pytest.fixture
    def cfg(self) -> VAEConfig:
        return VAEConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            max_particles=20, n_particle_types=5,
            id_embed_dim=16, embed_hidden_dim=64,
            latent_dim=8, decoder_layers=1, decoder_heads=4, decoder_ff_dim=128,
            kl_weight=1.0, kl_warmup_steps=0,
        )

    @pytest.fixture
    def trainer(self, cfg: VAEConfig) -> ParticleTrainer:
        model = VAEParticleModel(cfg)
        dl = _make_vae_dataloader(seq_len=cfg.max_particles, n_particle_types=cfg.n_particle_types)
        return ParticleTrainer(model, dl, device='cpu', num_epochs=1, mode='vae')

    def test_train_step_returns_dict(self, trainer: ParticleTrainer) -> None:
        batch = next(iter(trainer.train_dataloader))
        losses = trainer.train_step(batch)
        assert losses is not None
        assert isinstance(losses, dict)

    def test_train_step_loss_keys(self, trainer: ParticleTrainer) -> None:
        batch = next(iter(trainer.train_dataloader))
        losses = trainer.train_step(batch)
        assert losses is not None
        for key in ('total_loss', 'kl_loss', 'existence_loss', 'recon_pt_loss',
                    'recon_eta_loss', 'recon_phi_loss', 'recon_particle_id_loss', 'kl_weight'):
            assert key in losses, f"Expected key '{key}' in losses"

    def test_train_step_losses_finite(self, trainer: ParticleTrainer) -> None:
        batch = next(iter(trainer.train_dataloader))
        losses = trainer.train_step(batch)
        assert losses is not None
        for k, v in losses.items():
            assert torch.isfinite(torch.tensor(v)), f"{k} is not finite"

    def test_kl_warmup_increments_step_count(self, cfg: VAEConfig) -> None:
        cfg2 = VAEConfig(**{**cfg.__dict__, 'kl_warmup_steps': 10})
        model = VAEParticleModel(cfg2)
        dl = _make_vae_dataloader(seq_len=cfg2.max_particles, n_particle_types=cfg2.n_particle_types)
        trainer = ParticleTrainer(model, dl, device='cpu', num_epochs=1, mode='vae')
        assert trainer._vae_step_count == 0
        batch = next(iter(dl))
        trainer.train_step(batch)
        assert trainer._vae_step_count == 1
        trainer.train_step(batch)
        assert trainer._vae_step_count == 2

    def test_kl_warmup_weight_ramps(self, cfg: VAEConfig) -> None:
        warmup = 4
        cfg2 = VAEConfig(**{**cfg.__dict__, 'kl_warmup_steps': warmup, 'kl_weight': 1.0})
        model = VAEParticleModel(cfg2)
        dl = _make_vae_dataloader(seq_len=cfg2.max_particles, n_particle_types=cfg2.n_particle_types)
        trainer = ParticleTrainer(model, dl, device='cpu', num_epochs=1, mode='vae')
        batch = next(iter(dl))
        weights = []
        for _ in range(warmup + 1):
            losses = trainer.train_step(batch)
            assert losses is not None
            weights.append(losses['kl_weight'])
        # weights should be monotonically increasing and cap at 1.0
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]
        assert weights[-1] == pytest.approx(1.0, abs=1e-5)

    def test_invalid_mode_raises(self, cfg: VAEConfig) -> None:
        model = VAEParticleModel(cfg)
        dl = _make_vae_dataloader(seq_len=cfg.max_particles, n_particle_types=cfg.n_particle_types)
        with pytest.raises(ValueError, match="mode must be"):
            ParticleTrainer(model, dl, device='cpu', mode='bad')
