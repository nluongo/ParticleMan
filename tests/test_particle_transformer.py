"""Test the Particle Transformer model."""

import torch
import pytest

from particleman.models.particle_transformer import (
    ParticleTransformer,
    ParticleConfig,
    EmbeddingType,
    ConcatEmbedding,
    JointEmbedding,
)


class TestParticleConfig:
    """Test the ParticleConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ParticleConfig()
        
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 1024
        assert config.dropout == 0.1
        assert config.max_particles == 100
        assert config.n_particle_types == 13
        assert config.mask_prob == 0.15
        assert config.mask_continuous_std == 0.1
        assert config.pt_range == (0.0, 1000.0)
        assert config.eta_range == (-5.0, 5.0)
        assert config.phi_range == (-3.14159, 3.14159)
        assert config.embedding_type == EmbeddingType.JOINT
    
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ParticleConfig(
            d_model=128,
            n_heads=4,
            n_layers=3,
            max_particles=50
        )
        
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 3
        assert config.max_particles == 50
        # Check that other values remain default
        assert config.dropout == 0.1
        assert config.n_particle_types == 13
    
    def test_concat_embedding_config(self) -> None:
        """Test configuration with concat embedding."""
        config = ParticleConfig(
            embedding_type=EmbeddingType.CONCAT
        )
        assert config.embedding_type == EmbeddingType.CONCAT
    
    def test_joint_embedding_config(self) -> None:
        """Test configuration with joint embedding."""
        config = ParticleConfig(
            embedding_type=EmbeddingType.JOINT,
            id_embed_dim=64,
            embed_hidden_dim=256,
        )
        assert config.embedding_type == EmbeddingType.JOINT
        assert config.id_embed_dim == 64
        assert config.embed_hidden_dim == 256


class TestParticleTransformer:
    """Test the ParticleTransformer model."""
    
    @pytest.fixture
    def config(self) -> ParticleConfig:
        """Create a test configuration with joint embedding (default)."""
        return ParticleConfig(
            d_model=64,  # Small model for testing
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_particles=20,
            n_particle_types=5,
            embedding_type=EmbeddingType.JOINT,
        )
    
    @pytest.fixture
    def concat_config(self) -> ParticleConfig:
        """Create a test configuration with concat embedding."""
        return ParticleConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_particles=20,
            n_particle_types=5,
            embedding_type=EmbeddingType.CONCAT,
        )
    
    @pytest.fixture
    def model(self, config: ParticleConfig) -> ParticleTransformer:
        """Create a test model with joint embedding."""
        return ParticleTransformer(config)
    
    @pytest.fixture
    def concat_model(self, concat_config: ParticleConfig) -> ParticleTransformer:
        """Create a test model with concat embedding."""
        return ParticleTransformer(concat_config)
    
    @pytest.fixture
    def sample_data(self, config: ParticleConfig) -> dict:
        """Create sample particle data."""
        batch_size = 2
        seq_len = config.max_particles
        
        return {
            'pt': torch.rand(batch_size, seq_len) * 100,  # 0-100 GeV
            'eta': torch.randn(batch_size, seq_len) * 2,   # -4 to 4
            'phi': torch.rand(batch_size, seq_len) * 2 * 3.14159 - 3.14159,  # -pi to pi
            'particle_id': torch.randint(0, config.n_particle_types - 1, (batch_size, seq_len))
        }
    
    def test_model_initialization_joint(self, model: ParticleTransformer) -> None:
        """Test that the model with joint embedding initializes correctly."""
        assert isinstance(model, ParticleTransformer)
        assert hasattr(model, 'config')
        assert hasattr(model, 'embedding')
        assert isinstance(model.embedding, JointEmbedding)
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'pt_head')
        assert hasattr(model, 'eta_head')
        assert hasattr(model, 'phi_head')
        assert hasattr(model, 'particle_id_head')
    
    def test_model_initialization_concat(self, concat_model: ParticleTransformer) -> None:
        """Test that the model with concat embedding initializes correctly."""
        assert isinstance(concat_model, ParticleTransformer)
        assert hasattr(concat_model, 'embedding')
        assert isinstance(concat_model.embedding, ConcatEmbedding)
    
    def test_forward_pass(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test the forward pass with joint embedding."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        with torch.no_grad():
            predictions = model(pt, eta, phi, particle_id)
        
        # Check output structure
        assert isinstance(predictions, dict)
        assert 'pt' in predictions
        assert 'eta' in predictions
        assert 'phi' in predictions
        assert 'particle_id' in predictions
        assert 'hidden_states' in predictions
        
        # Check output shapes
        batch_size, seq_len = pt.shape
        assert predictions['pt'].shape == (batch_size, seq_len)
        assert predictions['eta'].shape == (batch_size, seq_len)
        assert predictions['phi'].shape == (batch_size, seq_len)
        assert predictions['particle_id'].shape == (batch_size, seq_len, model.config.n_particle_types)
        assert predictions['hidden_states'].shape == (batch_size, seq_len, model.config.d_model)
    
    def test_forward_pass_with_padding_mask(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test the forward pass with padding mask."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        # Create padding mask
        batch_size, seq_len = pt.shape
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -5:] = True  # Last 5 positions are padding
        
        with torch.no_grad():
            predictions = model(pt, eta, phi, particle_id, padding_mask=padding_mask)
        
        # Check output shapes are unchanged
        assert predictions['pt'].shape == (batch_size, seq_len)
        assert predictions['hidden_states'].shape == (batch_size, seq_len, model.config.d_model)
    
    def test_forward_pass_concat(self, concat_model: ParticleTransformer, sample_data: dict) -> None:
        """Test the forward pass with concat embedding."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        with torch.no_grad():
            predictions = concat_model(pt, eta, phi, particle_id)
        
        # Check output shapes
        batch_size, seq_len = pt.shape
        assert predictions['pt'].shape == (batch_size, seq_len)
        assert predictions['hidden_states'].shape == (batch_size, seq_len, concat_model.config.d_model)
    
    def test_both_embeddings_produce_same_output_shape(self, model: ParticleTransformer, concat_model: ParticleTransformer, sample_data: dict) -> None:
        """Test that both embedding types produce the same output shapes."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        with torch.no_grad():
            joint_out = model(pt, eta, phi, particle_id)
            concat_out = concat_model(pt, eta, phi, particle_id)
        
        for key in ['pt', 'eta', 'phi', 'particle_id', 'hidden_states']:
            assert joint_out[key].shape == concat_out[key].shape, f"Shape mismatch for {key}"
    
    def test_create_masks(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test the mask creation functionality."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        masked_inputs, mask_targets = model.create_masks(pt, eta, phi, particle_id, mask_prob=0.2)
        
        # Check output structure
        assert isinstance(masked_inputs, dict)
        assert isinstance(mask_targets, dict)
        
        # Check masked inputs
        assert 'pt' in masked_inputs
        assert 'eta' in masked_inputs
        assert 'phi' in masked_inputs
        assert 'particle_id' in masked_inputs
        
        # Check mask targets
        assert 'pt' in mask_targets
        assert 'eta' in mask_targets
        assert 'phi' in mask_targets
        assert 'particle_id' in mask_targets
        assert 'mask' in mask_targets
        
        # Check shapes
        batch_size, seq_len = pt.shape
        for key in ['pt', 'eta', 'phi', 'particle_id']:
            assert masked_inputs[key].shape == (batch_size, seq_len)
            assert mask_targets[key].shape == (batch_size, seq_len)
        
        assert mask_targets['mask'].shape == (batch_size, seq_len)
        assert mask_targets['mask'].dtype == torch.bool
        
        # Check that approximately 20% of positions are masked
        mask_ratio = mask_targets['mask'].float().mean()
        assert 0.1 < mask_ratio < 0.3  # Allow some variance due to randomness
    
    def test_create_masks_with_padding(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test that masking respects padding."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        # Create a padding mask: last 5 positions are padding
        batch_size, seq_len = pt.shape
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -5:] = True  # Last 5 are padding
        
        masked_inputs, mask_targets = model.create_masks(
            pt, eta, phi, particle_id, 
            padding_mask=padding_mask,
            mask_prob=0.5  # High prob to make test reliable
        )
        
        # Check that padding positions are never in the prediction mask
        prediction_mask = mask_targets['mask']
        assert not prediction_mask[:, -5:].any(), "Padding positions should not be masked for prediction"
    
    def test_compute_loss(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test the loss computation."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        # Create masks and predictions
        masked_inputs, mask_targets = model.create_masks(pt, eta, phi, particle_id)
        
        with torch.no_grad():
            predictions = model(
                masked_inputs['pt'],
                masked_inputs['eta'],
                masked_inputs['phi'],
                masked_inputs['particle_id']
            )
        
        losses = model.compute_loss(predictions, mask_targets)
        
        # Check loss structure
        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert 'pt_loss' in losses
        assert 'eta_loss' in losses
        assert 'phi_loss' in losses
        assert 'particle_id_loss' in losses
        
        # Check that losses are tensors
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.dim() == 0  # Scalar tensor
            assert loss_value.item() >= 0  # Losses should be non-negative
    
    def test_zero_mask_loss(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test loss computation when no positions are masked."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        # Create targets with no masked positions
        mask_targets = {
            'pt': pt,
            'eta': eta,
            'phi': phi,
            'particle_id': particle_id,
            'mask': torch.zeros_like(pt, dtype=torch.bool)
        }
        
        with torch.no_grad():
            predictions = model(pt, eta, phi, particle_id)
        
        losses = model.compute_loss(predictions, mask_targets)
        
        # All losses should be zero when no positions are masked
        for loss_name, loss_value in losses.items():
            assert loss_value.item() == 0.0
    
    def test_model_parameters_require_grad(self, model: ParticleTransformer) -> None:
        """Test that model parameters require gradients."""
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require gradients"
    
    def test_model_training_mode(self, model: ParticleTransformer) -> None:
        """Test that model can switch between training and evaluation modes."""
        # Test training mode
        model.train()
        assert model.training
        
        # Test evaluation mode
        model.eval()
        assert not model.training
    
    def test_different_mask_probabilities(self, model: ParticleTransformer, sample_data: dict) -> None:
        """Test masking with different probabilities."""
        pt = sample_data['pt']
        eta = sample_data['eta']
        phi = sample_data['phi']
        particle_id = sample_data['particle_id']
        
        # Test different mask probabilities
        mask_probs = [0.1, 0.5, 0.9]
        
        for mask_prob in mask_probs:
            masked_inputs, mask_targets = model.create_masks(pt, eta, phi, particle_id, mask_prob=mask_prob)
            actual_mask_ratio = mask_targets['mask'].float().mean().item()
            
            # Allow 10% tolerance for randomness
            assert abs(actual_mask_ratio - mask_prob) < 0.1, f"Expected ~{mask_prob}, got {actual_mask_ratio}"


class TestEmbeddingModules:
    """Test the embedding modules directly."""
    
    @pytest.fixture
    def config(self) -> ParticleConfig:
        """Create a test configuration."""
        return ParticleConfig(
            d_model=64,
            n_particle_types=5,
            id_embed_dim=16,
            embed_hidden_dim=32,
        )
    
    @pytest.fixture
    def sample_normalized_data(self, config: ParticleConfig) -> dict:
        """Create sample normalized particle data."""
        batch_size = 2
        seq_len = 10
        
        return {
            'pt_norm': torch.rand(batch_size, seq_len),  # [0, 1]
            'eta_norm': torch.rand(batch_size, seq_len) * 2 - 1,  # [-1, 1]
            'phi_norm': torch.rand(batch_size, seq_len) * 2 - 1,  # [-1, 1]
            'particle_id': torch.randint(0, config.n_particle_types, (batch_size, seq_len))
        }
    
    def test_concat_embedding_output_shape(self, config: ParticleConfig, sample_normalized_data: dict) -> None:
        """Test ConcatEmbedding output shape."""
        embed = ConcatEmbedding(config)
        
        output = embed(
            sample_normalized_data['pt_norm'],
            sample_normalized_data['eta_norm'],
            sample_normalized_data['phi_norm'],
            sample_normalized_data['particle_id']
        )
        
        batch_size, seq_len = sample_normalized_data['pt_norm'].shape
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_joint_embedding_output_shape(self, config: ParticleConfig, sample_normalized_data: dict) -> None:
        """Test JointEmbedding output shape."""
        embed = JointEmbedding(config)
        
        output = embed(
            sample_normalized_data['pt_norm'],
            sample_normalized_data['eta_norm'],
            sample_normalized_data['phi_norm'],
            sample_normalized_data['particle_id']
        )
        
        batch_size, seq_len = sample_normalized_data['pt_norm'].shape
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_joint_embedding_learns_interactions(self, config: ParticleConfig) -> None:
        """Test that JointEmbedding can learn feature interactions.
        
        The joint embedding should produce different outputs for different
        combinations of features, even if individual features are the same.
        """
        embed = JointEmbedding(config)
        
        # Two particles with same pt but different eta
        pt1 = torch.tensor([[0.5]])
        eta1 = torch.tensor([[0.0]])
        phi1 = torch.tensor([[0.0]])
        pid1 = torch.tensor([[0]])
        
        pt2 = torch.tensor([[0.5]])  # Same pt
        eta2 = torch.tensor([[1.0]])  # Different eta
        phi2 = torch.tensor([[0.0]])
        pid2 = torch.tensor([[0]])
        
        out1 = embed(pt1, eta1, phi1, pid1)
        out2 = embed(pt2, eta2, phi2, pid2)
        
        # Outputs should be different
        assert not torch.allclose(out1, out2)
