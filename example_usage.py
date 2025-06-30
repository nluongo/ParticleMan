"""
Example usage of the ParticleTransformer model for pre-training.

This script demonstrates how to:
1. Create synthetic particle data
2. Initialize the model
3. Run a simple training loop
4. Save and load the model
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from src.particleman.models.particle_transformer import ParticleTransformer, ParticleConfig
from src.particleman.training.trainer import ParticleTrainer


class SyntheticParticleDataset(Dataset):
    """Synthetic particle dataset for demonstration."""
    
    def __init__(self, num_samples: int = 1000, max_particles: int = 50) -> None:
        """
        Generate synthetic particle physics data.
        
        Args:
            num_samples: Number of events to generate
            max_particles: Maximum number of particles per event
        """
        self.num_samples = num_samples
        self.max_particles = max_particles
        
        # Generate synthetic data
        self.data = []
        for _ in range(num_samples):
            # Random number of particles per event
            n_particles = np.random.randint(5, max_particles + 1)
            
            # Generate particle features (following realistic physics distributions)
            pt = np.random.exponential(50, n_particles)  # Exponential distribution for pt (always positive!)
            eta = np.random.normal(0, 2.5, n_particles)  # Gaussian distribution for eta (can be +/-)
            phi = np.random.uniform(-np.pi, np.pi, n_particles)  # Uniform distribution for phi (can be +/-)
            particle_id = np.random.randint(0, 12, n_particles)  # Random particle types
            
            # Pad to max_particles
            if n_particles < max_particles:
                padding_size = max_particles - n_particles
                pt = np.pad(pt, (0, padding_size), constant_values=0)
                eta = np.pad(eta, (0, padding_size), constant_values=0)
                phi = np.pad(phi, (0, padding_size), constant_values=0)
                particle_id = np.pad(particle_id, (0, padding_size), constant_values=12)  # Use 12 as padding token
            
            self.data.append({
                'pt': torch.tensor(pt, dtype=torch.float32),
                'eta': torch.tensor(eta, dtype=torch.float32),
                'phi': torch.tensor(phi, dtype=torch.float32),
                'particle_id': torch.tensor(particle_id, dtype=torch.long),
                'n_particles': n_particles
            })
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def main() -> None:
    """Main function demonstrating the usage."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🚀 ParticleMan Demo: Pre-training Foundation Models for Particle Physics")
    print("=" * 70)
    
    # 1. Create model configuration
    print("1. Creating model configuration...")
    config = ParticleConfig(
        d_model=128,        # Smaller model for demo
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        max_particles=50,
        n_particle_types=13,  # 12 particle types + 1 padding token
        mask_prob=0.15,
        mask_continuous_std=0.1
    )
    print(f"   Model dimension: {config.d_model}")
    print(f"   Number of heads: {config.n_heads}")
    print(f"   Number of layers: {config.n_layers}")
    print(f"   Max particles per event: {config.max_particles}")
    
    # 2. Initialize model
    print("\n2. Initializing ParticleTransformer model...")
    model = ParticleTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # 3. Create synthetic dataset
    print("\n3. Creating synthetic particle dataset...")
    train_dataset = SyntheticParticleDataset(num_samples=800, max_particles=config.max_particles)
    val_dataset = SyntheticParticleDataset(num_samples=200, max_particles=config.max_particles)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # 4. Demo forward pass
    print("\n4. Running demo forward pass...")
    sample_batch = next(iter(train_loader))
    
    # Show original data
    print(f"   Batch shape - pt: {sample_batch['pt'].shape}")
    print(f"   Sample pt values: {sample_batch['pt'][0][:5].tolist()}")
    print(f"   Sample eta values: {sample_batch['eta'][0][:5].tolist()}")
    print(f"   Sample phi values: {sample_batch['phi'][0][:5].tolist()}")
    print(f"   Sample particle_id: {sample_batch['particle_id'][0][:5].tolist()}")
    
    # Create masks and run forward pass
    with torch.no_grad():
        masked_inputs, mask_targets = model.create_masks(
            sample_batch['pt'],
            sample_batch['eta'],
            sample_batch['phi'],
            sample_batch['particle_id']
        )
        
        predictions = model(
            masked_inputs['pt'],
            masked_inputs['eta'],
            masked_inputs['phi'],
            masked_inputs['particle_id']
        )
        
        losses = model.compute_loss(predictions, mask_targets)
        
        print(f"   Mask ratio: {mask_targets['mask'].float().mean():.2%}")
        print(f"   Initial losses:")
        for k, v in losses.items():
            print(f"     {k}: {v.item():.4f}")
    
    # 5. Training demo
    print("\n5. Starting training demo (2 epochs)...")
    
    # Create trainer
    trainer = ParticleTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr=1e-3,
        weight_decay=0.01,
        log_interval=10,
        save_dir=Path("checkpoints")
    )
    
    # Train for a few epochs
    trainer.train(num_epochs=2)
    
    # 6. Demo inference after training
    print("\n6. Running inference after training...")
    model.eval()
    with torch.no_grad():
        # Use same sample batch
        masked_inputs, mask_targets = model.create_masks(
            sample_batch['pt'],
            sample_batch['eta'],
            sample_batch['phi'],
            sample_batch['particle_id']
        )
        
        predictions = model(
            masked_inputs['pt'],
            masked_inputs['eta'],
            masked_inputs['phi'],
            masked_inputs['particle_id']
        )
        
        losses = model.compute_loss(predictions, mask_targets)
        
        print(f"   Final losses:")
        for k, v in losses.items():
            print(f"     {k}: {v.item():.4f}")
        
        # Show some predictions vs targets for masked positions
        mask = mask_targets['mask'][0]  # First sample in batch
        if mask.sum() > 0:
            masked_indices = mask.nonzero().flatten()[:3]  # Show first 3 masked positions
            print(f"\n   Sample predictions vs targets (first 3 masked positions):")
            for i, idx in enumerate(masked_indices):
                print(f"     Position {idx.item()}:")
                print(f"       pt: pred={predictions['pt'][0, idx].item():.3f}, target={mask_targets['pt'][0, idx].item():.3f}")
                print(f"       eta: pred={predictions['eta'][0, idx].item():.3f}, target={mask_targets['eta'][0, idx].item():.3f}")
                print(f"       phi: pred={predictions['phi'][0, idx].item():.3f}, target={mask_targets['phi'][0, idx].item():.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("   - Use real particle physics data (e.g., from ROOT files)")
    print("   - Scale up the model size and training data")
    print("   - Fine-tune on downstream tasks (classification, regression)")
    print("   - Experiment with different masking strategies")
    print("   - Add physics-informed losses and constraints")


if __name__ == "__main__":
    main() 