#!/usr/bin/env python3
"""
Example: Loading mc20 ttbar data with ParticleMan.

This script demonstrates how to:
1. Load configuration from YAML
2. Create datasets and dataloaders
3. Iterate through batches
4. Visualize particle distributions

Usage:
    python examples/load_mc20_data.py

Requirements:
    - The mc20*.root file in the project root
    - uproot and awkward installed
"""

import sys
from pathlib import Path

# Add src to path if running from examples directory
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import numpy as np

# Check for required dependencies
try:
    import uproot
    import awkward as ak
except ImportError:
    print("ERROR: uproot and awkward are required.")
    print("Install with: pip install uproot awkward")
    sys.exit(1)

from particleman.data import (
    DataConfig,
    ParticleDataset,
    create_dataloaders,
    load_config,
)


def main():
    print("=" * 70)
    print("ParticleMan: Loading mc20 ttbar Data")
    print("=" * 70)
    
    # Path to configuration
    config_path = Path("configs/data/mc20_ttbar.yaml")
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    # Load configuration
    print(f"\n1. Loading configuration from {config_path}")
    config = load_config(config_path)
    
    print(f"   Source type: {config.source.type}")
    print(f"   Files: {config.source.files}")
    print(f"   Tree: {config.source.tree_name}")
    print(f"   Enabled collections: {list(config.get_enabled_collections().keys())}")
    print(f"   Max particles: {config.preprocessing.max_particles}")
    print(f"   pT cut: {config.preprocessing.pt_cut} GeV")
    
    # Check if ROOT file exists
    root_file = Path(config.source.files[0])
    if not root_file.exists():
        print(f"\nERROR: ROOT file not found: {root_file}")
        sys.exit(1)
    
    # Create a single dataset first to explore
    print("\n2. Creating ParticleDataset...")
    try:
        dataset = ParticleDataset(config)
        print(f"   Total events: {len(dataset)}")
    except Exception as e:
        print(f"   ERROR creating dataset: {e}")
        print("\n   This might be due to branch name mismatches.")
        print("   Run: python scripts/explore_root_file.py <your_file.root>")
        print("   to discover the correct branch names.")
        sys.exit(1)
    
    # Get a sample event
    print("\n3. Sampling first event...")
    sample = dataset[0]
    
    print(f"   pt shape: {sample['pt'].shape}")
    print(f"   eta shape: {sample['eta'].shape}")
    print(f"   phi shape: {sample['phi'].shape}")
    print(f"   particle_id shape: {sample['particle_id'].shape}")
    print(f"   mask shape: {sample['mask'].shape}")
    print(f"   n_particles: {sample['n_particles'].item()}")
    
    # Show some values
    n_real = sample['n_particles'].item()
    print(f"\n   First 5 real particles:")
    print(f"   pt:  {sample['pt'][:min(5, n_real)].numpy()}")
    print(f"   eta: {sample['eta'][:min(5, n_real)].numpy()}")
    print(f"   phi: {sample['phi'][:min(5, n_real)].numpy()}")
    print(f"   id:  {sample['particle_id'][:min(5, n_real)].numpy()}")
    
    # Create dataloaders with smaller subset for demo
    print("\n4. Creating DataLoaders (using first 1000 events for demo)...")
    
    # Modify config for smaller demo
    demo_config = load_config(config_path)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        demo_config,
        batch_size=32,
        num_workers=0,  # Use 0 for debugging
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Get a batch
    print("\n5. Getting a training batch...")
    batch = next(iter(train_loader))
    
    print(f"   Batch pt shape: {batch['pt'].shape}")
    print(f"   Batch particle_id shape: {batch['particle_id'].shape}")
    
    # Compute some statistics
    print("\n6. Computing batch statistics...")
    
    # pT statistics (excluding padding)
    mask = batch['mask']
    pt_values = batch['pt'][mask]
    
    print(f"   Total real particles in batch: {mask.sum().item()}")
    print(f"   Mean particles per event: {mask.sum(dim=1).float().mean().item():.1f}")
    
    if len(pt_values) > 0:
        print(f"   pT range: [{pt_values.min().item():.1f}, {pt_values.max().item():.1f}] GeV")
        print(f"   pT mean: {pt_values.mean().item():.1f} GeV")
    
    # Particle ID distribution
    pid_values = batch['particle_id'][mask]
    unique_ids, counts = pid_values.unique(return_counts=True)
    
    print(f"\n   Particle ID distribution:")
    id_names = {
        0: "electron",
        1: "muon",
        6: "photon",
        7: "light jet/pion",
        13: "b-jet/B meson",
        14: "c-jet/D meson",
        15: "unknown/other",
    }
    for pid, count in zip(unique_ids.tolist(), counts.tolist()):
        name = id_names.get(pid, f"id={pid}")
        print(f"     {name}: {count}")
    
    # Show how to use with the model
    print("\n7. Example: Using with ParticleTransformer...")
    print("""
    from particleman import ParticleTransformer, ParticleConfig
    
    # Create model config matching data
    model_config = ParticleConfig(
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_particles=50,  # Match data config
        n_particle_types=16,
    )
    
    # Create model
    model = ParticleTransformer(model_config)
    
    # Forward pass
    for batch in train_loader:
        outputs = model(
            pt=batch['pt'],
            eta=batch['eta'],
            phi=batch['phi'],
            particle_id=batch['particle_id'],
        )
        # outputs['hidden_states'] contains the embeddings
        break
    """)
    
    print("\n" + "=" * 70)
    print("Done! Data loading is working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()