#!/usr/bin/env python3
"""
Quick test of the data loading module.

This script tests the data loading infrastructure without pytest,
making it easy to run and debug.

Usage:
    python test_data_loading_quick.py
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Add src to path
sys.path.insert(0, "src")


def create_test_h5_file(filepath: Path, n_events: int = 50, max_particles: int = 30):
    """Create a simple test HDF5 file."""
    print(f"  Creating test HDF5 file: {filepath}")
    
    np.random.seed(42)
    
    # Generate data
    pt = np.random.exponential(20, (n_events, max_particles)).astype(np.float32)
    eta = np.random.normal(0, 2, (n_events, max_particles)).astype(np.float32)
    phi = np.random.uniform(-np.pi, np.pi, (n_events, max_particles)).astype(np.float32)
    particle_id = np.random.randint(0, 15, (n_events, max_particles)).astype(np.int32)
    
    # Add some padding (zero pT)
    for i in range(n_events):
        n_real = np.random.randint(5, max_particles)
        pt[i, n_real:] = 0
        particle_id[i, n_real:] = 15
    
    with h5py.File(filepath, "w") as f:
        f.create_dataset("pt", data=pt)
        f.create_dataset("eta", data=eta)
        f.create_dataset("phi", data=phi)
        f.create_dataset("particle_id", data=particle_id)
        f.attrs["n_events"] = n_events
    
    print(f"  Created {n_events} events with up to {max_particles} particles each")


def test_config():
    """Test configuration classes."""
    print("\n1. Testing configuration classes...")
    
    from particleman.data.config import (
        CollectionColumns,
        CollectionConfig,
        DataConfig,
        PreprocessingConfig,
        SourceConfig,
        SplitConfig,
    )
    
    # Test CollectionColumns
    cols = CollectionColumns(pt="pt", eta="eta", phi="phi", particle_id="pid")
    assert cols.pt == "pt"
    print("  ✓ CollectionColumns works")
    
    # Test CollectionConfig with fixed_particle_id
    coll = CollectionConfig(
        enabled=True,
        columns=CollectionColumns(pt="pt", eta="eta", phi="phi"),
        fixed_particle_id=0,
    )
    assert coll.fixed_particle_id == 0
    print("  ✓ CollectionConfig with fixed_particle_id works")
    
    # Test CollectionConfig with particle_id column
    coll2 = CollectionConfig(
        enabled=True,
        columns=CollectionColumns(pt="pt", eta="eta", phi="phi", particle_id="pid"),
    )
    assert coll2.columns.particle_id == "pid"
    print("  ✓ CollectionConfig with particle_id column works")
    
    # Test validation - must have either particle_id or fixed_particle_id
    try:
        CollectionConfig(
            enabled=True,
            columns=CollectionColumns(pt="pt", eta="eta", phi="phi"),
            # No fixed_particle_id and no particle_id column!
        )
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError:
        print("  ✓ Validation correctly rejects invalid config")
    
    # Test SourceConfig
    src = SourceConfig(type="hdf5", files=["test.h5"])
    assert src.type == "hdf5"
    print("  ✓ SourceConfig works")
    
    # Test SplitConfig validation
    try:
        SplitConfig(train=0.5, val=0.3, test=0.1)  # Sum != 1.0
        print("  ✗ Should have raised ValueError for invalid split")
        return False
    except ValueError:
        print("  ✓ SplitConfig validation works")
    
    print("  All config tests passed!")
    return True


def test_hdf5_loader(test_h5_path: Path):
    """Test HDF5 loader."""
    print("\n2. Testing HDF5 loader...")
    
    from particleman.data.config import (
        CollectionColumns,
        CollectionConfig,
        DataConfig,
        PreprocessingConfig,
        SourceConfig,
        SplitConfig,
    )
    from particleman.data.hdf5_loader import HDF5ParticleLoader
    
    # Create config
    config = DataConfig(
        source=SourceConfig(type="hdf5", files=[str(test_h5_path)]),
        collections={
            "particles": CollectionConfig(
                enabled=True,
                columns=CollectionColumns(
                    pt="pt", eta="eta", phi="phi", particle_id="particle_id"
                ),
                is_vector=False,
            )
        },
        particle_id_map={i: i for i in range(16)},
        default_particle_id=15,
        preprocessing=PreprocessingConfig(
            pt_cut=0.0,
            eta_cut=999.0,
            max_particles=30,
            pt_scale=1.0,
            shuffle_particles=False,
        ),
    )
    
    # Create loader
    loader = HDF5ParticleLoader(config)
    print(f"  Loader created with {len(loader)} events")
    
    # Test getting an event
    event = loader.get_event(0)
    assert "pt" in event
    assert "eta" in event
    assert "phi" in event
    assert "particle_id" in event
    assert "mask" in event
    assert "n_particles" in event
    print(f"  ✓ Event has all required keys")
    
    # Check shapes
    assert event["pt"].shape == (30,)
    assert event["mask"].shape == (30,)
    print(f"  ✓ Event shapes are correct")
    
    # Check dtypes
    assert event["pt"].dtype == np.float32
    assert event["particle_id"].dtype == np.int32
    assert event["mask"].dtype == bool
    print(f"  ✓ Event dtypes are correct")
    
    # Test iteration
    count = 0
    for e in loader:
        count += 1
        if count >= 5:
            break
    assert count == 5
    print(f"  ✓ Iteration works")
    
    print("  All HDF5 loader tests passed!")
    return True


def test_dataset(test_h5_path: Path):
    """Test PyTorch Dataset."""
    print("\n3. Testing ParticleDataset...")
    
    import torch
    from particleman.data.config import (
        CollectionColumns,
        CollectionConfig,
        DataConfig,
        PreprocessingConfig,
        SourceConfig,
    )
    from particleman.data.dataset import ParticleDataset
    
    # Create config
    config = DataConfig(
        source=SourceConfig(type="hdf5", files=[str(test_h5_path)]),
        collections={
            "particles": CollectionConfig(
                enabled=True,
                columns=CollectionColumns(
                    pt="pt", eta="eta", phi="phi", particle_id="particle_id"
                ),
                is_vector=False,
            )
        },
        particle_id_map={i: i for i in range(16)},
        default_particle_id=15,
        preprocessing=PreprocessingConfig(
            pt_cut=0.0,
            eta_cut=999.0,
            max_particles=30,
            pt_scale=1.0,
            shuffle_particles=False,
        ),
    )
    
    # Create dataset
    dataset = ParticleDataset(config)
    print(f"  Dataset created with {len(dataset)} samples")
    
    # Get a sample
    sample = dataset[0]
    
    # Check types are tensors
    assert isinstance(sample["pt"], torch.Tensor)
    assert isinstance(sample["eta"], torch.Tensor)
    assert isinstance(sample["particle_id"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    print(f"  ✓ Sample contains torch tensors")
    
    # Check dtypes
    assert sample["pt"].dtype == torch.float32
    assert sample["particle_id"].dtype == torch.long
    assert sample["mask"].dtype == torch.bool
    print(f"  ✓ Tensor dtypes are correct")
    
    print("  All dataset tests passed!")
    return True


def test_dataloader(test_h5_path: Path):
    """Test DataLoader creation."""
    print("\n4. Testing DataLoader creation...")
    
    from particleman.data.config import (
        CollectionColumns,
        CollectionConfig,
        DataConfig,
        PreprocessingConfig,
        SourceConfig,
        SplitConfig,
    )
    from particleman.data.dataset import create_dataloaders
    
    # Create config
    config = DataConfig(
        source=SourceConfig(type="hdf5", files=[str(test_h5_path)]),
        collections={
            "particles": CollectionConfig(
                enabled=True,
                columns=CollectionColumns(
                    pt="pt", eta="eta", phi="phi", particle_id="particle_id"
                ),
                is_vector=False,
            )
        },
        particle_id_map={i: i for i in range(16)},
        default_particle_id=15,
        preprocessing=PreprocessingConfig(
            pt_cut=0.0,
            eta_cut=999.0,
            max_particles=30,
            pt_scale=1.0,
            shuffle_particles=False,
        ),
        split=SplitConfig(train=0.8, val=0.1, test=0.1, seed=42),
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=8,
        num_workers=0,
    )
    
    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Val loader: {len(val_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")
    
    # Get a batch
    batch = next(iter(train_loader))
    
    assert batch["pt"].shape[0] == 8  # Batch size
    assert batch["pt"].shape[1] == 30  # Max particles
    print(f"  ✓ Batch shape is correct: {batch['pt'].shape}")
    
    print("  All dataloader tests passed!")
    return True


def test_yaml_config():
    """Test loading config from YAML."""
    print("\n5. Testing YAML config loading...")
    
    from particleman.data.config import load_config
    
    # Check if the mc20 config exists
    config_path = Path("configs/data/mc20_ttbar.yaml")
    if not config_path.exists():
        print(f"  Skipping: {config_path} not found")
        return True
    
    config = load_config(config_path)
    
    assert config.source.type == "root"
    assert config.source.tree_name == "AnalysisMiniTree"
    assert "reco_jets" in config.collections
    assert "electrons" in config.collections
    assert "muons" in config.collections
    print(f"  ✓ YAML config loaded successfully")
    print(f"  ✓ Source type: {config.source.type}")
    print(f"  ✓ Collections: {list(config.collections.keys())}")
    
    print("  All YAML config tests passed!")
    return True


def main():
    print("=" * 60)
    print("ParticleMan Data Loading - Quick Tests")
    print("=" * 60)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test HDF5 file
        test_h5_path = tmpdir / "test_data.h5"
        create_test_h5_file(test_h5_path)
        
        # Run tests
        all_passed = True
        
        all_passed &= test_config()
        all_passed &= test_hdf5_loader(test_h5_path)
        all_passed &= test_dataset(test_h5_path)
        all_passed &= test_dataloader(test_h5_path)
        all_passed &= test_yaml_config()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed!")
        print("=" * 60)
        
        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())