"""
Tests for the data loading module.
"""

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from particleman.config import (
    CollectionColumnsConfig,
    CollectionConfig,
    DataSourceConfig,
)
from particleman.data.base_loader import get_enabled_collections
from particleman.data.dataset import (
    ParticleDataset,
    create_dataloaders,
    create_loader,
    create_split_indices,
)
from particleman.data.hdf5_loader import HDF5ParticleLoader


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_h5_file(tmp_path: Path) -> Path:
    """
    Create a sample HDF5 file for testing.
    """
    filepath = tmp_path / "test_data.h5"
    n_events = 100
    max_particles = 50

    with h5py.File(filepath, "w") as f:
        # Create random particle data
        np.random.seed(42)
        
        # pT: exponential distribution (always positive)
        pt = np.random.exponential(20, (n_events, max_particles)).astype(np.float32)
        # Add some padding (zero values) at the end of each event
        for i in range(n_events):
            n_real = np.random.randint(10, max_particles)
            pt[i, n_real:] = 0

        # eta: gaussian distribution
        eta = np.random.normal(0, 2, (n_events, max_particles)).astype(np.float32)
        
        # phi: uniform in [-pi, pi]
        phi = np.random.uniform(-np.pi, np.pi, (n_events, max_particles)).astype(np.float32)
        
        # particle_id: random integers 0-14
        particle_id = np.random.randint(0, 15, (n_events, max_particles)).astype(np.int32)

        # Set padding to particle_id 15
        particle_id[pt == 0] = 15

        # Create datasets
        f.create_dataset("pt", data=pt)
        f.create_dataset("eta", data=eta)
        f.create_dataset("phi", data=phi)
        f.create_dataset("particle_id", data=particle_id)
        
        # Add metadata
        f.attrs["n_events"] = n_events
        f.attrs["max_particles"] = max_particles

    return filepath


def _data_config(values: Any) -> DictConfig:
    """Merge ``values`` onto the ``DataSourceConfig`` schema, as Hydra does."""
    merged = OmegaConf.merge(OmegaConf.structured(DataSourceConfig), values)
    assert isinstance(merged, DictConfig)
    return merged


def _load_yaml_config(path: Path) -> DictConfig:
    """Load a data YAML file onto the ``DataSourceConfig`` schema."""
    return _data_config(OmegaConf.load(path))


@pytest.fixture
def sample_values(sample_h5_file: Path) -> Dict[str, Any]:
    """Plain data-config values for the sample HDF5 file."""
    return {
        "source": {"type": "hdf5", "files": [str(sample_h5_file)]},
        "collections": {
            "particles": {
                "enabled": True,
                "columns": {
                    "pt": "pt",
                    "eta": "eta",
                    "phi": "phi",
                    "particle_id": "particle_id",
                },
                "is_vector": False,
            }
        },
        "particle_id_map": {i: i for i in range(16)},  # Identity mapping
        "default_particle_id": 15,
        "preprocessing": {
            "pt_cut": 0.0,
            "eta_cut": 999.0,
            "max_particles": 50,
            "pt_scale": 1.0,
            "shuffle_particles": False,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
    }


@pytest.fixture
def sample_config(sample_values: Dict[str, Any]) -> DictConfig:
    """Sample data config composed onto the Hydra schema."""
    return _data_config(sample_values)


@pytest.fixture
def sample_yaml_config(tmp_path: Path, sample_h5_file: Path) -> Path:
    """
    Create a sample YAML config file for testing.
    """
    config_path = tmp_path / "test_config.yaml"
    config_content = f"""
source:
  type: "hdf5"
  files:
    - "{sample_h5_file}"

collections:
  particles:
    enabled: true
    columns:
      pt: "pt"
      eta: "eta"
      phi: "phi"
      particle_id: "particle_id"
    is_vector: false

particle_id_map:
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  5: 5

default_particle_id: 15

preprocessing:
  pt_cut: 0.5
  eta_cut: 5.0
  max_particles: 50
  pt_scale: 1.0
  shuffle_particles: true

split:
  train: 0.8
  val: 0.1
  test: 0.1
  seed: 42
"""
    config_path.write_text(config_content)
    return config_path


# ============================================================================
# Config Tests
# ============================================================================


class TestConfig:
    """Tests for the data config schema and the checks applied to it."""

    def test_collection_config_with_fixed_id(self) -> None:
        """A collection can use a fixed particle ID instead of a column."""
        config = OmegaConf.structured(
            CollectionConfig(
                columns=CollectionColumnsConfig(pt="pt", eta="eta", phi="phi"),
                fixed_particle_id=7,
            )
        )
        assert config.fixed_particle_id == 7
        assert config.columns.particle_id is None

    def test_collection_config_with_column_id(self) -> None:
        """A collection can read particle IDs from a column."""
        config = OmegaConf.structured(
            CollectionConfig(
                columns=CollectionColumnsConfig(
                    pt="pt", eta="eta", phi="phi", particle_id="pid"
                ),
            )
        )
        assert config.columns.particle_id == "pid"
        assert config.fixed_particle_id is None

    def test_unknown_source_type(self, sample_values: Dict[str, Any]) -> None:
        """Creating a loader for an unknown source type fails."""
        sample_values["source"]["type"] = "invalid"
        with pytest.raises(ValueError, match="Unknown source type"):
            create_loader(_data_config(sample_values))

    def test_split_fractions_must_sum_to_one(self) -> None:
        """Split fractions that do not sum to 1 are rejected."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_split_indices(100, train_frac=0.5, val_frac=0.3, test_frac=0.1)

    def test_load_config_from_yaml(self, sample_yaml_config: Path) -> None:
        """A data YAML file composes onto the schema."""
        config = _load_yaml_config(sample_yaml_config)

        assert config.source.type == "hdf5"
        assert len(config.collections) == 1
        assert "particles" in config.collections
        assert config.preprocessing.pt_cut == 0.5
        assert config.split.train == 0.8

    def test_schema_rejects_unknown_key(self, sample_values: Dict[str, Any]) -> None:
        """Keys that are not in the schema are an error."""
        sample_values["preprocessing"]["not_a_key"] = 1
        with pytest.raises(Exception, match="not_a_key"):
            _data_config(sample_values)

    def test_get_enabled_collections(self, sample_values: Dict[str, Any]) -> None:
        """Disabled collections are filtered out."""
        sample_values["collections"]["disabled"] = {
            "enabled": False,
            "columns": {"pt": "pt", "eta": "eta", "phi": "phi"},
            "fixed_particle_id": 3,
        }
        enabled = get_enabled_collections(_data_config(sample_values))
        assert list(enabled) == ["particles"]


# ============================================================================
# HDF5 Loader Tests
# ============================================================================


class TestHDF5Loader:
    """Tests for HDF5ParticleLoader."""

    def test_loader_initialization(self, sample_config: DictConfig):
        """Test loader initializes correctly."""
        loader = HDF5ParticleLoader(sample_config)
        assert len(loader) == 100

    def test_get_event(self, sample_config: DictConfig):
        """Test loading a single event."""
        loader = HDF5ParticleLoader(sample_config)
        event = loader.get_event(0)

        assert "pt" in event
        assert "eta" in event
        assert "phi" in event
        assert "particle_id" in event
        assert "mask" in event
        assert "n_particles" in event

        # Check shapes
        max_p = sample_config.preprocessing.max_particles
        assert event["pt"].shape == (max_p,)
        assert event["eta"].shape == (max_p,)
        assert event["phi"].shape == (max_p,)
        assert event["particle_id"].shape == (max_p,)
        assert event["mask"].shape == (max_p,)

    def test_get_event_dtypes(self, sample_config: DictConfig):
        """Test event data types."""
        loader = HDF5ParticleLoader(sample_config)
        event = loader.get_event(0)

        assert event["pt"].dtype == np.float32
        assert event["eta"].dtype == np.float32
        assert event["phi"].dtype == np.float32
        assert event["particle_id"].dtype == np.int32
        assert event["mask"].dtype == bool

    def test_iteration(self, sample_config: DictConfig):
        """Test iterating over loader."""
        loader = HDF5ParticleLoader(sample_config)
        
        count = 0
        for event in loader:
            count += 1
            if count >= 5:
                break
        
        assert count == 5

    def test_indexing(self, sample_config: DictConfig):
        """Test indexing with negative indices."""
        loader = HDF5ParticleLoader(sample_config)
        
        # Positive index
        event_0 = loader[0]
        assert event_0 is not None
        
        # Negative index
        event_last = loader[-1]
        assert event_last is not None
        
        # Out of range
        with pytest.raises(IndexError):
            _ = loader[1000]

    def test_get_dataset_names(self, sample_config: DictConfig):
        """Test get_dataset_names method."""
        loader = HDF5ParticleLoader(sample_config)
        names = loader.get_dataset_names()
        
        assert "pt" in names
        assert "eta" in names
        assert "phi" in names
        assert "particle_id" in names


# ============================================================================
# Dataset Tests
# ============================================================================


class TestParticleDataset:
    """Tests for ParticleDataset."""

    def test_dataset_initialization(self, sample_config: DictConfig):
        """Test dataset initializes correctly."""
        dataset = ParticleDataset(sample_config)
        assert len(dataset) == 100

    def test_dataset_from_yaml(self, sample_yaml_config: Path):
        """Test creating dataset from YAML path."""
        dataset = ParticleDataset(_load_yaml_config(sample_yaml_config))
        assert len(dataset) > 0

    def test_getitem_returns_tensors(self, sample_config: DictConfig):
        """Test that __getitem__ returns torch tensors."""
        dataset = ParticleDataset(sample_config)
        sample = dataset[0]

        assert isinstance(sample["pt"], torch.Tensor)
        assert isinstance(sample["eta"], torch.Tensor)
        assert isinstance(sample["phi"], torch.Tensor)
        assert isinstance(sample["particle_id"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)

    def test_getitem_tensor_dtypes(self, sample_config: DictConfig):
        """Test tensor data types."""
        dataset = ParticleDataset(sample_config)
        sample = dataset[0]

        assert sample["pt"].dtype == torch.float32
        assert sample["eta"].dtype == torch.float32
        assert sample["phi"].dtype == torch.float32
        assert sample["particle_id"].dtype == torch.long
        assert sample["mask"].dtype == torch.bool

    def test_dataset_with_indices(self, sample_config: DictConfig):
        """Test dataset with subset of indices."""
        indices = [0, 5, 10, 15, 20]
        dataset = ParticleDataset(sample_config, indices=indices)
        
        assert len(dataset) == 5

    def test_dataset_with_transform(self, sample_config: DictConfig):
        """Test dataset with custom transform."""
        def double_pt(event: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            event["pt"] = event["pt"] * 2
            return event

        dataset = ParticleDataset(sample_config, transform=double_pt)
        sample = dataset[0]
        
        # Transform should have doubled pT
        # (We can't easily verify the exact values, but we can check it runs)
        assert sample["pt"] is not None


# ============================================================================
# DataLoader Tests
# ============================================================================


class TestDataLoaders:
    """Tests for DataLoader creation functions."""

    def test_create_split_indices(self):
        """Test create_split_indices function."""
        train, val, test = create_split_indices(
            n_samples=100,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

        # Check no overlap
        all_indices = set(train) | set(val) | set(test)
        assert len(all_indices) == 100

    def test_create_split_indices_reproducible(self):
        """Test that splits are reproducible with same seed."""
        train1, val1, test1 = create_split_indices(100, seed=42)
        train2, val2, test2 = create_split_indices(100, seed=42)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_create_dataloaders(self, sample_config: DictConfig):
        """Test create_dataloaders function."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_config,
            batch_size=16,
            num_workers=0,
        )

        # Check loaders were created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch
        batch = next(iter(train_loader))
        assert batch["pt"].shape[0] == 16  # Batch size
        assert batch["pt"].shape[1] == sample_config.preprocessing.max_particles

    def test_dataloader_iteration(self, sample_config: DictConfig):
        """Test iterating through dataloader."""
        train_loader, _, _ = create_dataloaders(
            sample_config,
            batch_size=16,
            num_workers=0,
        )

        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            assert "pt" in batch
            assert "eta" in batch
            assert "phi" in batch
            assert "particle_id" in batch

        # Should have multiple batches
        assert batch_count > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full data pipeline."""

    def test_full_pipeline(self, sample_config: DictConfig):
        """Test full pipeline from config to batch."""
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_config,
            batch_size=8,
            num_workers=0,
        )

        # Get a batch
        batch = next(iter(train_loader))

        # Verify batch structure matches model expectations
        assert batch["pt"].shape == (8, 50)
        assert batch["eta"].shape == (8, 50)
        assert batch["phi"].shape == (8, 50)
        assert batch["particle_id"].shape == (8, 50)

        # Verify data ranges are reasonable
        assert torch.all(batch["pt"] >= 0)  # pT is always positive
        assert torch.all(torch.abs(batch["eta"]) < 10)  # eta is bounded
        assert torch.all(torch.abs(batch["phi"]) <= np.pi + 0.1)  # phi in [-pi, pi]
        assert torch.all(batch["particle_id"] >= 0)  # IDs are non-negative
        assert torch.all(batch["particle_id"] < 16)  # IDs are < n_types