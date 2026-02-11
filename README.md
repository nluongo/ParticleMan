# ParticleMan

A framework for pre-training foundation models for particle physics tasks.

[![CI](https://github.com/yourusername/particleman/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/particleman/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ParticleMan is a comprehensive framework designed to facilitate the development, training, and deployment of foundation models for particle physics applications. It provides modular, readable, and safe tools for handling particle physics data, building neural network architectures, and implementing state-of-the-art pre-training techniques.

## Features

- **Modular Architecture**: Clean, modular design for easy extension and customization
- **Flexible Data Loading**: Config-driven data loading from ROOT and HDF5 files
- **Multiple Particle Collections**: Support for combining electrons, muons, jets, and other particle types
- **Type Safety**: Full type hints and static type checking with mypy
- **Comprehensive Testing**: Extensive test suite with pytest
- **Code Quality**: Automated linting and formatting with ruff, black, and isort
- **CI/CD**: Automated testing and quality checks with GitHub Actions

## Installation

### Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/particleman.git
cd particleman
```

2. Create and activate the conda environment:

   **For CPU-only systems (default):**
   ```bash
   conda env create -f environment.yml
   conda activate particleman
   ```

   **For GPU systems with CUDA support:**
   ```bash
   conda env create -f environment-gpu.yml
   conda activate particleman-gpu
   ```

3. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

**Note:** If you encounter CUDA-related errors, use the CPU version first. You can always install PyTorch with GPU support later using:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Using pip

```bash
pip install particleman
```

## Quick Start

### Loading Data

ParticleMan uses YAML configuration files to define how data is loaded from ROOT or HDF5 files:

```python
from particleman import create_dataloaders, load_config

# Create train/val/test dataloaders from a config file
train_loader, val_loader, test_loader = create_dataloaders(
    "configs/data/my_config.yaml",
    batch_size=32,
    num_workers=4,
)

# Training loop
for batch in train_loader:
    pt = batch['pt']           # (batch_size, max_particles)
    eta = batch['eta']         # (batch_size, max_particles)
    phi = batch['phi']         # (batch_size, max_particles)
    particle_id = batch['particle_id']  # (batch_size, max_particles)
    mask = batch['mask']       # (batch_size, max_particles) - True for real particles
    # ... your training code
```

### Training a Model

```python
from particleman import ParticleTransformer, ParticleConfig, ParticleTrainer
from particleman import create_dataloaders

# Create model
config = ParticleConfig(
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_particles=200,
)
model = ParticleTransformer(config)

# Load data
train_loader, val_loader, _ = create_dataloaders(
    "configs/data/physlite.yaml",
    batch_size=64,
)

# Train
trainer = ParticleTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    lr=1e-4,
)
trainer.train(num_epochs=10)
```

## Data Configuration

ParticleMan uses YAML files to configure data loading. This allows you to:
- Switch between ROOT and HDF5 files without code changes
- Define multiple particle collections (electrons, muons, jets, etc.)
- Map column names to model inputs
- Configure preprocessing (pT cuts, scaling, etc.)

### Example Configuration

```yaml
# configs/data/my_config.yaml
source:
  type: "root"  # or "hdf5"
  files:
    - "data/*.root"
  tree_name: "CollectionTree"

# Define particle collections to load
collections:
  electrons:
    enabled: true
    columns:
      pt: "AnalysisElectronsAuxDyn.pt"
      eta: "AnalysisElectronsAuxDyn.eta"
      phi: "AnalysisElectronsAuxDyn.phi"
    fixed_particle_id: 0  # Assign all electrons to category 0
    is_vector: true

  jets:
    enabled: true
    columns:
      pt: "AnalysisJetsAuxDyn.pt"
      eta: "AnalysisJetsAuxDyn.eta"
      phi: "AnalysisJetsAuxDyn.phi"
      particle_id: "AnalysisJetsAuxDyn.PartonTruthLabelID"
    is_vector: true

# Map raw IDs to model categories
particle_id_map:
  5: 13   # b-jet -> B meson category
  4: 14   # c-jet -> D meson category
  0: 7    # light jet -> pion category

preprocessing:
  pt_cut: 0.5       # Minimum pT in GeV
  eta_cut: 5.0      # Maximum |eta|
  max_particles: 200
  pt_scale: 1000.0  # MeV -> GeV
```

### Exploring Your Data Files

Use the exploration scripts to discover column names in your data:

```bash
# For ROOT files
python scripts/explore_root_file.py data.root --filter "*pt*" --sample 0

# For HDF5 files
python scripts/explore_h5_file.py data.h5 --sample 0
```

These scripts will suggest configuration snippets based on your file structure.

## Data Processing

ParticleMan includes tools for converting particle physics data from various formats to HDF5 for efficient machine learning training.

### Converting xAOD ROOT Files

The `convert_xaod_to_h5.py` script processes ATLAS xAOD files and extracts both truth and reconstructed particles:

```bash
# Basic usage
python scripts/convert_xaod_to_h5.py input.root output.h5

# With options
python scripts/convert_xaod_to_h5.py input.root output.h5 --max-events 1000 --max-particles 150
```

**Requirements for xAOD processing:**
```bash
# Install ROOT with Python bindings
conda install -c conda-forge root
```

### Demo Data Format

To see the expected HDF5 output format without requiring ROOT:

```bash
# Create demo data
python scripts/demo_h5_format.py demo_output.h5 --events 100

# This will show:
# - Particle type distributions
# - Physics ranges (pt, eta, phi)
# - PyTorch integration example
```

### Supported Particle Types

The conversion maps PDG particle codes to categorical IDs (0-15):

| ID | Particle Type | PDG Codes | Notes |
|----|---------------|-----------|-------|
| 0  | Electrons/Positrons | ±11 | |
| 1  | Muons/Antimuons | ±13 | |
| 2  | Taus/Antitaus | ±15 | |
| 6  | Photons | 22 | |
| 7  | Charged Pions | ±211 | Also used for jets |
| 15 | Other/Unknown | All others | Also used for padding |

See `scripts/README.md` for complete documentation.

## Development

### Setting up the Development Environment

1. Clone the repository and create the conda environment as described above
2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests to ensure everything is working:
```bash
pytest
```

### Code Quality

This project uses several tools to maintain code quality:

- **ruff**: Fast Python linter
- **black**: Code formatter
- **isort**: Import sorter
- **mypy**: Static type checker
- **flake8**: Additional linting

Run all quality checks:
```bash
# Linting
ruff check src/
flake8 src/

# Formatting
black src/
isort src/

# Type checking
mypy src/

# Tests
pytest
```

### Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=particleman

# Run specific test file
pytest src/particleman/tests/test_version.py
```

## Project Structure

```
particleman/
├── src/
│   └── particleman/
│       ├── __init__.py
│       ├── data/                  # Data loading module
│       │   ├── config.py          # Configuration dataclasses
│       │   ├── base_loader.py     # Abstract base loader
│       │   ├── root_loader.py     # ROOT file loader (uproot)
│       │   ├── hdf5_loader.py     # HDF5 file loader
│       │   └── dataset.py         # PyTorch Dataset wrapper
│       ├── models/
│       │   └── particle_transformer.py
│       ├── training/
│       │   └── trainer.py
│       └── tests/
├── configs/
│   └── data/                      # Data configuration files
│       ├── default.yaml
│       ├── physlite.yaml
│       └── preprocessed_h5.yaml
├── scripts/
│   ├── explore_root_file.py       # ROOT file structure explorer
│   ├── explore_h5_file.py         # HDF5 file structure explorer
│   ├── convert_xaod_to_h5.py
│   └── demo_h5_format.py
├── environment.yml                # CPU-only conda environment
├── environment-gpu.yml            # GPU-enabled conda environment
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ParticleMan in your research, please cite:

```bibtex
@software{particleman,
  title={ParticleMan: A framework for pre-training foundation models for particle physics tasks},
  author=Nicholas Luongo,
  year={2024},
  url={https://github.com/nluongo/particleman}
}
``` 
