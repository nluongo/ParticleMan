# ParticleMan

A framework for pre-training foundation models for particle physics tasks.

[![CI](https://github.com/yourusername/particleman/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/particleman/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ParticleMan is a comprehensive framework designed to facilitate the development, training, and deployment of foundation models for particle physics applications. It provides modular, readable, and safe tools for handling particle physics data, building neural network architectures, and implementing state-of-the-art pre-training techniques.

## Features

- **Modular Architecture**: Clean, modular design for easy extension and customization
- **Type Safety**: Full type hints and static type checking with mypy
- **Comprehensive Testing**: Extensive test suite with pytest
- **Code Quality**: Automated linting and formatting with ruff, black, and isort
- **CI/CD**: Automated testing and quality checks with GitHub Actions
- **Documentation**: Comprehensive documentation and usage examples

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

```python
import particleman

# Check version
print(particleman.__version__)

# Your code here...
```

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
│       └── tests/
│           ├── __init__.py
│           └── test_version.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── environment.yml          # CPU-only conda environment
├── environment-gpu.yml      # GPU-enabled conda environment
├── .pre-commit-config.yaml
├── .gitignore
├── .cursorrules
├── README.md
└── LICENSE
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
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/particleman}
}
``` 