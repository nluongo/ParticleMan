# ParticleMan Data Conversion Scripts

This directory contains scripts for converting particle physics data to formats suitable for training the ParticleTransformer model.

## `convert_xaod_to_h5.py`

Converts ROOT xAOD files (ATLAS Analysis Object Data format) to HDF5 files suitable for machine learning training.

### Features

- **Extracts both truth and reconstructed particles**
- **Applies physics-motivated quality cuts**
- **Maps PDG particle IDs to categorical IDs (0-15)**
- **Handles variable-length events with padding/truncation**
- **Preserves event structure and metadata**

### Particle ID Mapping

The script maps complex PDG particle codes to simple categorical IDs:

| ID | Particle Type | PDG Codes | Notes |
|----|---------------|-----------|-------|
| 0  | Electrons/Positrons | ±11 | |
| 1  | Muons/Antimuons | ±13 | |
| 2  | Taus/Antitaus | ±15 | |
| 3  | Electron Neutrinos | ±12 | Truth only |
| 4  | Muon Neutrinos | ±14 | Truth only |
| 5  | Tau Neutrinos | ±16 | Truth only |
| 6  | Photons | 22 | |
| 7  | Charged Pions | ±211 | Also used for jets |
| 8  | Neutral Pions | 111 | |
| 9  | Charged Kaons | ±321 | |
| 10 | Neutral Kaons | 130, 310 | |
| 11 | Protons/Antiprotons | ±2212 | |
| 12 | Neutrons/Antineutrons | ±2112 | |
| 13 | B Mesons | ±511, ±521, ±531 | |
| 14 | D Mesons | ±411, ±421, ±431 | |
| 15 | Other/Unknown | All others | Also used for padding |

### Usage

```bash
# Basic usage
python scripts/convert_xaod_to_h5.py input.root output.h5

# Limit number of events
python scripts/convert_xaod_to_h5.py input.root output.h5 --max-events 1000

# Custom maximum particles per event
python scripts/convert_xaod_to_h5.py input.root output.h5 --max-particles 150

# Verbose output
python scripts/convert_xaod_to_h5.py input.root output.h5 --verbose
```

### Requirements

Install the required dependencies:

```bash
# Using conda (recommended)
conda install -c conda-forge root h5py numpy tqdm

# Or using pip (if ROOT is already installed)
pip install h5py numpy tqdm
```

### Output Format

The resulting HDF5 file contains:

**Datasets:**
- `pt`: Transverse momentum in GeV [n_events, max_particles]
- `eta`: Pseudorapidity [n_events, max_particles]  
- `phi`: Azimuthal angle in radians [n_events, max_particles]
- `particle_id`: Categorical particle type (0-15) [n_events, max_particles]
- `is_truth`: Boolean flag (True=truth, False=reco) [n_events, max_particles]
- `event_id`: Original event index [n_events]

**Metadata:**
- Source file information
- Particle ID mapping
- Quality cut parameters
- Dataset descriptions

### Quality Cuts Applied

**Truth Particles:**
- Only final state particles (status code 1 or 23)
- pt > 0.5 GeV
- |eta| < 5.0

**Reconstructed Particles:**
- **Electrons:** pt > 7 GeV, |eta| < 2.47
- **Muons:** pt > 6 GeV, |eta| < 2.5  
- **Photons:** pt > 10 GeV, |eta| < 2.37
- **Jets:** pt > 20 GeV, |eta| < 4.5

### Example Integration with ParticleMan

```python
import h5py
import torch
from torch.utils.data import Dataset

class ParticlePhysicsDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.n_events = len(f['pt'])
    
    def __len__(self):
        return self.n_events
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return {
                'pt': torch.tensor(f['pt'][idx], dtype=torch.float32),
                'eta': torch.tensor(f['eta'][idx], dtype=torch.float32),
                'phi': torch.tensor(f['phi'][idx], dtype=torch.float32),
                'particle_id': torch.tensor(f['particle_id'][idx], dtype=torch.long),
                'is_truth': torch.tensor(f['is_truth'][idx], dtype=torch.bool)
            }

# Usage
dataset = ParticlePhysicsDataset('output.h5')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

### Troubleshooting

**Common Issues:**

1. **ROOT not found:** Install ROOT with Python bindings
2. **Tree not found:** Check tree names in your xAOD file
3. **Container not found:** xAOD container names may vary between versions
4. **Memory issues:** Reduce `--max-events` or `--max-particles`

**Debugging:**

Use `--verbose` flag for detailed logging:
```bash
python scripts/convert_xaod_to_h5.py input.root output.h5 --verbose
```

### Extending the Script

To add support for new particle types or containers:

1. **Add PDG codes to `PARTICLE_ID_MAP`**
2. **Update particle extraction functions**
3. **Add quality cuts as needed**
4. **Update documentation**

The script is designed to be modular and easily extensible for different xAOD formats and analysis requirements. 