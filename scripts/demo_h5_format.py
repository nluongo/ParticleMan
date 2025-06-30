#!/usr/bin/env python3
"""
Demonstration of the HDF5 output format for ParticleMan.

This script shows what the output of convert_xaod_to_h5.py looks like
without requiring ROOT to be installed.

Usage:
    python demo_h5_format.py [output_file.h5]
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Particle ID mapping (same as in convert_xaod_to_h5.py)
PARTICLE_ID_MAP = {
    0: "Electrons/Positrons",
    1: "Muons/Antimuons", 
    2: "Taus/Antitaus",
    3: "Electron Neutrinos",
    4: "Muon Neutrinos",
    5: "Tau Neutrinos",
    6: "Photons",
    7: "Charged Pions",
    8: "Neutral Pions",
    9: "Charged Kaons",
    10: "Neutral Kaons",
    11: "Protons/Antiprotons",
    12: "Neutrons/Antineutrons",
    13: "B Mesons",
    14: "D Mesons",
    15: "Other/Unknown/Padding"
}


def create_demo_data(n_events: int = 100, max_particles: int = 50) -> dict:
    """
    Create synthetic particle physics data matching the expected format.
    
    Args:
        n_events: Number of events to generate
        max_particles: Maximum particles per event
        
    Returns:
        Dictionary with particle data arrays
    """
    logger.info(f"Generating {n_events} demo events...")
    
    np.random.seed(42)  # For reproducible results
    
    # Initialize arrays
    all_pt = np.zeros((n_events, max_particles), dtype=np.float32)
    all_eta = np.zeros((n_events, max_particles), dtype=np.float32)
    all_phi = np.zeros((n_events, max_particles), dtype=np.float32)
    all_particle_id = np.full((n_events, max_particles), 15, dtype=np.int32)  # Default to padding
    all_is_truth = np.zeros((n_events, max_particles), dtype=bool)
    all_event_id = np.arange(n_events, dtype=np.int32)
    
    for i in range(n_events):
        # Random number of particles per event (5-30)
        n_particles = np.random.randint(5, min(30, max_particles))
        
        # Generate truth particles (60% of total)
        n_truth = int(0.6 * n_particles)
        n_reco = n_particles - n_truth
        
        # Truth particles
        for j in range(n_truth):
            # Physics-motivated pt distribution (exponential with minimum)
            all_pt[i, j] = np.random.exponential(15.0) + 1.0
            
            # Eta distribution (roughly Gaussian, wider for truth)
            all_eta[i, j] = np.random.normal(0, 2.0)
            
            # Phi uniform in [-π, π]
            all_phi[i, j] = np.random.uniform(-np.pi, np.pi)
            
            # Random particle type (exclude padding type 15)
            all_particle_id[i, j] = np.random.randint(0, 15)
            
            # Mark as truth
            all_is_truth[i, j] = True
        
        # Reconstructed particles  
        for j in range(n_truth, n_truth + n_reco):
            # Higher pt threshold for reconstructed particles
            all_pt[i, j] = np.random.exponential(20.0) + 5.0
            
            # Narrower eta acceptance for detectors
            all_eta[i, j] = np.random.normal(0, 1.5)
            
            # Phi uniform
            all_phi[i, j] = np.random.uniform(-np.pi, np.pi)
            
            # More common particle types for reconstruction
            common_types = [0, 1, 6, 7]  # electrons, muons, photons, pions
            all_particle_id[i, j] = np.random.choice(common_types)
            
            # Mark as reconstructed
            all_is_truth[i, j] = False
    
    return {
        'pt': all_pt,
        'eta': all_eta,
        'phi': all_phi,
        'particle_id': all_particle_id,
        'is_truth': all_is_truth,
        'event_id': all_event_id
    }


def save_demo_h5(data: dict, output_file: str) -> None:
    """
    Save the demo data to HDF5 format.
    
    Args:
        data: Dictionary with particle data
        output_file: Output HDF5 file path
    """
    logger.info(f"Saving demo data to {output_file}")
    
    with h5py.File(output_file, 'w') as h5f:
        # Create datasets with compression
        for key, value in data.items():
            h5f.create_dataset(key, data=value, compression='gzip')
        
        # Add metadata
        h5f.attrs['description'] = 'Demo particle physics data for ParticleMan transformer'
        h5f.attrs['format_version'] = '1.0'
        h5f.attrs['n_events'] = len(data['pt'])
        h5f.attrs['max_particles_per_event'] = data['pt'].shape[1]
        h5f.attrs['particle_id_map'] = str(PARTICLE_ID_MAP)
        h5f.attrs['data_type'] = 'synthetic_demo'
        
        # Add dataset descriptions
        h5f['pt'].attrs['description'] = 'Transverse momentum in GeV (always positive)'
        h5f['pt'].attrs['units'] = 'GeV'
        h5f['eta'].attrs['description'] = 'Pseudorapidity (can be positive or negative)'
        h5f['phi'].attrs['description'] = 'Azimuthal angle in radians'
        h5f['phi'].attrs['units'] = 'radians'
        h5f['particle_id'].attrs['description'] = 'Categorical particle type ID (0-15)'
        h5f['is_truth'].attrs['description'] = 'Boolean flag: True=truth particle, False=reconstructed'
        h5f['event_id'].attrs['description'] = 'Event index identifier'


def analyze_demo_data(h5_file: str) -> None:
    """
    Analyze and display statistics about the demo data.
    
    Args:
        h5_file: Path to HDF5 file
    """
    logger.info(f"Analyzing {h5_file}...")
    
    with h5py.File(h5_file, 'r') as h5f:
        # Basic info
        print(f"\n📊 Dataset Information:")
        print(f"  Number of events: {h5f.attrs['n_events']}")
        print(f"  Max particles per event: {h5f.attrs['max_particles_per_event']}")
        print(f"  Data shape: {h5f['pt'].shape}")
        
        # Load data
        pt = h5f['pt'][:]
        eta = h5f['eta'][:]
        phi = h5f['phi'][:]
        particle_id = h5f['particle_id'][:]
        is_truth = h5f['is_truth'][:]
        
        # Mask for non-padding particles (pt > 0)
        valid_mask = pt > 0
        
        print(f"\n🔢 Statistics (non-padding particles only):")
        print(f"  Total particles: {valid_mask.sum():,}")
        print(f"  Truth particles: {(valid_mask & is_truth).sum():,}")
        print(f"  Reconstructed particles: {(valid_mask & ~is_truth).sum():,}")
        
        # Physics ranges
        valid_pt = pt[valid_mask]
        valid_eta = eta[valid_mask]
        valid_phi = phi[valid_mask]
        
        print(f"\n⚛️  Physics Ranges:")
        print(f"  PT: {valid_pt.min():.1f} - {valid_pt.max():.1f} GeV")
        print(f"  Eta: {valid_eta.min():.2f} - {valid_eta.max():.2f}")
        print(f"  Phi: {valid_phi.min():.2f} - {valid_phi.max():.2f} rad")
        
        # Particle type distribution
        valid_pid = particle_id[valid_mask]
        unique_ids, counts = np.unique(valid_pid, return_counts=True)
        
        print(f"\n🎯 Particle Type Distribution:")
        for pid, count in zip(unique_ids, counts):
            if pid != 15 and count > 0:  # Skip padding
                percentage = 100 * count / len(valid_pid)
                print(f"  {pid:2d}: {PARTICLE_ID_MAP[pid]:<20} {count:6,} ({percentage:4.1f}%)")
        
        # Event size distribution
        particles_per_event = valid_mask.sum(axis=1)
        print(f"\n📈 Particles per Event:")
        print(f"  Mean: {particles_per_event.mean():.1f}")
        print(f"  Std:  {particles_per_event.std():.1f}")
        print(f"  Min:  {particles_per_event.min()}")
        print(f"  Max:  {particles_per_event.max()}")


def demonstrate_pytorch_loading(h5_file: str) -> None:
    """
    Demonstrate how to load the data for PyTorch training.
    
    Args:
        h5_file: Path to HDF5 file
    """
    print(f"\n💡 PyTorch Integration Example:")
    
    # Show the basic loading pattern
    code_example = '''
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

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
dataset = ParticlePhysicsDataset('demo_output.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in dataloader:
    # batch['pt'] shape: [32, max_particles]
    # batch['particle_id'] shape: [32, max_particles] 
    # Ready for ParticleTransformer!
    pass
'''
    
    print(code_example)


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Create demo HDF5 file showing ParticleMan data format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('output_file', nargs='?', default='demo_output.h5',
                       help='Output HDF5 file (default: demo_output.h5)')
    parser.add_argument('--events', type=int, default=100,
                       help='Number of events to generate (default: 100)')
    parser.add_argument('--max-particles', type=int, default=50,
                       help='Maximum particles per event (default: 50)')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip data analysis output')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate demo data
        data = create_demo_data(args.events, args.max_particles)
        
        # Save to HDF5
        save_demo_h5(data, args.output_file)
        
        if not args.no_analysis:
            # Analyze the data
            analyze_demo_data(args.output_file)
            
            # Show PyTorch integration
            demonstrate_pytorch_loading(args.output_file)
        
        logger.info(f"✅ Demo complete! Output saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 