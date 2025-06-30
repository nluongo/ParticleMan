#!/usr/bin/env python3
"""
Test script for the xAOD to HDF5 conversion functionality.

This script creates synthetic ROOT data to test the conversion pipeline
without requiring actual xAOD files.

Usage:
    python test_conversion.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Add the scripts directory to the path so we can import the conversion module
sys.path.insert(0, str(Path(__file__).parent))

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
except ImportError:
    print("ROOT not available - this test requires ROOT with PyROOT")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_root_file(filename: str, n_events: int = 10) -> None:
    """
    Create a synthetic ROOT file with particle data for testing.
    
    Args:
        filename: Output ROOT file name
        n_events: Number of events to generate
    """
    logger.info(f"Creating synthetic ROOT file: {filename}")
    
    # Create ROOT file and tree
    root_file = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("CollectionTree", "Test particle data")
    
    # Define branches for truth particles
    truth_n = np.array([0], dtype=np.int32)
    truth_pt = np.zeros(100, dtype=np.float32)
    truth_eta = np.zeros(100, dtype=np.float32)
    truth_phi = np.zeros(100, dtype=np.float32)
    truth_pdg_id = np.zeros(100, dtype=np.int32)
    truth_status = np.zeros(100, dtype=np.int32)
    
    # Define branches for reconstructed particles
    reco_n = np.array([0], dtype=np.int32)
    reco_pt = np.zeros(50, dtype=np.float32)
    reco_eta = np.zeros(50, dtype=np.float32)
    reco_phi = np.zeros(50, dtype=np.float32)
    reco_type = np.zeros(50, dtype=np.int32)  # 0=electron, 1=muon, 2=photon, 3=jet
    
    # Create branches
    tree.Branch("truth_n", truth_n, "truth_n/I")
    tree.Branch("truth_pt", truth_pt, "truth_pt[truth_n]/F")
    tree.Branch("truth_eta", truth_eta, "truth_eta[truth_n]/F")
    tree.Branch("truth_phi", truth_phi, "truth_phi[truth_n]/F")
    tree.Branch("truth_pdg_id", truth_pdg_id, "truth_pdg_id[truth_n]/I")
    tree.Branch("truth_status", truth_status, "truth_status[truth_n]/I")
    
    tree.Branch("reco_n", reco_n, "reco_n/I")
    tree.Branch("reco_pt", reco_pt, "reco_pt[reco_n]/F")
    tree.Branch("reco_eta", reco_eta, "reco_eta[reco_n]/F")
    tree.Branch("reco_phi", reco_phi, "reco_phi[reco_n]/F")
    tree.Branch("reco_type", reco_type, "reco_type[reco_n]/I")
    
    # Generate events
    np.random.seed(42)  # For reproducible results
    
    for event in range(n_events):
        # Generate truth particles
        n_truth = np.random.poisson(15)  # Average 15 truth particles per event
        n_truth = min(n_truth, 100)  # Limit to array size
        truth_n[0] = n_truth
        
        for i in range(n_truth):
            # Generate random kinematic variables
            truth_pt[i] = np.random.exponential(20.0) + 1.0  # Exponential pt distribution
            truth_eta[i] = np.random.normal(0, 2.0)  # Gaussian eta
            truth_phi[i] = np.random.uniform(-np.pi, np.pi)  # Uniform phi
            
            # Random particle types (using common PDG codes)
            particle_types = [11, -11, 13, -13, 22, 211, -211, 111, 321, -321, 2212, -2212]
            truth_pdg_id[i] = np.random.choice(particle_types)
            truth_status[i] = 1  # Final state
        
        # Generate reconstructed particles (fewer than truth)
        n_reco = np.random.poisson(8)  # Average 8 reco particles per event
        n_reco = min(n_reco, 50)  # Limit to array size
        reco_n[0] = n_reco
        
        for i in range(n_reco):
            # Generate random kinematic variables
            reco_pt[i] = np.random.exponential(25.0) + 5.0  # Higher pt threshold for reco
            reco_eta[i] = np.random.normal(0, 1.5)  # Slightly narrower eta for reco
            reco_phi[i] = np.random.uniform(-np.pi, np.pi)
            
            # Random object types
            reco_type[i] = np.random.choice([0, 1, 2, 3])  # electron, muon, photon, jet
        
        tree.Fill()
    
    # Write and close
    tree.Write()
    root_file.Close()
    logger.info(f"Created {n_events} events in {filename}")


def create_mock_xaod_converter():
    """
    Create a mock version of the conversion functions for testing.
    """
    from convert_xaod_to_h5 import get_particle_id, pad_or_truncate_event
    
    def extract_truth_particles_mock(tree, event_idx):
        """Mock truth particle extraction."""
        tree.GetEntry(event_idx)
        
        n_truth = tree.truth_n
        pt_list = []
        eta_list = []
        phi_list = []
        particle_id_list = []
        
        for i in range(n_truth):
            if tree.truth_status[i] == 1:  # Final state
                pt = tree.truth_pt[i]
                eta = tree.truth_eta[i]
                phi = tree.truth_phi[i]
                pdg_id = tree.truth_pdg_id[i]
                
                # Apply cuts
                if pt > 0.5 and abs(eta) < 5.0:
                    pt_list.append(pt)
                    eta_list.append(eta)
                    phi_list.append(phi)
                    particle_id_list.append(get_particle_id(pdg_id))
        
        return pt_list, eta_list, phi_list, particle_id_list
    
    def extract_reco_particles_mock(tree, event_idx):
        """Mock reconstructed particle extraction."""
        tree.GetEntry(event_idx)
        
        n_reco = tree.reco_n
        pt_list = []
        eta_list = []
        phi_list = []
        particle_id_list = []
        
        # Map reco types to particle IDs
        type_to_id = {0: 0, 1: 1, 2: 6, 3: 7}  # electron, muon, photon, jet->pion
        
        for i in range(n_reco):
            pt = tree.reco_pt[i]
            eta = tree.reco_eta[i]
            phi = tree.reco_phi[i]
            reco_type = tree.reco_type[i]
            
            # Apply cuts based on object type
            if reco_type == 0 and pt > 7.0 and abs(eta) < 2.47:  # Electron
                pass
            elif reco_type == 1 and pt > 6.0 and abs(eta) < 2.5:  # Muon
                pass
            elif reco_type == 2 and pt > 10.0 and abs(eta) < 2.37:  # Photon
                pass
            elif reco_type == 3 and pt > 20.0 and abs(eta) < 4.5:  # Jet
                pass
            else:
                continue
            
            pt_list.append(pt)
            eta_list.append(eta)
            phi_list.append(phi)
            particle_id_list.append(type_to_id[reco_type])
        
        return pt_list, eta_list, phi_list, particle_id_list
    
    return extract_truth_particles_mock, extract_reco_particles_mock


def test_conversion_pipeline():
    """Test the complete conversion pipeline."""
    logger.info("Testing conversion pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic ROOT file
        root_file = Path(temp_dir) / "test_data.root"
        h5_file = Path(temp_dir) / "test_output.h5"
        
        create_synthetic_root_file(str(root_file), n_events=20)
        
        # Test the conversion
        from convert_xaod_to_h5 import pad_or_truncate_event
        
        # Open ROOT file
        rf = ROOT.TFile.Open(str(root_file), "READ")
        tree = rf.Get("CollectionTree")
        
        # Get mock extraction functions
        extract_truth_mock, extract_reco_mock = create_mock_xaod_converter()
        
        # Process events
        all_pt = []
        all_eta = []
        all_phi = []
        all_particle_id = []
        all_is_truth = []
        
        max_particles = 50
        n_events = tree.GetEntries()
        
        for event_idx in range(n_events):
            # Extract particles
            truth_pt, truth_eta, truth_phi, truth_pid = extract_truth_mock(tree, event_idx)
            reco_pt, reco_eta, reco_phi, reco_pid = extract_reco_mock(tree, event_idx)
            
            # Combine
            combined_pt = truth_pt + reco_pt
            combined_eta = truth_eta + reco_eta
            combined_phi = truth_phi + reco_phi
            combined_pid = truth_pid + reco_pid
            combined_is_truth = [True] * len(truth_pt) + [False] * len(reco_pt)
            
            if len(combined_pt) == 0:
                continue
                
            # Pad/truncate
            pt_arr, eta_arr, phi_arr, pid_arr, truth_arr = pad_or_truncate_event(
                combined_pt, combined_eta, combined_phi, combined_pid, combined_is_truth, max_particles
            )
            
            all_pt.append(pt_arr)
            all_eta.append(eta_arr)
            all_phi.append(phi_arr)
            all_particle_id.append(pid_arr)
            all_is_truth.append(truth_arr)
        
        # Convert to numpy arrays
        all_pt = np.array(all_pt)
        all_eta = np.array(all_eta)
        all_phi = np.array(all_phi)
        all_particle_id = np.array(all_particle_id)
        all_is_truth = np.array(all_is_truth)
        
        # Save to HDF5
        with h5py.File(h5_file, 'w') as h5f:
            h5f.create_dataset('pt', data=all_pt)
            h5f.create_dataset('eta', data=all_eta)
            h5f.create_dataset('phi', data=all_phi)
            h5f.create_dataset('particle_id', data=all_particle_id)
            h5f.create_dataset('is_truth', data=all_is_truth)
            
            # Add metadata
            h5f.attrs['n_events'] = len(all_pt)
            h5f.attrs['max_particles'] = max_particles
        
        rf.Close()
        
        # Verify the output
        logger.info("Verifying output...")
        with h5py.File(h5_file, 'r') as h5f:
            logger.info(f"Output shape: {h5f['pt'].shape}")
            logger.info(f"Number of events: {h5f.attrs['n_events']}")
            logger.info(f"Max particles per event: {h5f.attrs['max_particles']}")
            
            # Check data ranges
            pt_data = h5f['pt'][:]
            eta_data = h5f['eta'][:]
            phi_data = h5f['phi'][:]
            pid_data = h5f['particle_id'][:]
            truth_data = h5f['is_truth'][:]
            
            logger.info(f"PT range: {pt_data.min():.2f} - {pt_data.max():.2f}")
            logger.info(f"Eta range: {eta_data.min():.2f} - {eta_data.max():.2f}")
            logger.info(f"Phi range: {phi_data.min():.2f} - {phi_data.max():.2f}")
            logger.info(f"Particle IDs: {np.unique(pid_data)}")
            logger.info(f"Truth particles: {truth_data.sum()}, Reco particles: {(~truth_data).sum()}")
            
            # Verify physics constraints
            assert np.all(pt_data >= 0), "PT should be non-negative"
            assert np.all(np.abs(phi_data) <= np.pi), "Phi should be in [-π, π]"
            assert np.all(pid_data >= 0) and np.all(pid_data <= 15), "Particle IDs should be 0-15"
            
            logger.info("✓ All physics constraints verified!")
            logger.info("✓ Conversion test completed successfully!")


def main():
    """Main test function."""
    logger.info("Starting conversion pipeline test...")
    
    try:
        test_conversion_pipeline()
        logger.info("🎉 All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 