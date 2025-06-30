#!/usr/bin/env python3
"""
Convert ROOT xAOD files to HDF5 format for ParticleMan training.

This script processes ATLAS xAOD files and extracts both truth and reconstructed
particles, saving them in a format suitable for transformer-based pre-training.

Usage:
    python convert_xaod_to_h5.py input.root output.h5 [--max-events 1000]

Requirements:
    - ROOT with PyROOT
    - h5py
    - numpy
    - tqdm (for progress bars)

Author: ParticleMan Framework
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Try to import ROOT, provide helpful error if not available
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)  # Suppress ROOT graphics
except ImportError:
    print("ERROR: ROOT with PyROOT is required but not found.")
    print("Please install ROOT with Python bindings:")
    print("  conda install -c conda-forge root")
    print("  or follow instructions at: https://root.cern/install/")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Particle ID mappings based on PDG codes
# These IDs will be used as categorical inputs to the transformer
PARTICLE_ID_MAP = {
    # Leptons (0-5)
    11: 0,   # electron
    -11: 0,  # positron (treat same as electron)
    13: 1,   # muon
    -13: 1,  # antimuon (treat same as muon)
    15: 2,   # tau
    -15: 2,  # antitau (treat same as tau)
    
    # Neutrinos (3-5) - usually not reconstructed but may appear in truth
    12: 3,   # electron neutrino
    -12: 3,  # electron antineutrino
    14: 4,   # muon neutrino
    -14: 4,  # muon antineutrino
    16: 5,   # tau neutrino
    -16: 5,  # tau antineutrino
    
    # Photons (6)
    22: 6,   # photon
    
    # Light mesons (7-9)
    211: 7,  # charged pion
    -211: 7, # charged pion (opposite charge)
    111: 8,  # neutral pion
    321: 9,  # charged kaon
    -321: 9, # charged kaon (opposite charge)
    130: 10, # neutral kaon (K_L)
    310: 10, # neutral kaon (K_S)
    
    # Protons and neutrons (11-12)
    2212: 11, # proton
    -2212: 11, # antiproton
    2112: 12, # neutron
    -2112: 12, # antineutron
    
    # Heavy mesons and baryons (13-15)
    # B mesons
    511: 13,  # B0
    -511: 13, # anti-B0
    521: 13,  # B+
    -521: 13, # B-
    531: 13,  # B_s
    -531: 13, # anti-B_s
    
    # D mesons  
    411: 14,  # D+
    -411: 14, # D-
    421: 14,  # D0
    -421: 14, # anti-D0
    431: 14,  # D_s+
    -431: 14, # D_s-
    
    # Other particles (15)
    # All other particles get mapped to category 15
}

# Default ID for unknown particles
UNKNOWN_PARTICLE_ID = 15

# Maximum number of particles per event (for padding/truncation)
MAX_PARTICLES_PER_EVENT = 200


def get_particle_id(pdg_id: int) -> int:
    """
    Map PDG particle ID to our categorical particle ID.
    
    Args:
        pdg_id: PDG particle identifier
        
    Returns:
        Categorical particle ID (0-15)
    """
    return PARTICLE_ID_MAP.get(abs(pdg_id), UNKNOWN_PARTICLE_ID)


def print_particle_id_mapping() -> None:
    """Print the particle ID mapping for reference."""
    logger.info("Particle ID Mapping:")
    logger.info("  0: Electrons/Positrons (PDG: ±11)")
    logger.info("  1: Muons/Antimuons (PDG: ±13)")
    logger.info("  2: Taus/Antitaus (PDG: ±15)")
    logger.info("  3: Electron Neutrinos (PDG: ±12)")
    logger.info("  4: Muon Neutrinos (PDG: ±14)")
    logger.info("  5: Tau Neutrinos (PDG: ±16)")
    logger.info("  6: Photons (PDG: 22)")
    logger.info("  7: Charged Pions (PDG: ±211)")
    logger.info("  8: Neutral Pions (PDG: 111)")
    logger.info("  9: Charged Kaons (PDG: ±321)")
    logger.info(" 10: Neutral Kaons (PDG: 130, 310)")
    logger.info(" 11: Protons/Antiprotons (PDG: ±2212)")
    logger.info(" 12: Neutrons/Antineutrons (PDG: ±2112)")
    logger.info(" 13: B Mesons (PDG: ±511, ±521, ±531)")
    logger.info(" 14: D Mesons (PDG: ±411, ±421, ±431)")
    logger.info(" 15: Other/Unknown Particles")


def is_final_state_particle(status_code: int) -> bool:
    """
    Check if a truth particle is in the final state.
    
    Different generators use different status codes:
    - Pythia8: status 1 = final state
    - Herwig: status 1 = final state  
    - Some use status 23 for final state before hadronization
    
    Args:
        status_code: HepMC status code
        
    Returns:
        True if particle is final state
    """
    # Common final state status codes
    final_state_codes = {1, 23}
    return status_code in final_state_codes


def extract_truth_particles(tree: ROOT.TTree, event_idx: int) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    Extract final state truth particles from an xAOD event.
    
    Args:
        tree: ROOT TTree containing the event
        event_idx: Event index to process
        
    Returns:
        Tuple of (pt_list, eta_list, phi_list, particle_id_list)
    """
    tree.GetEntry(event_idx)
    
    pt_list = []
    eta_list = []
    phi_list = []
    particle_id_list = []
    
    # Access truth particles container
    # Note: Container names may vary depending on xAOD version
    truth_containers = ["TruthParticles", "TruthParticle", "xAOD::TruthParticleContainer"]
    
    truth_particles = None
    for container_name in truth_containers:
        if hasattr(tree, container_name):
            truth_particles = getattr(tree, container_name)
            break
    
    if truth_particles is None:
        logger.warning(f"No truth particle container found in event {event_idx}")
        return pt_list, eta_list, phi_list, particle_id_list
    
    # Iterate through truth particles
    n_particles = truth_particles.size() if hasattr(truth_particles, 'size') else len(truth_particles)
    
    for i in range(n_particles):
        particle = truth_particles.at(i) if hasattr(truth_particles, 'at') else truth_particles[i]
        
        # Check if particle is final state
        status = particle.status() if hasattr(particle, 'status') else particle.auxdata("status")
        if not is_final_state_particle(status):
            continue
            
        # Get particle properties
        pt = particle.pt() / 1000.0  # Convert MeV to GeV
        eta = particle.eta()
        phi = particle.phi()
        pdg_id = particle.pdgId() if hasattr(particle, 'pdgId') else particle.auxdata("pdgId")
        
        # Apply basic quality cuts
        if pt < 0.5:  # Minimum pt cut of 500 MeV
            continue
        if abs(eta) > 5.0:  # Maximum eta cut
            continue
            
        # Convert PDG ID to our categorical ID
        particle_id = get_particle_id(pdg_id)
        
        pt_list.append(pt)
        eta_list.append(eta)
        phi_list.append(phi)
        particle_id_list.append(particle_id)
    
    return pt_list, eta_list, phi_list, particle_id_list


def extract_reco_particles(tree: ROOT.TTree, event_idx: int) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    Extract reconstructed particles from an xAOD event.
    
    Args:
        tree: ROOT TTree containing the event
        event_idx: Event index to process
        
    Returns:
        Tuple of (pt_list, eta_list, phi_list, particle_id_list)
    """
    tree.GetEntry(event_idx)
    
    pt_list = []
    eta_list = []
    phi_list = []
    particle_id_list = []
    
    # Process different reconstructed object types
    
    # 1. Electrons
    electrons = getattr(tree, "Electrons", None) or getattr(tree, "ElectronCollection", None)
    if electrons:
        n_electrons = electrons.size() if hasattr(electrons, 'size') else len(electrons)
        for i in range(n_electrons):
            electron = electrons.at(i) if hasattr(electrons, 'at') else electrons[i]
            
            # Apply electron selection cuts
            pt = electron.pt() / 1000.0  # Convert MeV to GeV
            if pt < 7.0:  # Minimum electron pt
                continue
            if abs(electron.eta()) > 2.47:  # Electron eta acceptance
                continue
                
            pt_list.append(pt)
            eta_list.append(electron.eta())
            phi_list.append(electron.phi())
            particle_id_list.append(0)  # Electron ID
    
    # 2. Muons
    muons = getattr(tree, "Muons", None) or getattr(tree, "MuonCollection", None)
    if muons:
        n_muons = muons.size() if hasattr(muons, 'size') else len(muons)
        for i in range(n_muons):
            muon = muons.at(i) if hasattr(muons, 'at') else muons[i]
            
            # Apply muon selection cuts
            pt = muon.pt() / 1000.0
            if pt < 6.0:  # Minimum muon pt
                continue
            if abs(muon.eta()) > 2.5:  # Muon eta acceptance
                continue
                
            pt_list.append(pt)
            eta_list.append(muon.eta())
            phi_list.append(muon.phi())
            particle_id_list.append(1)  # Muon ID
    
    # 3. Photons
    photons = getattr(tree, "Photons", None) or getattr(tree, "PhotonCollection", None)
    if photons:
        n_photons = photons.size() if hasattr(photons, 'size') else len(photons)
        for i in range(n_photons):
            photon = photons.at(i) if hasattr(photons, 'at') else photons[i]
            
            # Apply photon selection cuts
            pt = photon.pt() / 1000.0
            if pt < 10.0:  # Minimum photon pt
                continue
            if abs(photon.eta()) > 2.37:  # Photon eta acceptance
                continue
                
            pt_list.append(pt)
            eta_list.append(photon.eta())
            phi_list.append(photon.phi())
            particle_id_list.append(6)  # Photon ID
    
    # 4. Jets (represent hadrons)
    jets = getattr(tree, "AntiKt4EMTopoJets", None) or getattr(tree, "Jets", None)
    if jets:
        n_jets = jets.size() if hasattr(jets, 'size') else len(jets)
        for i in range(n_jets):
            jet = jets.at(i) if hasattr(jets, 'at') else jets[i]
            
            # Apply jet selection cuts
            pt = jet.pt() / 1000.0
            if pt < 20.0:  # Minimum jet pt
                continue
            if abs(jet.eta()) > 4.5:  # Jet eta acceptance
                continue
                
            pt_list.append(pt)
            eta_list.append(jet.eta())
            phi_list.append(jet.phi())
            particle_id_list.append(7)  # Treat jets as charged pions for now
    
    return pt_list, eta_list, phi_list, particle_id_list


def pad_or_truncate_event(
    pt_list: List[float], 
    eta_list: List[float], 
    phi_list: List[float], 
    particle_id_list: List[int],
    is_truth_list: List[bool],
    max_particles: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad or truncate particle lists to fixed length.
    
    Args:
        pt_list, eta_list, phi_list, particle_id_list, is_truth_list: Particle data
        max_particles: Maximum number of particles per event
        
    Returns:
        Tuple of padded numpy arrays
    """
    n_particles = len(pt_list)
    
    # Truncate if too many particles
    if n_particles > max_particles:
        pt_list = pt_list[:max_particles]
        eta_list = eta_list[:max_particles]
        phi_list = phi_list[:max_particles]
        particle_id_list = particle_id_list[:max_particles]
        is_truth_list = is_truth_list[:max_particles]
        n_particles = max_particles
    
    # Pad if too few particles
    padding_size = max_particles - n_particles
    if padding_size > 0:
        pt_list.extend([0.0] * padding_size)
        eta_list.extend([0.0] * padding_size)
        phi_list.extend([0.0] * padding_size)
        particle_id_list.extend([UNKNOWN_PARTICLE_ID] * padding_size)  # Padding token
        is_truth_list.extend([False] * padding_size)
    
    return (
        np.array(pt_list, dtype=np.float32),
        np.array(eta_list, dtype=np.float32),
        np.array(phi_list, dtype=np.float32),
        np.array(particle_id_list, dtype=np.int32),
        np.array(is_truth_list, dtype=bool)
    )


def process_xaod_file(
    input_file: str, 
    output_file: str, 
    max_events: Optional[int] = None,
    max_particles: int = MAX_PARTICLES_PER_EVENT
) -> None:
    """
    Process an xAOD ROOT file and convert to HDF5.
    
    Args:
        input_file: Path to input ROOT file
        output_file: Path to output HDF5 file
        max_events: Maximum number of events to process (None for all)
        max_particles: Maximum particles per event
    """
    logger.info(f"Processing {input_file} -> {output_file}")
    print_particle_id_mapping()
    
    # Open ROOT file
    try:
        root_file = ROOT.TFile.Open(input_file, "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Cannot open ROOT file: {input_file}")
        
        # Get the main tree (common names in xAOD)
        tree_names = ["CollectionTree", "physics", "nominal"]
        tree = None
        for name in tree_names:
            tree = root_file.Get(name)
            if tree:
                break
        
        if not tree:
            raise RuntimeError(f"Cannot find main tree in {input_file}")
            
        n_events = tree.GetEntries()
        if max_events:
            n_events = min(n_events, max_events)
            
        logger.info(f"Processing {n_events} events")
        
        # Prepare data storage
        all_pt = []
        all_eta = []
        all_phi = []
        all_particle_id = []
        all_is_truth = []
        all_event_id = []
        
        # Process events
        for event_idx in tqdm(range(n_events), desc="Processing events"):
            try:
                # Extract truth particles
                truth_pt, truth_eta, truth_phi, truth_particle_id = extract_truth_particles(tree, event_idx)
                truth_is_truth = [True] * len(truth_pt)
                
                # Extract reconstructed particles
                reco_pt, reco_eta, reco_phi, reco_particle_id = extract_reco_particles(tree, event_idx)
                reco_is_truth = [False] * len(reco_pt)
                
                # Combine truth and reco particles
                combined_pt = truth_pt + reco_pt
                combined_eta = truth_eta + reco_eta
                combined_phi = truth_phi + reco_phi
                combined_particle_id = truth_particle_id + reco_particle_id
                combined_is_truth = truth_is_truth + reco_is_truth
                
                # Skip events with no particles
                if len(combined_pt) == 0:
                    continue
                
                # Pad or truncate to fixed length
                pt_array, eta_array, phi_array, particle_id_array, is_truth_array = pad_or_truncate_event(
                    combined_pt, combined_eta, combined_phi, combined_particle_id, combined_is_truth, max_particles
                )
                
                # Store event data
                all_pt.append(pt_array)
                all_eta.append(eta_array)
                all_phi.append(phi_array)
                all_particle_id.append(particle_id_array)
                all_is_truth.append(is_truth_array)
                all_event_id.append(event_idx)
                
            except Exception as e:
                logger.warning(f"Error processing event {event_idx}: {e}")
                continue
        
        # Convert to numpy arrays
        all_pt = np.array(all_pt)
        all_eta = np.array(all_eta)
        all_phi = np.array(all_phi)
        all_particle_id = np.array(all_particle_id)
        all_is_truth = np.array(all_is_truth)
        all_event_id = np.array(all_event_id)
        
        logger.info(f"Processed {len(all_pt)} events successfully")
        logger.info(f"Final data shape: {all_pt.shape}")
        
        # Save to HDF5
        with h5py.File(output_file, 'w') as h5f:
            # Create datasets
            h5f.create_dataset('pt', data=all_pt, compression='gzip')
            h5f.create_dataset('eta', data=all_eta, compression='gzip')
            h5f.create_dataset('phi', data=all_phi, compression='gzip')
            h5f.create_dataset('particle_id', data=all_particle_id, compression='gzip')
            h5f.create_dataset('is_truth', data=all_is_truth, compression='gzip')
            h5f.create_dataset('event_id', data=all_event_id, compression='gzip')
            
            # Add metadata
            h5f.attrs['description'] = 'Particle physics data for transformer pre-training'
            h5f.attrs['source_file'] = input_file
            h5f.attrs['n_events'] = len(all_pt)
            h5f.attrs['max_particles_per_event'] = max_particles
            h5f.attrs['particle_id_map'] = str(PARTICLE_ID_MAP)
            h5f.attrs['unknown_particle_id'] = UNKNOWN_PARTICLE_ID
            
            # Add dataset descriptions
            h5f['pt'].attrs['description'] = 'Transverse momentum in GeV (always positive)'
            h5f['eta'].attrs['description'] = 'Pseudorapidity (can be positive or negative)'
            h5f['phi'].attrs['description'] = 'Azimuthal angle in radians'
            h5f['particle_id'].attrs['description'] = 'Categorical particle type ID (0-15)'
            h5f['is_truth'].attrs['description'] = 'Boolean flag: True=truth particle, False=reconstructed'
            h5f['event_id'].attrs['description'] = 'Original event index from ROOT file'
        
        logger.info(f"Successfully saved {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise
    finally:
        if 'root_file' in locals():
            root_file.Close()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert ROOT xAOD files to HDF5 format for ParticleMan training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_xaod_to_h5.py data.root output.h5
    python convert_xaod_to_h5.py data.root output.h5 --max-events 1000 --max-particles 150
        """
    )
    
    parser.add_argument('input_file', help='Input ROOT xAOD file')
    parser.add_argument('output_file', help='Output HDF5 file')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (default: all)')
    parser.add_argument('--max-particles', type=int, default=MAX_PARTICLES_PER_EVENT,
                       help=f'Maximum particles per event (default: {MAX_PARTICLES_PER_EVENT})')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    try:
        process_xaod_file(
            args.input_file, 
            args.output_file, 
            args.max_events,
            args.max_particles
        )
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 