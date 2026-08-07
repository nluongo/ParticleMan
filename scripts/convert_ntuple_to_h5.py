#!/usr/bin/env python3
"""
Convert ATLAS flat ntuple ROOT files to HDF5 format for ParticleMan training.

Reads AnalysisMiniTree (or similar) flat ntuples using uproot — no PyROOT required.
For raw xAOD files (requiring PyROOT), see convert_xaod_to_h5.py instead.

Usage:
    python convert_ntuple_to_h5.py input.root output.h5
    python convert_ntuple_to_h5.py input.root output.h5 --max-events 1000
    python convert_ntuple_to_h5.py input.root output.h5 --list-branches

Requirements:
    uproot, awkward, h5py, numpy, tqdm (all in ParticleMan's pyproject.toml)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import h5py
import numpy as np
import uproot
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PARTICLE_ID_MAP = {
    11: 0, -11: 0,   # electron / positron
    13: 1, -13: 1,   # muon / antimuon
    15: 2, -15: 2,   # tau
    12: 3, -12: 3,   # electron neutrino
    14: 4, -14: 4,   # muon neutrino
    16: 5, -16: 5,   # tau neutrino
    22: 6,           # photon
    211: 7, -211: 7, # charged pion
    111: 8,          # neutral pion
    321: 9, -321: 9, # charged kaon
    130: 10, 310: 10, # neutral kaon
    2212: 11, -2212: 11, # proton
    2112: 12, -2112: 12, # neutron
    511: 13, -511: 13, 521: 13, -521: 13, 531: 13, -531: 13, # B mesons
    411: 14, -411: 14, 421: 14, -421: 14, 431: 14, -431: 14, # D mesons
}

UNKNOWN_PARTICLE_ID = 15
MAX_PARTICLES = 200

# Tree names to try in order
TREE_NAMES = ["AnalysisMiniTree", "CollectionTree", "Events", "tree", "ntuple"]

# Reco collections: (pt_branch, eta_branch, phi_branch, particle_id, pt_cut_gev, eta_cut)
RECO_COLLECTIONS = [
    ("el_pt_NOSYS",                       "el_eta",                       "el_phi",                       0,  7.0,  2.47),
    ("mu_pt_NOSYS",                        "mu_eta",                       "mu_phi",                       1,  6.0,  2.5),
    ("recojet_antikt4PFlow_pt_NOSYS",      "recojet_antikt4PFlow_eta",     "recojet_antikt4PFlow_phi",     7, 20.0,  4.5),
]

# Truth collections: (pt_branch, eta_branch, phi_branch, particle_id, pt_cut_gev, eta_cut)
TRUTH_COLLECTIONS = [
    ("truthjet_antikt4_pt", "truthjet_antikt4_eta", "truthjet_antikt4_phi", 7, 0.5, 5.0),
]


def open_tree(filepath: str) -> uproot.TTree:
    f = uproot.open(filepath)
    available = list(f.keys())
    for name in TREE_NAMES:
        if name in f or f"{name};1" in f:
            tree = f[name]
            logger.info(f"Using tree '{name}' ({tree.num_entries} events)")
            return tree
    raise ValueError(
        f"No recognised tree found in {filepath}. "
        f"Available keys: {[k for k in available if not k.startswith('CutBookkeeper')][:15]}"
    )


def _to_numpy(arr) -> np.ndarray:
    flat = ak.flatten(arr, axis=None)
    return ak.to_numpy(flat).astype(np.float64)


def _extract_collection(
    batch_event: ak.Array,
    pt_branch: str,
    eta_branch: str,
    phi_branch: str,
    particle_id: int,
    pt_cut_gev: float,
    eta_cut: float,
    available_branches: set,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Extract one collection from a single-event awkward record."""
    if pt_branch not in available_branches:
        return [], [], [], []

    pt_mev = _to_numpy(batch_event[pt_branch])
    eta = _to_numpy(batch_event[eta_branch])
    phi = _to_numpy(batch_event[phi_branch])
    pt_gev = pt_mev / 1000.0

    mask = (pt_gev > pt_cut_gev) & (np.abs(eta) < eta_cut)
    return (
        pt_gev[mask].tolist(),
        eta[mask].tolist(),
        phi[mask].tolist(),
        [particle_id] * int(mask.sum()),
    )


def pad_or_truncate(
    pt: List[float],
    eta: List[float],
    phi: List[float],
    pid: List[int],
    is_truth: List[bool],
    max_p: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(pt)
    if n > max_p:
        pt, eta, phi, pid, is_truth = pt[:max_p], eta[:max_p], phi[:max_p], pid[:max_p], is_truth[:max_p]
        n = max_p
    pad = max_p - n
    if pad:
        pt += [0.0] * pad
        eta += [0.0] * pad
        phi += [0.0] * pad
        pid += [UNKNOWN_PARTICLE_ID] * pad
        is_truth += [False] * pad
    return (
        np.array(pt, dtype=np.float32),
        np.array(eta, dtype=np.float32),
        np.array(phi, dtype=np.float32),
        np.array(pid, dtype=np.int32),
        np.array(is_truth, dtype=bool),
    )


def process_file(
    input_file: str,
    output_file: str,
    max_events: Optional[int],
    max_particles: int,
    chunk_size: int,
) -> None:
    logger.info(f"Processing {input_file} -> {output_file}")

    tree = open_tree(input_file)
    n_total = tree.num_entries
    if max_events:
        n_total = min(n_total, max_events)
    logger.info(f"Will process {n_total} events")

    available = set(tree.keys())

    # Determine which branches to read
    all_collections = TRUTH_COLLECTIONS + RECO_COLLECTIONS
    branches_needed = set()
    for pt_b, eta_b, phi_b, *_ in all_collections:
        if pt_b in available:
            branches_needed.update([pt_b, eta_b, phi_b])
        else:
            logger.warning(f"Branch '{pt_b}' not found — skipping that collection")

    if not branches_needed:
        raise RuntimeError("No recognised particle branches found in this file.")

    branch_list = sorted(branches_needed)

    all_pt, all_eta, all_phi, all_pid, all_is_truth, all_event_id = [], [], [], [], [], []
    event_offset = 0
    entries_left = n_total

    with tqdm(total=n_total, desc="Processing events") as pbar:
        batch_begin_i = 0
        for batch in tree.iterate(branch_list, step_size=chunk_size, entry_stop=n_total, library="ak"):
            n_batch = len(batch[branch_list[0]])
            if args.first_index and batch_begin_i < args.first_index:
                batch_begin_i += n_batch
                continue
            for i in range(n_batch):
                ev = {b: batch[b][i] for b in branch_list}

                # Truth side
                t_pt, t_eta, t_phi, t_pid = [], [], [], []
                for pt_b, eta_b, phi_b, pid, pt_cut, eta_cut in TRUTH_COLLECTIONS:
                    pp, ee, ff, ii = _extract_collection(ev, pt_b, eta_b, phi_b, pid, pt_cut, eta_cut, available)
                    t_pt += pp; t_eta += ee; t_phi += ff; t_pid += ii

                # Reco side
                r_pt, r_eta, r_phi, r_pid = [], [], [], []
                for pt_b, eta_b, phi_b, pid, pt_cut, eta_cut in RECO_COLLECTIONS:
                    pp, ee, ff, ii = _extract_collection(ev, pt_b, eta_b, phi_b, pid, pt_cut, eta_cut, available)
                    r_pt += pp; r_eta += ee; r_phi += ff; r_pid += ii

                # Trim each side to half max_particles before combining
                half = max_particles // 2
                t_pt, t_eta, t_phi, t_pid = t_pt[:half], t_eta[:half], t_phi[:half], t_pid[:half]
                r_pt, r_eta, r_phi, r_pid = r_pt[:half], r_eta[:half], r_phi[:half], r_pid[:half]

                combined_pt  = t_pt + r_pt
                combined_eta = t_eta + r_eta
                combined_phi = t_phi + r_phi
                combined_pid = t_pid + r_pid
                combined_is_truth = [True] * len(t_pt) + [False] * len(r_pt)

                if not combined_pt:
                    event_offset += 1
                    pbar.update(1)
                    continue

                pt_arr, eta_arr, phi_arr, pid_arr, truth_arr = pad_or_truncate(
                    combined_pt, combined_eta, combined_phi, combined_pid, combined_is_truth, max_particles
                )
                all_pt.append(pt_arr)
                all_eta.append(eta_arr)
                all_phi.append(phi_arr)
                all_pid.append(pid_arr)
                all_is_truth.append(truth_arr)
                all_event_id.append(event_offset)
                event_offset += 1
                pbar.update(1)

    logger.info(f"Processed {len(all_pt)} non-empty events out of {n_total}")

    all_pt       = np.array(all_pt)
    all_eta      = np.array(all_eta)
    all_phi      = np.array(all_phi)
    all_pid      = np.array(all_pid)
    all_is_truth = np.array(all_is_truth)
    all_event_id = np.array(all_event_id)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('pt',          data=all_pt,       compression='gzip')
        h5f.create_dataset('eta',         data=all_eta,      compression='gzip')
        h5f.create_dataset('phi',         data=all_phi,      compression='gzip')
        h5f.create_dataset('particle_id', data=all_pid,      compression='gzip')
        h5f.create_dataset('is_truth',    data=all_is_truth, compression='gzip')
        h5f.create_dataset('event_id',    data=all_event_id, compression='gzip')

        h5f.attrs['description']            = 'Particle physics data for transformer pre-training'
        h5f.attrs['source_file']            = input_file
        h5f.attrs['n_events']               = len(all_pt)
        h5f.attrs['max_particles_per_event'] = max_particles
        h5f.attrs['particle_id_map']        = str(PARTICLE_ID_MAP)
        h5f.attrs['unknown_particle_id']    = UNKNOWN_PARTICLE_ID

        h5f['pt'].attrs['description']          = 'Transverse momentum in GeV'
        h5f['eta'].attrs['description']         = 'Pseudorapidity'
        h5f['phi'].attrs['description']         = 'Azimuthal angle in radians'
        h5f['particle_id'].attrs['description'] = 'Categorical particle type ID (0-15)'
        h5f['is_truth'].attrs['description']    = 'True=truth jet, False=reconstructed object'
        h5f['event_id'].attrs['description']    = 'Original event index from ROOT file'

    logger.info(f"Saved {output_file} — shape {all_pt.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ATLAS flat ntuple ROOT files to HDF5 (uproot, no PyROOT required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_ntuple_to_h5.py data.root output.h5
    python convert_ntuple_to_h5.py data.root output.h5 --max-events 1000
    python convert_ntuple_to_h5.py data.root output.h5 --list-branches
        """,
    )
    parser.add_argument('input_file',  help='Input ROOT ntuple file')
    parser.add_argument('output_file', help='Output HDF5 file')
    parser.add_argument('--max-events',   type=int, default=None, help='Maximum events to process (default: all)')
    parser.add_argument('--first-index', type=int, default=None, help='First event to process, used when splitting file')
    parser.add_argument('--last-index', type=int, default=None, help='Last event to process, used when splitting file')
    parser.add_argument('--file-index', type=int, default=None, help='File index, used when splitting file')
    parser.add_argument('--max-particles', type=int, default=MAX_PARTICLES, help=f'Max particles per event (default: {MAX_PARTICLES})')
    parser.add_argument('--chunk-size',   type=int, default=1000, help='Events per uproot batch (default: 1000)')
    parser.add_argument('--list-branches', action='store_true', help='Print available branches and exit')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    if Path(args.output_file).exists():
        logger.info(f"Output file {args.output_file} already exists, skipping...")
        sys.exit(1)
    systematic_ids = ['AF3', 'hdamp', 'norew', 'PwH7', 'pthard']
    #if 'AF3' in args.input_file or 'hdamp' in args.input_file or 'norew' in args.input_file or 'PwH7' in args.input_file:
    if any([id in args.input_file for id in systematic_ids]):
        logger.info(f"Alternative systematic sample detected from file name, skipping...")
        sys.exit(1)

    if args.list_branches:
        tree = open_tree(args.input_file)
        for k in sorted(tree.keys()):
            print(k)
        return

    try:
        process_file(args.input_file, args.output_file, args.max_events, args.max_particles, args.chunk_size)
        logger.info("Conversion complete.")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
