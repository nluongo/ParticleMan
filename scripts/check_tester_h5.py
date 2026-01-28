#!/usr/bin/env python3
"""
Explore the existing tester.h5 file to understand the data structure.
Run this script to see the output.
"""

import h5py
import numpy as np

print("Exploring tester.h5...\n")

with h5py.File("tester.h5", "r") as f:
    print("File attributes:")
    for key, val in f.attrs.items():
        print(f"  {key}: {val}")
    
    print("\nDatasets:")
    for key in f.keys():
        ds = f[key]
        print(f"  {key}:")
        print(f"    shape: {ds.shape}")
        print(f"    dtype: {ds.dtype}")
        if ds.attrs:
            for attr_key, attr_val in ds.attrs.items():
                print(f"    @{attr_key}: {attr_val}")
    
    print("\nSample from event 0:")
    print(f"  pt[:10]: {f['pt'][0][:10]}")
    print(f"  eta[:10]: {f['eta'][0][:10]}")
    print(f"  phi[:10]: {f['phi'][0][:10]}")
    print(f"  particle_id[:10]: {f['particle_id'][0][:10]}")
    if 'is_truth' in f:
        print(f"  is_truth[:10]: {f['is_truth'][0][:10]}")
    
    # Count non-zero particles
    pt = f['pt'][0]
    n_nonzero = np.count_nonzero(pt)
    print(f"\n  Non-zero pt values in event 0: {n_nonzero}")