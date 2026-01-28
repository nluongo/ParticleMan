#!/usr/bin/env python3
"""
Quick script to explore the mc20 ROOT file structure.
"""

import uproot
import awkward as ak
from pathlib import Path

# Find the mc20 file
root_files = list(Path(".").glob("mc20*.root"))
if not root_files:
    print("No mc20*.root file found!")
    exit(1)

filepath = root_files[0]
print(f"Exploring: {filepath}\n")

f = uproot.open(filepath)

print("=" * 70)
print("Keys in file (first 30):")
print("=" * 70)
for i, key in enumerate(f.keys()):
    if i >= 30:
        print(f"  ... and {len(f.keys()) - 30} more")
        break
    print(f"  {key}")

# Try to find trees
print("\n" + "=" * 70)
print("Looking for TTrees:")
print("=" * 70)

trees_found = []
for key in f.keys():
    try:
        obj = f[key]
        if hasattr(obj, 'num_entries'):
            name = key.split(";")[0]
            if name not in [t[0] for t in trees_found]:
                trees_found.append((name, obj.num_entries))
                print(f"  {name}: {obj.num_entries} entries")
    except:
        pass

if not trees_found:
    print("  No trees found!")
    exit(1)

# Explore the first tree
tree_name = trees_found[0][0]
print(f"\n" + "=" * 70)
print(f"Exploring tree: {tree_name}")
print("=" * 70)

tree = f[tree_name]
print(f"Number of entries: {tree.num_entries}")
print(f"Number of branches: {len(tree.keys())}")

# Look for pt, eta, phi branches
print("\n" + "-" * 70)
print("Branches containing 'pt' (case-insensitive):")
print("-" * 70)
pt_branches = [k for k in tree.keys() if 'pt' in k.lower()]
for b in pt_branches[:20]:
    try:
        branch = tree[b]
        print(f"  {b}")
        print(f"    type: {branch.typename}")
    except Exception as e:
        print(f"  {b}: error - {e}")

print("\n" + "-" * 70)
print("Branches containing 'eta' (case-insensitive):")
print("-" * 70)
eta_branches = [k for k in tree.keys() if 'eta' in k.lower()]
for b in eta_branches[:20]:
    try:
        branch = tree[b]
        print(f"  {b}")
        print(f"    type: {branch.typename}")
    except Exception as e:
        print(f"  {b}: error - {e}")

print("\n" + "-" * 70)
print("Branches containing 'phi' (case-insensitive):")
print("-" * 70)
phi_branches = [k for k in tree.keys() if 'phi' in k.lower()]
for b in phi_branches[:20]:
    try:
        branch = tree[b]
        print(f"  {b}")
        print(f"    type: {branch.typename}")
    except Exception as e:
        print(f"  {b}: error - {e}")

# Look for Analysis* branches which are common in PHYSLITE
print("\n" + "-" * 70)
print("Branches starting with 'Analysis':")
print("-" * 70)
analysis_branches = [k for k in tree.keys() if k.startswith('Analysis')]
for b in sorted(set(analysis_branches))[:30]:
    try:
        branch = tree[b]
        print(f"  {b}")
        print(f"    type: {branch.typename}")
    except Exception as e:
        print(f"  {b}: error - {e}")

# Sample some data from the first event
print("\n" + "=" * 70)
print("Sampling data from event 0:")
print("=" * 70)

# Find a pt branch to sample
for pt_branch in pt_branches[:5]:
    try:
        data = tree[pt_branch].array(entry_start=0, entry_stop=1, library="ak")
        flat = ak.to_list(data)
        if isinstance(flat, list) and len(flat) > 0:
            flat = flat[0]
        print(f"\n{pt_branch}:")
        print(f"  values: {flat[:10] if isinstance(flat, list) else flat}")
        if isinstance(flat, list) and len(flat) > 0:
            print(f"  n_values: {len(flat)}")
    except Exception as e:
        print(f"\n{pt_branch}: error - {e}")