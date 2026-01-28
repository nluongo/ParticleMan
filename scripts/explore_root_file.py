#!/usr/bin/env python3
"""
Explore the structure of a ROOT file to help create data configurations.

This script reads a ROOT file and displays information about its trees,
branches, and data types, making it easier to create column mappings
for the ParticleMan data loader.

Usage:
    python explore_root_file.py input.root [--tree TREE_NAME] [--filter PATTERN]

Examples:
    python explore_root_file.py data.root
    python explore_root_file.py data.root --tree CollectionTree
    python explore_root_file.py data.root --filter "*pt*"
    python explore_root_file.py data.root --filter "*Electron*" --sample 0
"""

import argparse
import fnmatch
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import uproot
    import awkward as ak
except ImportError:
    print("ERROR: uproot and awkward are required.")
    print("Install with: pip install uproot awkward")
    sys.exit(1)

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_trees(filepath: str) -> Dict[str, int]:
    """
    Get all trees in a ROOT file with their entry counts.

    Args:
        filepath: Path to ROOT file.

    Returns:
        Dictionary mapping tree names to number of entries.
    """
    f = uproot.open(filepath)
    trees = {}

    for key in f.keys():
        # Remove cycle number (e.g., "tree;1" -> "tree")
        name = key.split(";")[0]
        try:
            obj = f[key]
            if hasattr(obj, "num_entries"):
                trees[name] = obj.num_entries
        except Exception:
            pass

    return trees


def get_branch_info(tree: uproot.TTree, filter_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get information about all branches in a tree.

    Args:
        tree: uproot TTree object.
        filter_pattern: Optional glob pattern to filter branch names.

    Returns:
        List of dictionaries with branch information.
    """
    branches = []

    for name in tree.keys():
        # Apply filter if provided
        if filter_pattern and not fnmatch.fnmatch(name.lower(), filter_pattern.lower()):
            continue

        try:
            branch = tree[name]
            info = {
                "name": name,
                "typename": str(branch.typename) if hasattr(branch, "typename") else "unknown",
                "interpretation": str(branch.interpretation) if hasattr(branch, "interpretation") else "unknown",
            }

            # Determine if it's a vector/jagged type
            typename = info["typename"].lower()
            info["is_vector"] = "vector" in typename or "[]" in typename

            branches.append(info)
        except Exception as e:
            branches.append({"name": name, "error": str(e)})

    return branches


def sample_branch_data(
    tree: uproot.TTree, 
    branch_name: str, 
    event_idx: int = 0,
    max_values: int = 10
) -> Dict[str, Any]:
    """
    Sample data from a branch for a specific event.

    Args:
        tree: uproot TTree object.
        branch_name: Name of the branch.
        event_idx: Event index to sample.
        max_values: Maximum number of values to show.

    Returns:
        Dictionary with sample data.
    """
    try:
        data = tree[branch_name].array(
            entry_start=event_idx,
            entry_stop=event_idx + 1,
            library="ak"
        )

        # Convert to numpy/list for display
        flat = ak.to_list(data)
        if isinstance(flat, list) and len(flat) > 0:
            flat = flat[0]  # Get first event

        # Handle nested structures
        if isinstance(flat, list):
            n_values = len(flat)
            sample = flat[:max_values]
        else:
            n_values = 1
            sample = [flat]

        return {
            "n_values": n_values,
            "sample": sample,
            "dtype": str(type(sample[0]).__name__) if sample else "unknown",
        }
    except Exception as e:
        return {"error": str(e)}


def print_tree_structure(
    filepath: str,
    tree_name: Optional[str] = None,
    filter_pattern: Optional[str] = None,
    sample_event: Optional[int] = None,
    max_branches: int = 100,
) -> None:
    """
    Print the structure of a ROOT file.

    Args:
        filepath: Path to ROOT file.
        tree_name: Specific tree to examine (auto-detect if None).
        filter_pattern: Glob pattern to filter branches.
        sample_event: Event index to sample data from.
        max_branches: Maximum number of branches to display.
    """
    print(f"\n{'='*70}")
    print(f"ROOT File: {filepath}")
    print(f"{'='*70}\n")

    # Get trees
    trees = get_trees(filepath)
    if not trees:
        print("No trees found in file!")
        return

    print("📁 Trees in file:")
    for name, n_entries in trees.items():
        print(f"   {name}: {n_entries:,} entries")
    print()

    # Select tree to examine
    if tree_name is None:
        # Auto-select first tree
        tree_name = list(trees.keys())[0]
        print(f"Auto-selected tree: {tree_name}\n")

    if tree_name not in trees:
        print(f"Tree '{tree_name}' not found!")
        return

    # Open tree
    f = uproot.open(filepath)
    tree = f[tree_name]

    # Get branches
    branches = get_branch_info(tree, filter_pattern)

    if filter_pattern:
        print(f"🔍 Branches matching '{filter_pattern}': {len(branches)}")
    else:
        print(f"🌿 Total branches: {len(branches)}")

    if len(branches) > max_branches:
        print(f"   (showing first {max_branches}, use --filter to narrow down)")
        branches = branches[:max_branches]

    print()

    # Group branches by prefix for better readability
    groups: Dict[str, List[Dict]] = {}
    for branch in branches:
        name = branch["name"]
        # Extract prefix (e.g., "AnalysisJetsAuxDyn" from "AnalysisJetsAuxDyn.pt")
        if "." in name:
            prefix = name.split(".")[0]
        elif "_" in name:
            prefix = name.split("_")[0]
        else:
            prefix = "_other_"
        
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(branch)

    # Print grouped branches
    for prefix, group_branches in sorted(groups.items()):
        print(f"📦 {prefix}:")
        for branch in group_branches:
            name = branch["name"]
            if "error" in branch:
                print(f"   ❌ {name}: {branch['error']}")
                continue

            vector_flag = "[vector]" if branch.get("is_vector") else "[scalar]"
            typename = branch.get("typename", "unknown")
            
            # Shorten long type names
            if len(typename) > 40:
                typename = typename[:37] + "..."

            print(f"   {vector_flag:10} {name}")
            print(f"              └─ type: {typename}")

            # Sample data if requested
            if sample_event is not None:
                sample = sample_branch_data(tree, name, sample_event)
                if "error" not in sample:
                    print(f"              └─ event {sample_event}: {sample['n_values']} values, sample: {sample['sample']}")

        print()

    # Suggest configuration
    print("\n" + "="*70)
    print("💡 Configuration Suggestions:")
    print("="*70 + "\n")

    # Look for common patterns
    pt_candidates = [b["name"] for b in branches if "pt" in b["name"].lower()]
    eta_candidates = [b["name"] for b in branches if "eta" in b["name"].lower()]
    phi_candidates = [b["name"] for b in branches if "phi" in b["name"].lower()]

    if pt_candidates:
        print("Potential pT columns:")
        for c in pt_candidates[:5]:
            print(f"   - {c}")
        print()

    if eta_candidates:
        print("Potential eta columns:")
        for c in eta_candidates[:5]:
            print(f"   - {c}")
        print()

    if phi_candidates:
        print("Potential phi columns:")
        for c in phi_candidates[:5]:
            print(f"   - {c}")
        print()

    # Generate sample YAML snippet
    if pt_candidates and eta_candidates and phi_candidates:
        print("\nSample collection configuration:")
        print("-" * 40)
        prefix = pt_candidates[0].split(".")[0] if "." in pt_candidates[0] else "particles"
        print(f"""
collections:
  {prefix.lower().replace('auxdyn', '')}:
    enabled: true
    columns:
      pt: "{pt_candidates[0]}"
      eta: "{eta_candidates[0]}"
      phi: "{phi_candidates[0]}"
    fixed_particle_id: 0  # Set appropriate ID
    is_vector: true  # Set based on branch type
""")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Explore ROOT file structure for ParticleMan configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python explore_root_file.py data.root
    python explore_root_file.py data.root --tree CollectionTree
    python explore_root_file.py data.root --filter "*Jets*"
    python explore_root_file.py data.root --filter "*pt*" --sample 0
        """
    )

    parser.add_argument("input_file", help="Input ROOT file")
    parser.add_argument("--tree", "-t", default=None, help="Tree name to examine")
    parser.add_argument("--filter", "-f", default=None, help="Glob pattern to filter branches")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Event index to sample data from")
    parser.add_argument("--max-branches", "-m", type=int, default=100, help="Maximum branches to display")

    args = parser.parse_args()

    # Validate input
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Run exploration
    print_tree_structure(
        args.input_file,
        tree_name=args.tree,
        filter_pattern=args.filter,
        sample_event=args.sample,
        max_branches=args.max_branches,
    )


if __name__ == "__main__":
    main()