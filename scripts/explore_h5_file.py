#!/usr/bin/env python3
"""
Explore the structure of an HDF5 file to help create data configurations.

This script reads an HDF5 file and displays information about its datasets,
shapes, and data types, making it easier to create column mappings
for the ParticleMan data loader.

Usage:
    python explore_h5_file.py input.h5 [--filter PATTERN] [--sample EVENT_IDX]

Examples:
    python explore_h5_file.py data.h5
    python explore_h5_file.py data.h5 --filter "*pt*"
    python explore_h5_file.py data.h5 --sample 0
"""

import argparse
import fnmatch
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_datasets(
    group: h5py.Group, 
    prefix: str = "",
    filter_pattern: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Recursively collect all datasets in an HDF5 group.

    Args:
        group: HDF5 group to explore.
        prefix: Current path prefix.
        filter_pattern: Optional glob pattern to filter dataset names.

    Returns:
        List of dictionaries with dataset information.
    """
    datasets = []

    for key in group.keys():
        full_path = f"{prefix}/{key}" if prefix else key
        item = group[key]

        if isinstance(item, h5py.Dataset):
            # Apply filter if provided
            if filter_pattern and not fnmatch.fnmatch(full_path.lower(), filter_pattern.lower()):
                continue

            info = {
                "name": full_path,
                "shape": item.shape,
                "dtype": str(item.dtype),
                "size": item.size,
                "ndim": item.ndim,
                "attrs": dict(item.attrs) if item.attrs else {},
            }

            # Add compression info if available
            if item.compression:
                info["compression"] = item.compression

            datasets.append(info)

        elif isinstance(item, h5py.Group):
            # Recurse into subgroups
            datasets.extend(collect_datasets(item, full_path, filter_pattern))

    return datasets


def sample_dataset(
    f: h5py.File,
    dataset_name: str,
    event_idx: int = 0,
    max_values: int = 10,
) -> Dict[str, Any]:
    """
    Sample data from a dataset for a specific event.

    Args:
        f: Open HDF5 file.
        dataset_name: Name of the dataset.
        event_idx: Event index to sample.
        max_values: Maximum number of values to show.

    Returns:
        Dictionary with sample data.
    """
    try:
        dataset = f[dataset_name]
        
        if dataset.ndim == 0:
            # Scalar dataset
            return {"value": float(dataset[()])}
        elif dataset.ndim == 1:
            # 1D dataset - single value per event
            if event_idx < len(dataset):
                value = dataset[event_idx]
                return {
                    "value": float(value) if np.isscalar(value) else value.tolist(),
                    "n_events": len(dataset),
                }
        elif dataset.ndim == 2:
            # 2D dataset - (n_events, n_particles) or similar
            if event_idx < dataset.shape[0]:
                row = dataset[event_idx]
                non_zero = np.count_nonzero(row)
                return {
                    "n_values": len(row),
                    "non_zero": non_zero,
                    "sample": row[:max_values].tolist(),
                    "min": float(np.min(row)),
                    "max": float(np.max(row)),
                    "mean": float(np.mean(row[row != 0])) if non_zero > 0 else 0.0,
                }
        else:
            # Higher dimensional
            return {
                "shape": dataset.shape,
                "sample_shape": dataset[event_idx].shape if event_idx < dataset.shape[0] else None,
            }

        return {"error": f"Event index {event_idx} out of range"}

    except Exception as e:
        return {"error": str(e)}


def print_file_structure(
    filepath: str,
    filter_pattern: Optional[str] = None,
    sample_event: Optional[int] = None,
) -> None:
    """
    Print the structure of an HDF5 file.

    Args:
        filepath: Path to HDF5 file.
        filter_pattern: Glob pattern to filter datasets.
        sample_event: Event index to sample data from.
    """
    print(f"\n{'='*70}")
    print(f"HDF5 File: {filepath}")
    print(f"{'='*70}\n")

    with h5py.File(filepath, "r") as f:
        # Print file attributes
        if f.attrs:
            print("📋 File Attributes:")
            for key, value in f.attrs.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 60:
                    value_str = value_str[:57] + "..."
                print(f"   {key}: {value_str}")
            print()

        # Collect all datasets
        datasets = collect_datasets(f, filter_pattern=filter_pattern)

        if filter_pattern:
            print(f"🔍 Datasets matching '{filter_pattern}': {len(datasets)}")
        else:
            print(f"📊 Total datasets: {len(datasets)}")
        print()

        if not datasets:
            print("No datasets found!")
            return

        # Group by prefix/directory
        groups: Dict[str, List[Dict]] = {}
        for ds in datasets:
            name = ds["name"]
            if "/" in name:
                prefix = name.rsplit("/", 1)[0]
            else:
                prefix = "_root_"
            
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(ds)

        # Print datasets
        for prefix, group_datasets in sorted(groups.items()):
            if prefix != "_root_":
                print(f"📁 {prefix}/")
            else:
                print("📁 Root level:")

            for ds in group_datasets:
                name = ds["name"]
                short_name = name.rsplit("/", 1)[-1] if "/" in name else name
                
                shape_str = str(ds["shape"])
                dtype_str = ds["dtype"]
                
                print(f"   📈 {short_name}")
                print(f"      ├─ shape: {shape_str}")
                print(f"      ├─ dtype: {dtype_str}")
                
                if ds.get("compression"):
                    print(f"      ├─ compression: {ds['compression']}")
                
                if ds.get("attrs"):
                    for attr_key, attr_val in list(ds["attrs"].items())[:3]:
                        print(f"      ├─ @{attr_key}: {attr_val}")

                # Sample data if requested
                if sample_event is not None:
                    sample = sample_dataset(f, name, sample_event)
                    if "error" not in sample:
                        if "sample" in sample:
                            print(f"      └─ event {sample_event}: {sample['non_zero']}/{sample['n_values']} non-zero")
                            print(f"         values: {sample['sample']}")
                            print(f"         range: [{sample['min']:.3f}, {sample['max']:.3f}], mean: {sample['mean']:.3f}")
                        elif "value" in sample:
                            print(f"      └─ event {sample_event}: {sample['value']}")
                    else:
                        print(f"      └─ sample error: {sample['error']}")

            print()

        # Analyze structure for configuration suggestions
        print("\n" + "="*70)
        print("💡 Configuration Suggestions:")
        print("="*70 + "\n")

        # Look for common column patterns
        names = [ds["name"].lower() for ds in datasets]
        
        pt_found = [ds["name"] for ds in datasets if "pt" in ds["name"].lower()]
        eta_found = [ds["name"] for ds in datasets if "eta" in ds["name"].lower()]
        phi_found = [ds["name"] for ds in datasets if "phi" in ds["name"].lower()]
        pid_found = [ds["name"] for ds in datasets if "particle_id" in ds["name"].lower() or "pdg" in ds["name"].lower()]

        if pt_found:
            print("Potential pT columns:")
            for c in pt_found:
                ds = next(d for d in datasets if d["name"] == c)
                print(f"   - {c} (shape: {ds['shape']}, dtype: {ds['dtype']})")
            print()

        if eta_found:
            print("Potential eta columns:")
            for c in eta_found:
                ds = next(d for d in datasets if d["name"] == c)
                print(f"   - {c} (shape: {ds['shape']}, dtype: {ds['dtype']})")
            print()

        if phi_found:
            print("Potential phi columns:")
            for c in phi_found:
                ds = next(d for d in datasets if d["name"] == c)
                print(f"   - {c} (shape: {ds['shape']}, dtype: {ds['dtype']})")
            print()

        if pid_found:
            print("Potential particle_id columns:")
            for c in pid_found:
                ds = next(d for d in datasets if d["name"] == c)
                print(f"   - {c} (shape: {ds['shape']}, dtype: {ds['dtype']})")
            print()

        # Infer if data is already preprocessed (2D with fixed size)
        if pt_found and len(datasets[0]["shape"]) == 2:
            n_events, max_particles = datasets[0]["shape"]
            print(f"Data appears to be preprocessed with:")
            print(f"   - {n_events} events")
            print(f"   - {max_particles} max particles per event")
            print()

        # Generate sample YAML
        if pt_found and eta_found and phi_found:
            print("Sample configuration:")
            print("-" * 40)
            pid_col = pid_found[0] if pid_found else None
            
            if pid_col:
                pid_line = f'      particle_id: "{pid_col}"'
                fixed_line = "    # fixed_particle_id: null  # Using column"
            else:
                pid_line = "    # No particle_id column found"
                fixed_line = "    fixed_particle_id: 0  # Set appropriate ID"

            print(f"""
source:
  type: "hdf5"
  files:
    - "{filepath}"

collections:
  particles:
    enabled: true
    columns:
      pt: "{pt_found[0]}"
      eta: "{eta_found[0]}"
      phi: "{phi_found[0]}"
{pid_line}
{fixed_line}
    is_vector: false  # Set to true if variable-length
""")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Explore HDF5 file structure for ParticleMan configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python explore_h5_file.py data.h5
    python explore_h5_file.py data.h5 --filter "*pt*"
    python explore_h5_file.py data.h5 --sample 0
        """
    )

    parser.add_argument("input_file", help="Input HDF5 file")
    parser.add_argument("--filter", "-f", default=None, help="Glob pattern to filter datasets")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Event index to sample data from")

    args = parser.parse_args()

    # Validate input
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Run exploration
    print_file_structure(
        args.input_file,
        filter_pattern=args.filter,
        sample_event=args.sample,
    )


if __name__ == "__main__":
    main()