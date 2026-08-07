#!/usr/bin/env python3
"""
Run the data loading tests.

This script runs pytest on the data loading test suite.
Run from the project root directory.
"""

import subprocess
import sys


def main():
    print("Running ParticleMan data loading tests...")
    print("=" * 60)
    
    # Run pytest on the data loading tests
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_data_loading.py",
            "-v",
            "--tb=short",
        ],
        cwd=".",
    )
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())