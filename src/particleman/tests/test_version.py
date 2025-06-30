"""Test version information."""

import particleman


def test_version_exists() -> None:
    """Test that version is defined."""
    assert hasattr(particleman, "__version__")
    assert particleman.__version__ is not None
    assert isinstance(particleman.__version__, str)


def test_version_format() -> None:
    """Test that version follows semantic versioning format."""
    version = particleman.__version__
    parts = version.split(".")
    assert len(parts) >= 3, f"Version {version} should have at least 3 parts"
    
    # Check that first three parts are numbers
    for i, part in enumerate(parts[:3]):
        assert part.isdigit(), f"Version part {i+1} should be numeric, got {part}" 