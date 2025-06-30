"""Basic tests for ParticleMan package that don't require heavy dependencies."""

import particleman


def test_package_import() -> None:
    """Test that the package can be imported."""
    assert particleman is not None


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


def test_version_matches() -> None:
    """Test that VERSION constant matches __version__."""
    assert particleman.VERSION == particleman.__version__


def test_package_attributes() -> None:
    """Test that expected package attributes exist."""
    assert hasattr(particleman, "__author__")
    assert hasattr(particleman, "__email__")
    assert hasattr(particleman, "__version__")
    assert hasattr(particleman, "VERSION")
    
    # These should be strings
    assert isinstance(particleman.__author__, str)
    assert isinstance(particleman.__email__, str)


def test_version_is_development() -> None:
    """Test that we're using a development version."""
    version = particleman.__version__
    # Development versions should start with 0
    assert version.startswith("0."), f"Expected development version starting with 0, got {version}" 