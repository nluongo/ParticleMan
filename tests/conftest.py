"""Pytest configuration for ParticleMan tests."""

import sys

# Disable GaudiTesting plugin if present (produces verbose XML output)
# This needs to happen before pytest collects tests
collect_ignore_glob = []


def pytest_configure(config):
    """Configure pytest."""
    # Try to unregister GaudiTesting plugin if it's loaded
    try:
        plugin = config.pluginmanager.get_plugin("GaudiTesting")
        if plugin:
            config.pluginmanager.unregister(plugin)
    except Exception:
        pass
    
    # Also try lowercase variant
    try:
        plugin = config.pluginmanager.get_plugin("gauditesting")
        if plugin:
            config.pluginmanager.unregister(plugin)
    except Exception:
        pass