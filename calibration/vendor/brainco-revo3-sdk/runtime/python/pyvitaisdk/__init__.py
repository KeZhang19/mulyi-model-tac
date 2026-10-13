#!/usr/bin/env python3
# coding=utf-8
'''
Description  : 
'''

import sys
from pathlib import Path


def _enable_source_checkout_modules() -> None:
    """Make bundled module sources importable from an uninstalled checkout."""
    repository_root = Path(__file__).resolve().parent.parent
    module_sources = (
        repository_root / "modules" / "vitai-module-pre-process" / "src",
        repository_root / "modules" / "vitai-module-depth" / "src",
        repository_root / "modules" / "vitai-module-marker" / "src",
        repository_root / "modules" / "vitai-module-slip" / "src",
        repository_root / "modules" / "vitai-module-force" / "src",
    )
    for source in module_sources:
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))


_enable_source_checkout_modules()


__author__ = "ViTai Robotics"
__all__ = [
    # Device and Config
    "VTSDeviceFinder", "VTSDeviceConfig",
    # Sensors
    "GF225", "GF225VideoStreamProfile", "GF225OutputProfile",
    "VTSensor", "VTSensorType",
    # Data Types
    "VTSDataType", "VTSlipState",
    # Exceptions - Only expose base class for external users
    "VTSError",
    # Version
    "__version__"
]

from .devices import VTSDeviceFinder, VTSDeviceConfig
from .vts import (GF225, GF225OutputProfile, GF225VideoStreamProfile, VTSensor, VTSensorType, VTSDataType, VTSlipState)

# Only import and expose the base exception class
from .common.exceptions import VTSError

# All specific exceptions are available internally via:
# from pyvitaisdk.common.exceptions import DeviceNotFoundError, etc.

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("pyvitaisdk")
    except PackageNotFoundError:
        try:
            import setuptools_scm
            __version__ = setuptools_scm.get_version()
        except Exception:
            __version__ = "not found"
except ImportError:
    # Python < 3.8 fallback
    import pkg_resources
    try:
        __version__ = pkg_resources.get_distribution("pyvitaisdk").version
    except pkg_resources.DistributionNotFound:
        try:
            import setuptools_scm
            __version__ = setuptools_scm.get_version()
        except Exception:
            __version__ = "not found"
