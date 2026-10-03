import ctypes
import os
import platform
import sys

import numpy as np


def _get_lib_path():
    """根据平台和架构返回共享库路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    platform_map = {
        "linux": "linux",
        "win32": "win32",
        "darwin": "darwin",
    }
    arch_map = {
        "x86_64": "x86_64",
        "AMD64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }

    current_platform = platform_map.get(sys.platform)
    machine = platform.machine()
    current_arch = arch_map.get(machine, machine)

    if sys.platform == "darwin" and current_arch == "arm64":
        current_arch = "universal2"

    if sys.platform.startswith("win"):
        lib_name = "libtracking.dll"
    elif sys.platform == "darwin":
        lib_name = "libtracking.dylib"
    else:
        lib_name = "libtracking.so"

    lib_path = os.path.join(base_dir, "libs", current_platform, current_arch, lib_name)
    return lib_path


_lib_path = _get_lib_path()
_lib = ctypes.CDLL(_lib_path)

# void* matching_create(int N, int M, int fps, double x0, double y0, double dx, double dy)
_lib.matching_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.matching_create.restype = ctypes.c_void_p

# void matching_destroy(void* ptr)
_lib.matching_destroy.argtypes = [ctypes.c_void_p]
_lib.matching_destroy.restype = None

# void matching_init(void* ptr, double* data, int n)
_lib.matching_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
_lib.matching_init.restype = None

# void matching_run(void* ptr)
_lib.matching_run.argtypes = [ctypes.c_void_p]
_lib.matching_run.restype = None

# void matching_get_flow(void* ptr, double* ox, double* oy, double* cx, double* cy, double* occupied)
_lib.matching_get_flow.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_lib.matching_get_flow.restype = None


class Matching:
    """Marker tracking matcher, API-compatible with the original pybind11 version."""

    def __init__(self, N_=8, M_=8, fps_=30, x0_=80.0, y0_=15.0, dx=21.0, dy=21.0):
        self._N = N_
        self._M = M_
        self._ptr = _lib.matching_create(N_, M_, fps_, x0_, y0_, dx, dy)
        if not self._ptr:
            raise RuntimeError("Failed to create Matching instance")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.matching_destroy(self._ptr)
            self._ptr = None

    def init(self, input_data):
        """
        Args:
            input_data: list[list[float]] or ndarray of shape (N, 2)
        """
        arr = np.asarray(input_data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"input_data must be of shape (N, 2), got {arr.shape}")
        n = arr.shape[0]
        _lib.matching_init(self._ptr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)

    def run(self):
        _lib.matching_run(self._ptr)

    def test(self):
        """Not implemented in ctypes version, kept for API compatibility."""
        return (21.0, 21.0)

    def get_flow(self):
        n = self._N * self._M
        ox = np.zeros(n, dtype=np.float64)
        oy = np.zeros(n, dtype=np.float64)
        cx = np.zeros(n, dtype=np.float64)
        cy = np.zeros(n, dtype=np.float64)
        occ = np.zeros(n, dtype=np.float64)

        _lib.matching_get_flow(
            self._ptr,
            ox.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            oy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            occ.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        shape = (self._N, self._M)
        return (
            ox.reshape(shape).tolist(),
            oy.reshape(shape).tolist(),
            cx.reshape(shape).tolist(),
            cy.reshape(shape).tolist(),
            occ.reshape(shape).tolist(),
        )


tracking = type(sys)("tracking")
tracking.Matching = Matching
sys.modules[__name__ + ".tracking"] = tracking
