"""
cms-cuda — GPU-accelerated contact molecular surface (CMS) and shape
complementarity (SC) for protein design.

Built on bcov77/py_contact_ms (Brian Coventry), the Python port of Longxing
Cao's C++ contact molecular surface; SC after Lawrence & Colman (1993). The
original NumPy code is included unchanged in ``cms_cuda._core``.

Runs on an NVIDIA GPU through CuPy when available (``device="auto"``), and on
the CPU with the original NumPy implementation otherwise.
"""

__version__ = "0.2.0"

from cms_cuda._core import (
    get_radii_from_names,
    partition_pose,
    MolecularSurfaceCalculator,
)
from cms_cuda._api import (
    calculate_contact_ms,
    calculate_shape_complementarity,
    calculate_maximum_possible_contact_ms,
    gpu_available,
)

__all__ = [
    "calculate_contact_ms",
    "calculate_shape_complementarity",
    "get_radii_from_names",
    "calculate_maximum_possible_contact_ms",
    "partition_pose",
    "MolecularSurfaceCalculator",
    "MolecularSurfaceCalculatorGPU",
    "gpu_available",
]


def __getattr__(name):
    # Imported lazily so CPU-only installs never need CuPy.
    if name == "MolecularSurfaceCalculatorGPU":
        from cms_cuda._gpu import MolecularSurfaceCalculatorGPU
        return MolecularSurfaceCalculatorGPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
