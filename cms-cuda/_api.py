"""
Public entry points with CPU / GPU dispatch.

``device="auto"`` (the default) runs on the GPU when CuPy is installed and a
CUDA device is visible, and otherwise falls back to the original NumPy code in
``_core.py``.  ``device="cpu"`` always runs the original code; ``device="cuda"``
(or ``"gpu"``) requires the GPU.
"""

import functools

from cms_cuda import _core


@functools.lru_cache(maxsize=1)
def gpu_available():
    '''True when CuPy can be imported and at least one CUDA device is visible.'''
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _use_gpu(device):
    d = 'auto' if device is None else str(device).lower()
    if d == 'cpu':
        return False
    if d in ('cuda', 'gpu'):
        if not gpu_available():
            raise RuntimeError(
                "device='%s' requested but no CUDA GPU is usable: install CuPy "
                "(pip install cupy-cuda13x, or cms-cuda[gpu]) and make sure "
                "a CUDA device is visible." % device)
        return True
    if d == 'auto':
        return gpu_available()
    raise ValueError("device must be 'auto', 'cpu', 'cuda' or 'gpu', got %r" % (device,))


def calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii, device="auto"):
    '''
    Main entrypoint into the code

    Calculate contact molecular surface of your binder against the target

    Contact molecular surface has units of A^2 and is a distance weighted surface area of your target

    Do not provide your own radii, you need to use the very specific radii raturned from get_radii_from_names()

    Parameters
    -------
    binder_xyz   : np.ndarray (N0, 3)
    binder_radii : np.ndarray (N0,)
    target_xyz   : np.ndarray (N1, 3)
    target_radii : np.ndarray (N1,)
    device       : "auto" (GPU if available), "cpu", or "cuda"/"gpu"


    Returns
    -------
    cms                 : float -- The contact molecular surface
    per_atom_target_cms : np.ndarray shape (N1,) — The per-atom contact_ms of your target molecule
    calc                : the calculator (use calc.calc_contact_molecular_surface(target_side=False)
                          for the binder side)
    '''
    if _use_gpu(device):
        from cms_cuda._gpu import calculate_contact_ms_gpu
        return calculate_contact_ms_gpu(binder_xyz, binder_radii, target_xyz, target_radii)
    return _core.calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii)


def calculate_shape_complementarity(binder_xyz, binder_radii, target_xyz, target_radii, device="auto"):
    '''
    Main entrypoint into the code

    Calculate the Lawrence & Coleman shape complementarity (SC) statistic between your binder and target

    Shape complementarity has no natural units and typically falls in [0, 1]; well-packed
    protein interfaces are usually ~0.5-0.75

    Per-atom shape complementarity doesn't make sense (SC is a whole-interface statistic), so unlike
    calculate_contact_ms this returns no per-atom array.

    Do not provide your own radii, you need to use the very specific radii raturned from get_radii_from_names()

    Parameters
    -------
    binder_xyz   : np.ndarray (N0, 3)
    binder_radii : np.ndarray (N0,)
    target_xyz   : np.ndarray (N1, 3)
    target_radii : np.ndarray (N1,)
    device       : "auto" (GPU if available), "cpu", or "cuda"/"gpu"


    Returns
    -------
    sc          : float -- The shape complementarity statistic
    sc_int_area : float -- Summed trimmed interface area of both molecules (A^2)
    median_dist : float -- Median interface separation distance (A)
    calc        : the calculator
    '''
    if _use_gpu(device):
        from cms_cuda._gpu import calculate_shape_complementarity_gpu
        return calculate_shape_complementarity_gpu(binder_xyz, binder_radii, target_xyz, target_radii)
    return _core.calculate_shape_complementarity(binder_xyz, binder_radii, target_xyz, target_radii)


def calculate_maximum_possible_contact_ms(xyz, radii, device="auto"):
    '''
    Main entrypoint into the code

    Calculate maximum possible contact molecular surface of a molecule. This is basically just the surface area

    Do not provide your own radii, you need to use the very specific radii raturned from get_radii_from_names()

    Parameters
    -------
    xyz    : np.ndarray (N0, 3)
    radii  : np.ndarray (N0,)
    device : "auto" (GPU if available), "cpu", or "cuda"/"gpu"


    Returns
    -------
    cms                 : float -- The maximum possible CMS
    per_atom_target_cms : np.ndarray shape (N1,) — The per-atom contact_ms of your target molecule
    calc                : the calculator
    '''
    if _use_gpu(device):
        from cms_cuda._gpu import calculate_maximum_possible_contact_ms_gpu
        return calculate_maximum_possible_contact_ms_gpu(xyz, radii)
    return _core.calculate_maximum_possible_contact_ms(xyz, radii)
