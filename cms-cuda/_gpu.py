"""
GPU (CUDA, via CuPy) implementation of contact molecular surface and shape
complementarity.

``MolecularSurfaceCalculatorGPU`` mirrors ``_core.MolecularSurfaceCalculator``
method for method: the same settings, the same stages, the same formulas and
thresholds, float64 throughout.  Differences are confined to *how* arrays are
evaluated:

* the large broadcast collision / distance tests and the neighbour-list build
  run as fused kernels (``_gpu_kernels``) instead of materialising
  (points x neighbours x 3) or (N x N) intermediates;
* the few Python loops of the CPU version (per-atom neighbour fill, toroid
  queue appends) are replaced by array operations;
* NumPy semantics that CuPy does not share (``x**2`` rounding, einsum
  summation order, duplicate-index ``|=``) are reproduced explicitly.

Once the surface has been generated (by ``CalcLoaded``, ``CalcLoadedSC`` or
``CalcLoadedMaxPossibleCMS``) the final state is copied into the same host
containers the CPU calculator uses, so ``calc.run`` can be inspected exactly
like the CPU one.
"""

import math
import threading

import numpy as np
import cupy as cp
import cupyx
from cupyx.profiler import time_range

from cms_cuda._core import (
    RESULTS, AtomArray, DotArray, ProbeArray, SimpleNeighborArray,
    MolecularSurfaceCalculator,
    ATTEN_BLOCKER, ATTEN_2, ATTEN_BURIED_FLAGGED, ATTEN_6, MAX_SUBDIV,
)
from cms_cuda import _gpu_kernels as K
from cms_cuda._gpu_kernels import (
    dot3_seq, dot3_einsum, sumsq3, norm3, cross3, ORDER_SEQ, ORDER_ALT,
)


def calculate_contact_ms_gpu(binder_xyz, binder_radii, target_xyz, target_radii):
    '''GPU version of _core.calculate_contact_ms (same arguments and returns).'''
    calc = MolecularSurfaceCalculatorGPU()
    calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)
    cms, per_atom_target_cms = calc.CalcLoaded()

    return cms, per_atom_target_cms, calc


def calculate_shape_complementarity_gpu(binder_xyz, binder_radii, target_xyz, target_radii):
    '''GPU version of _core.calculate_shape_complementarity (same arguments and returns).'''
    calc = MolecularSurfaceCalculatorGPU()
    calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)
    sc, sc_int_area, median_dist = calc.CalcLoadedSC()

    return sc, sc_int_area, median_dist, calc


def calculate_maximum_possible_contact_ms_gpu(xyz, radii):
    '''GPU version of _core.calculate_maximum_possible_contact_ms.'''
    calc = MolecularSurfaceCalculatorGPU()
    calc.AddMolecule(0, xyz, radii)
    cms, per_atom_target_cms = calc.CalcLoadedMaxPossibleCMS()

    return cms, per_atom_target_cms, calc


_thread_local = threading.local()


def _thread_stream():
    '''
    One non-blocking CUDA stream per host thread, shared by the calculators
    created on that thread.  Reusing the stream lets CuPy's (per-stream) memory
    pool recycle device memory between structures instead of calling cudaMalloc;
    separate threads still get separate streams and can run concurrently.
    '''
    stream = getattr(_thread_local, 'stream', None)
    if stream is None:
        stream = _thread_local.stream = cp.cuda.Stream(non_blocking=True)
    return stream


def _normalize3(a):
    """a / np.linalg.norm(a, axis=-1, keepdims=True)"""
    return a / norm3(a)[..., None]


# ── device-side containers ──────────────────────────────────────────────────

class _DevAtoms:
    """Device copy of the finalized host AtomArray (struct of arrays)."""

    def __init__(self, host):
        n = len(host)
        self.n        = n
        self.xyz      = cp.asarray(np.ascontiguousarray(host.xyz[:n]), dtype=cp.float64)
        self.radius   = cp.asarray(host.radius[:n], dtype=cp.float64)
        self.density  = cp.asarray(host.density[:n], dtype=cp.float64)
        self.molecule = cp.asarray(host.molecule[:n], dtype=cp.int8)
        self.atten    = cp.asarray(host.atten[:n], dtype=cp.int8)
        self.access   = cp.asarray(host.access[:n], dtype=cp.int8)
        self.natom    = cp.asarray(host.natom[:n], dtype=cp.int32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return _AtomView(self, idx)


class _AtomView:
    """
    Rows `idx` of _DevAtoms, the GPU analogue of ``AtomArray[idx]``.  Fields
    are gathered lazily on first access (gathers are copies, like the CPU).
    """

    _FIELDS = ('xyz', 'radius', 'density', 'molecule', 'atten', 'access', 'natom')

    def __init__(self, base, idx):
        self._base = base
        self._idx  = cp.asarray(idx, dtype=cp.int64).reshape(-1)
        self._cache = {}

    def __len__(self):
        return int(self._idx.shape[0])

    def __getattr__(self, name):
        if name in _AtomView._FIELDS:
            cache = self.__dict__['_cache']
            if name not in cache:
                cache[name] = getattr(self.__dict__['_base'], name)[self.__dict__['_idx']]
            return cache[name]
        raise AttributeError(name)

    def __getitem__(self, idx):
        return _AtomView(self._base, self._idx[cp.asarray(idx)])


class _DevNeighbors:
    """Device SimpleNeighborArray: padded (N, K) rows, NaN / -1 padding."""

    def __init__(self, n, k):
        self.xyz        = cp.full((n, k, 3), cp.nan, dtype=cp.float64)
        self.radius     = cp.full((n, k), cp.nan, dtype=cp.float64)
        self.natom      = cp.full((n, k), -1, dtype=cp.int32)
        self.nneighbors = cp.zeros(n, dtype=cp.int32)

    def to_host(self):
        n, k = self.natom.shape
        host = SimpleNeighborArray.__new__(SimpleNeighborArray)
        host._cap = n
        host._max_neighbors = k
        host.xyz        = cp.asnumpy(self.xyz)
        host.radius     = cp.asnumpy(self.radius)
        host.natom      = cp.asnumpy(self.natom)
        host.nneighbors = cp.asnumpy(self.nneighbors)
        return host


class _DevDots:
    """
    Device DotArray.  Batches are appended in call order and concatenated
    once by finalize(), which yields the same ordering as DotArray.extend().
    """

    _FIELDS = ('coor_xyz', 'outnml_xyz', 'area', 'buried', 'type_', 'atom_idx')
    _DTYPES = (cp.float64, cp.float64, cp.float64, cp.int8, cp.int8, cp.int32)

    def __init__(self):
        self._batches = []
        self._n = 0
        for f, dt in zip(self._FIELDS, self._DTYPES):
            shape = (0, 3) if f.endswith('_xyz') else (0,)
            setattr(self, f, cp.zeros(shape, dtype=dt))

    def extend(self, coor_xyz, outnml_xyz, area, buried, type_, atom_idx):
        n = int(coor_xyz.shape[0])
        if n == 0:
            return
        self._batches.append((coor_xyz, outnml_xyz, area, buried, type_, atom_idx))
        self._n += n

    def __len__(self):  return self._n
    def __bool__(self): return self._n > 0

    def finalize(self):
        if not self._batches:
            return
        for j, (f, dt) in enumerate(zip(self._FIELDS, self._DTYPES)):
            parts = [getattr(self, f)] + [b[j] for b in self._batches]
            setattr(self, f, cp.concatenate(parts).astype(dt, copy=False))
        self._batches = []

    def to_host(self):
        host = DotArray()
        n = self._n
        host._n = n
        host._cap = n
        for f in self._FIELDS:
            setattr(host, f, cp.asnumpy(getattr(self, f)))
        return host


class _DevProbes:
    """Device ProbeArray (batched like _DevDots)."""

    _FIELDS = ('atom_idx_0', 'atom_idx_1', 'atom_idx_2', 'height', 'point_xyz', 'alt_xyz')
    _DTYPES = (cp.int32, cp.int32, cp.int32, cp.float64, cp.float64, cp.float64)

    def __init__(self):
        self._batches = []
        self._n = 0
        for f, dt in zip(self._FIELDS, self._DTYPES):
            shape = (0, 3) if f.endswith('_xyz') else (0,)
            setattr(self, f, cp.zeros(shape, dtype=dt))

    def extend_from_arrays(self, atom_idx_0, atom_idx_1, atom_idx_2, height, point_xyz, alt_xyz):
        n = int(height.shape[0])
        if n == 0:
            return
        self._batches.append((atom_idx_0, atom_idx_1, atom_idx_2, height, point_xyz, alt_xyz))
        self._n += n

    def __len__(self):  return self._n
    def __bool__(self): return self._n > 0

    def finalize(self):
        if not self._batches:
            return
        for j, (f, dt) in enumerate(zip(self._FIELDS, self._DTYPES)):
            parts = [getattr(self, f)] + [b[j] for b in self._batches]
            setattr(self, f, cp.concatenate(parts).astype(dt, copy=False))
        self._batches = []

    def to_host(self):
        host = ProbeArray()
        n = self._n
        host._n = n
        host._cap = n
        for f in self._FIELDS:
            setattr(host, f, cp.asnumpy(getattr(self, f)))
        return host


class _ToroidQueue:
    """The toroid queue as device arrays (the CPU keeps a list of tuples)."""

    def __init__(self):
        self.natom1  = cp.zeros(0, dtype=cp.int32)
        self.natom2  = cp.zeros(0, dtype=cp.int32)
        self.uij     = cp.zeros((0, 3), dtype=cp.float64)
        self.tij     = cp.zeros((0, 3), dtype=cp.float64)
        self.rij     = cp.zeros(0, dtype=cp.float64)
        self.between = cp.zeros(0, dtype=bool)

    def __len__(self):
        return int(self.natom1.shape[0])

    def to_host_list(self):
        n1, n2 = cp.asnumpy(self.natom1), cp.asnumpy(self.natom2)
        uij, tij = cp.asnumpy(self.uij), cp.asnumpy(self.tij)
        rij, bet = cp.asnumpy(self.rij), cp.asnumpy(self.between)
        return [(int(n1[i]), int(n2[i]), uij[i], tij[i], float(rij[i]), bool(bet[i]))
                for i in range(len(n1))]


class _GPURun:
    """
    Same attributes as the CPU calculator's ``run`` object.  ``toroid_queue``
    is materialised as the CPU's list of tuples only when first accessed.
    """

    def __init__(self):
        self.radmax       = 0.0
        self.results      = RESULTS()
        self.atoms        = AtomArray()
        self.dots         = [DotArray(), DotArray()]
        self.trimmed_dots = [DotArray(), DotArray()]
        self.probes       = ProbeArray()
        self.neighbor_array = None
        self.buried_array   = None
        self.surfaces_generated = False
        self.surfaces_generated_all_atoms = None
        self._toroid_dev  = None
        self._toroid_list = []

    @property
    def toroid_queue(self):
        if self._toroid_dev is not None:
            self._toroid_list = self._toroid_dev.to_host_list()
            self._toroid_dev = None
        return self._toroid_list


class _DevState:
    def __init__(self):
        self.atoms  = None
        self.dots   = [_DevDots(), _DevDots()]
        self.probes = _DevProbes()
        self.neighbor_array = None
        self.buried_array   = None
        self.toroid_queue   = _ToroidQueue()


# ── calculator ──────────────────────────────────────────────────────────────

class MolecularSurfaceCalculatorGPU:

    def __init__(self):

        self.settings = type("Settings", (), {})()
        self.settings.rp = 1.7
        self.settings.density = 15.0
        self.settings.band = 1.5
        self.settings.sep = 8.0
        self.settings.weight = 0.5
        self.settings.binwidth_dist = 0.02
        self.settings.binwidth_norm = 0.02

        # All GPU work of this calculator is queued on a non-default stream
        self._stream = _thread_stream()

        self.reset()

    def reset(self):
        self.run  = _GPURun()
        self._dev = _DevState()

    # ── entry points ───────────────────────────────────────────────────────

    def CalcLoaded(self):
        self.run.results.valid = 0
        self._ensure_surfaces_generated(all_atoms=False)

        cms_return = self.calc_contact_molecular_surface(target_side=True)

        return cms_return

    def CalcLoadedMaxPossibleCMS(self):
        self.run.results.valid = 0
        self._ensure_surfaces_generated(all_atoms=True)

        cms_return = self.calc_max_possible_contact_molecular_surface(target_side=True)

        return cms_return

    def CalcLoadedSC(self):
        '''
        Compute the Lawrence & Coleman shape complementarity statistic for the loaded molecules.

        Trims each molecule's dot cloud down to its buried "core" (discarding buried dots
        within settings.band of an accessible dot) and computes nearest-neighbor distance /
        normal-vector statistics between the two trimmed surfaces. Never mutates
        self.run.dots[0]/[1] -- trimmed results are stored as fresh DotArray copies in
        self.run.trimmed_dots[0]/[1]. Surface generation itself is shared/cached
        (see _ensure_surfaces_generated) with CalcLoaded()/calc_contact_molecular_surface(),
        so calling CalcLoaded() and CalcLoadedSC() (either order) on the same instance is
        safe and only generates the surface once.

        Returns
        -------
        sc          : float
        sc_int_area : float
        median_dist : float
        '''
        self.run.results.valid = 0
        self._ensure_surfaces_generated(all_atoms=False)

        with self._stream, time_range('CalcLoadedSC'):
            trimmed_dev = [None, None]
            for i in (0, 1):
                dots = self._dev.dots[i]
                area, keep_mask = self.trim_peripheral_band_vectorized(dots)

                keep = cp.nonzero(keep_mask)[0]
                trimmed_dev[i] = _DevDots()
                trimmed_dev[i].extend(
                    dots.coor_xyz[keep], dots.outnml_xyz[keep], dots.area[keep],
                    dots.buried[keep], dots.type_[keep], dots.atom_idx[keep],
                )
                trimmed_dev[i].finalize()

                # host copy, built exactly like the CPU version builds it
                host = self.run.dots[i]
                keep_h = cp.asnumpy(keep_mask)
                trimmed = DotArray()
                trimmed.extend(
                    host.coor_xyz[keep_h], host.outnml_xyz[keep_h], host.area[keep_h],
                    host.buried[keep_h], host.type_[keep_h], host.atom_idx[keep_h],
                )
                trimmed.finalize()
                self.run.trimmed_dots[i] = trimmed

                surf = self.run.results.surface[i]
                surf.trimmedArea = area
                surf.nTrimmedDots = int(keep_h.sum())
                surf.nAllDots = len(dots)

            self.calc_neighbor_distance_vectorized(0, trimmed_dev[0], trimmed_dev[1])
            self.calc_neighbor_distance_vectorized(1, trimmed_dev[1], trimmed_dev[0])

        s0, s1, s2 = self.run.results.surface
        s2.d_mean = (s0.d_mean + s1.d_mean) / 2
        s2.d_median = (s0.d_median + s1.d_median) / 2
        s2.s_mean = (s0.s_mean + s1.s_mean) / 2
        s2.s_median = (s0.s_median + s1.s_median) / 2
        s2.nAllDots = s0.nAllDots + s1.nAllDots
        s2.nTrimmedDots = s0.nTrimmedDots + s1.nTrimmedDots
        s2.trimmedArea = s0.trimmedArea + s1.trimmedArea

        self.run.results.sc = s2.s_median
        self.run.results.distance = s2.d_median
        self.run.results.area = s2.trimmedArea
        self.run.results.valid = 1

        return self.run.results.sc, self.run.results.area, self.run.results.distance

    def _ensure_surfaces_generated(self, all_atoms=False):
        """
        Idempotent, cached surface generation (same contract as the CPU version).

        The first call runs the atoms/attention/surface-generation pipeline on
        the GPU and mirrors the result into self.run; later calls with the same
        `all_atoms` mode reuse it.  Reusing it with a different mode raises,
        because the attention-number assignments are incompatible.
        """
        if self.run.surfaces_generated:
            if self.run.surfaces_generated_all_atoms != all_atoms:
                raise RuntimeError(
                    "This MolecularSurfaceCalculator instance already generated its "
                    f"molecular surface with all_atoms={self.run.surfaces_generated_all_atoms}, "
                    f"but was asked to reuse it with all_atoms={all_atoms}. These use "
                    "incompatible attention-number assignments (CalcLoadedMaxPossibleCMS's "
                    "all_atoms=True vs. CalcLoaded()/CalcLoadedSC()'s all_atoms=False), so "
                    "the cached surface can't be safely reused for both. Call self.reset() "
                    "and re-add your atoms, or use a fresh MolecularSurfaceCalculator()."
                )
            return

        assert len(self.run.atoms) > 0

        with self._stream, time_range('generate_surfaces'):
            # Trim atom arrays to true size, then move them to the device
            self.run.atoms.finalize()
            self._dev.atoms = _DevAtoms(self.run.atoms)

            self.assign_attention_numbers(self._dev.atoms, all_atoms=all_atoms)

            self.generate_molecular_surfaces()

            # Trim dot / probe arrays after surface generation
            self._dev.dots[0].finalize()
            self._dev.dots[1].finalize()

            self._copy_run_to_host()

        self.run.surfaces_generated = True
        self.run.surfaces_generated_all_atoms = all_atoms

    def generate_molecular_surfaces(self):

        assert len(self.run.atoms) > 0

        self.calc_dots_for_all_atoms(self._dev.atoms)

    def add_binder_and_target(self, binder_xyz, binder_radii, target_xyz, target_radii):
        self.reset()
        self.AddMolecule(0, binder_xyz, binder_radii)
        self.AddMolecule(1, target_xyz, target_radii)

    def AddMolecule(self, molecule, xyz, radii):
        """
        Load atoms for one molecule from numpy arrays (host side; the atoms are
        uploaded to the GPU when the calculation starts).

        Parameters
        ----------
        molecule : int          0 or 1
        xyz      : np.ndarray   shape (N, 3) — atom coordinates
        radii    : np.ndarray   shape (N,)   — atom radii; atoms with radius
                                               <= 0 contribute zero surface
                                               area but are kept so per-atom
                                               outputs stay positionally
                                               aligned with N
        """
        xyz   = cp.asnumpy(xyz) if isinstance(xyz, cp.ndarray) else xyz
        radii = cp.asnumpy(radii) if isinstance(radii, cp.ndarray) else radii
        mol_val = 1 if molecule == 1 else 0
        xyz_f   = np.asarray(xyz)
        radii_f = np.asarray(radii)
        n = len(radii_f)
        self.run.atoms.extend_from_arrays(xyz_f, radii_f, self.settings.density, mol_val)
        self.run.results.surface[mol_val].nAtoms += n
        self.run.results.nAtoms += n

    def _copy_run_to_host(self):
        """Mirror the generated surface into the CPU containers of self.run."""
        dev, run = self._dev, self.run
        n = len(run.atoms)
        run.atoms.atten[:n]  = cp.asnumpy(dev.atoms.atten)
        run.atoms.access[:n] = cp.asnumpy(dev.atoms.access)
        for m in (0, 1):
            run.dots[m] = dev.dots[m].to_host()
        run.probes = dev.probes.to_host()
        run.neighbor_array = dev.neighbor_array.to_host()
        run.buried_array   = dev.buried_array.to_host()
        run._toroid_dev  = dev.toroid_queue
        run._toroid_list = []

    # ── CMS from dots ──────────────────────────────────────────────────────

    def calc_contact_molecular_surface(self, target_side=True):
        """
        Compute the contact molecular surface.

        For each buried dot on the *query* molecule the nearest buried dot on
        the *reference* molecule is found; the weighted area sum gives the CMS.

        target_side=True  (default): query=mol1, reference=mol0.
            Returns per-atom values sized (n_mol1,).
        target_side=False           : query=mol0, reference=mol1.
            Returns per-atom values sized (n_mol0,).

        Returns
        -------
        total_cms    : float
        per_atom_cms : np.ndarray shape (n_query_atoms,)
        """
        with self._stream, time_range('calc_contact_molecular_surface'):
            n      = len(self.run.atoms)
            mol    = self.run.atoms.molecule[:n]
            n_mol1 = int((mol == 1).sum())
            n_mol0 = n - n_mol1

            if target_side:
                dots_q, dots_r = self._dev.dots[1], self._dev.dots[0]
                n_q            = n_mol1
                atom_offset    = n_mol0      # mol1 atoms start here in the global array
            else:
                dots_q, dots_r = self._dev.dots[0], self._dev.dots[1]
                n_q            = n_mol0
                atom_offset    = 0           # mol0 atoms start at index 0

            zero_ret = (0.0, np.zeros(n_q, dtype=np.float64))

            if len(dots_r) == 0:
                return zero_ret

            buried_q = dots_q.buried.astype(bool)
            buried_r = dots_r.buried.astype(bool)

            xyz_q  = dots_q.coor_xyz[buried_q]                              # (Kq, 3)
            xyz_r  = dots_r.coor_xyz[buried_r]                              # (Kr, 3)

            if xyz_q.shape[0] == 0 or xyz_r.shape[0] == 0:
                return zero_ret

            area_q = dots_q.area[buried_q]                                  # (Kq,)

            min_dist_sq = K.min_sqdist(xyz_q, xyz_r)                        # (Kq,)

            per_dot_cms = area_q * cp.exp(-min_dist_sq * self.settings.weight)  # (Kq,)

            local_idx    = dots_q.atom_idx[buried_q] - atom_offset          # (Kq,)
            per_atom_cms = cp.asnumpy(K.bincount_weights(local_idx, per_dot_cms, n_q))

            total_cms = float(per_atom_cms.sum())
            return total_cms, per_atom_cms

    def calc_max_possible_contact_molecular_surface(self, target_side=True):
        """
        Compute the maximum possible contact molecular surface.

        Returns the total surface area of molecule 0 and per-atom contributions,
        with no distance weighting.  Only molecule 0 needs to be loaded.

        Returns
        -------
        total_area    : float
        per_atom_area : np.ndarray shape (n_mol0,)
        """
        with self._stream, time_range('calc_max_possible_contact_molecular_surface'):
            n      = len(self.run.atoms)
            mol    = self.run.atoms.molecule[:n]
            n_mol0 = int((mol == 0).sum())

            dots_0   = self._dev.dots[0]
            zero_ret = (0.0, np.zeros(n_mol0, dtype=np.float64))

            if len(dots_0) == 0:
                return zero_ret

            per_atom_area = cp.asnumpy(
                K.bincount_weights(dots_0.atom_idx, dots_0.area, n_mol0))

            total_area = float(per_atom_area.sum())
            return total_area, per_atom_area

    # ── shape complementarity from dots ────────────────────────────────────

    def trim_peripheral_band_vectorized(self, dots):
        """
        TrimPeripheralBand on the GPU.

        For a single molecule's dots, keep only the buried dots that have no accessible
        (non-buried) dot within settings.band of them.  Read-only: never modifies `dots`.

        Parameters
        ----------
        dots : _DevDots

        Returns
        -------
        trimmed_area : float
        keep_mask    : cupy bool array shape (len(dots),)
        """
        n = len(dots)
        buried = dots.buried.astype(bool)
        buried_idx = cp.nonzero(buried)[0]

        if buried_idx.size == 0:
            return 0.0, cp.zeros(n, dtype=bool)

        if buried_idx.size == n:
            # Nothing to compare against, so nothing can be "near the edge" -- keep everything buried
            keep_mask = buried.copy()
        else:
            xyz_buried = dots.coor_xyz[buried_idx]
            xyz_accessible = dots.coor_xyz[~buried]

            # cdist(xyz_buried, xyz_accessible, 'sqeuclidean').min(axis=1)
            min_dist_sq = K.min_sqdist(xyz_buried, xyz_accessible)
            keep_buried = min_dist_sq > (self.settings.band ** 2)

            keep_mask = cp.zeros(n, dtype=bool)
            keep_mask[buried_idx[keep_buried]] = True

        # summed on the host with NumPy, the same summation the CPU uses
        trimmed_area = float(cp.asnumpy(dots.area[keep_mask]).sum())
        return trimmed_area, keep_mask

    # Grouped-data median: pure NumPy on a small host array, shared verbatim
    # with the CPU calculator.
    _binned_median = MolecularSurfaceCalculator._binned_median

    def calc_neighbor_distance_vectorized(self, molecule, my_dots, their_dots):
        """
        CalcNeighborDistance: the nearest-dot search (the O(N*M) part) runs on
        the GPU; the per-dot statistics then use the same NumPy expressions as
        the CPU version on host copies, so they round identically.

        Parameters
        ----------
        molecule   : int  0, 1, or 2 -- which surface slot to write results into
        my_dots    : _DevDots
        their_dots : _DevDots
        """
        if len(my_dots) == 0 or len(their_dots) == 0:
            return

        their_buried = cp.nonzero(their_dots.buried.astype(bool))[0]
        if their_buried.size == 0:
            return

        their_xyz = their_dots.coor_xyz[their_buried]
        their_outnml = their_dots.outnml_xyz[their_buried]

        # cdist(their_xyz, my_xyz, 'sqeuclidean') -> .argmin(axis=0) and the minimum itself
        min_dist_sq, nearest_idx = K.min_sqdist(my_dots.coor_xyz, their_xyz, return_index=True)

        distmin = np.sqrt(cp.asnumpy(min_dist_sq))
        my_outnml = cp.asnumpy(my_dots.outnml_xyz)
        neighbor_outnml = cp.asnumpy(their_outnml[nearest_idx])

        r = np.einsum('mi,mi->m', my_outnml, neighbor_outnml)
        r = r * np.exp(-np.square(distmin) * self.settings.weight)
        r = np.clip(r, -0.999, 0.999)

        surf = self.run.results.surface[molecule]
        surf.d_mean = float(distmin.mean())
        surf.d_median = self._binned_median(distmin, self.settings.binwidth_dist)
        surf.s_mean = float(-r.mean())
        surf.s_median = float(-self._binned_median(r, self.settings.binwidth_norm))

    # ── attention / neighbours ─────────────────────────────────────────────

    @time_range('assign_attention_numbers')
    def assign_attention_numbers(self, atoms, all_atoms=False):
        """
        Assign attention values to all atoms.

        The inter-molecule minimum distances come from the tiled min_sqdist
        kernel instead of a full cdist matrix.
        """
        n   = len(atoms)
        mol = self.run.atoms.molecule[:n]      # host copy (identical)

        if all_atoms:
            atoms.atten[:n] = ATTEN_BURIED_FLAGGED
            for m in range(2):
                self.run.results.surface[m].nBuriedAtoms += int((mol == m).sum())
            return 1

        idx0_h = np.where(mol == 0)[0]
        idx1_h = np.where(mol == 1)[0]
        idx0   = cp.asarray(idx0_h)
        idx1   = cp.asarray(idx1_h)
        xyz0   = atoms.xyz[idx0]    # (N0, 3)
        xyz1   = atoms.xyz[idx1]    # (N1, 3)

        if len(idx0_h) > 0 and len(idx1_h) > 0:
            # cdist(xyz0, xyz1).min(axis=1) / .min(axis=0); sqrt is monotone so
            # sqrt(min(d2)) == min(sqrt(d2)) exactly.
            min0 = cp.sqrt(K.min_sqdist(xyz0, xyz1))
            min1 = cp.sqrt(K.min_sqdist(xyz1, xyz0))
        else:
            min0 = cp.full(len(idx0_h), 99999.0)
            min1 = cp.full(len(idx1_h), 99999.0)

        blocker0  = min0 >= self.settings.sep
        blocker1  = min1 >= self.settings.sep

        atoms.atten[idx0] = cp.where(blocker0, ATTEN_BLOCKER, ATTEN_BURIED_FLAGGED).astype(cp.int8)
        atoms.atten[idx1] = cp.where(blocker1, ATTEN_BLOCKER, ATTEN_BURIED_FLAGGED).astype(cp.int8)

        nb0 = int(blocker0.sum())
        nb1 = int(blocker1.sum())
        self.run.results.surface[0].nBlockedAtoms += nb0
        self.run.results.surface[0].nBuriedAtoms  += len(idx0_h) - nb0
        self.run.results.surface[1].nBlockedAtoms += nb1
        self.run.results.surface[1].nBuriedAtoms  += len(idx1_h) - nb1

        return 1

    @time_range('calc_dots_for_all_atoms')
    def calc_dots_for_all_atoms(self, _atoms_unused):
        """
        Main surface generation loop.
        """
        dev = self._dev

        # Compute maximum atom radius (host copy -> identical np.float64)
        self.run.radmax = self.run.atoms.radius.max()

        good_atom = self.build_neighbor_arrays()

        # Run second_loop for all good atoms at once
        good_indices = cp.nonzero(good_atom)[0]
        self.second_loop(dev.atoms[good_indices])

        # Build convex_queue (vectorised filter over good atoms)
        atten    = dev.atoms.atten[good_indices]
        access   = dev.atoms.access[good_indices].astype(bool)
        n_buried = dev.buried_array.nneighbors[good_indices]

        convex_mask = (access
                       & (atten > ATTEN_BLOCKER)
                       & ~((atten == ATTEN_6) & (n_buried == 0)))
        convex_queue = good_indices[convex_mask]

        self.generate_convex_surface(dev.atoms[convex_queue])

        self.generate_toroidal_surfaces()

        dev.probes.finalize()
        # Concave surface
        if self.settings.rp > 0:
            self.generate_concave_surface()

        return 1

    @time_range('build_neighbor_arrays')
    def build_neighbor_arrays(self):
        """
        Same result as the CPU build_neighbor_arrays, without the (N, N)
        matrices: a counting kernel, an exclusive scan, an emit kernel, and a
        stable library sort that orders each atom's neighbours by ascending
        distance (ties by atom index).

        Returns
        -------
        good_atom : bool cupy array, shape (N,)
        """
        atoms  = self._dev.atoms
        N      = len(atoms)
        rp     = self.settings.rp

        radius = atoms.radius
        atten  = atoms.atten

        two_rp = 2.0 * rp
        bb2    = (4.0 * self.run.radmax + 4.0 * rp) ** 2

        # ── counts (and coincident atoms check) ───────────────────────────
        n_neighbors, n_buried, nbb, coin_j = K.neighbor_count(
            atoms.xyz, radius, atoms.molecule, atten, two_rp, bb2)

        coin_rows = cp.nonzero(coin_j >= 0)[0]
        if coin_rows.size:
            i0 = int(coin_rows[0])
            j0 = int(coin_j[i0])
            host = self.run.atoms
            raise RuntimeError(
                f"Coincident atoms: "
                f"{host.natom[i0]}:{host.residue_name[i0]}:{host.atom_name[i0]} == "
                f"{host.natom[j0]}:{host.residue_name[j0]}:{host.atom_name[j0]}"
            )

        # ── good_atom and access ──────────────────────────────────────────
        active        = atten > 0
        atten6_no_nbb = (atten == ATTEN_6) & (nbb == 0)

        new_access = active & ~atten6_no_nbb & (n_neighbors == 0)
        atoms.access |= new_access.astype(cp.int8)

        good_atom = active & ~atten6_no_nbb & (n_neighbors > 0)

        # ── compact neighbour / buried lists ──────────────────────────────
        neigh_offset  = cp.zeros(N + 1, dtype=cp.int64)
        buried_offset = cp.zeros(N + 1, dtype=cp.int64)
        cp.cumsum(n_neighbors, out=neigh_offset[1:])
        cp.cumsum(n_buried, out=buried_offset[1:])

        max_n = max(int(n_neighbors.max()), 1)
        max_b = max(int(n_buried.max()), 1)

        neigh_j, neigh_d2, buried_j = K.neighbor_emit(
            atoms.xyz, radius, atoms.molecule, atten, two_rp,
            neigh_offset, buried_offset)

        # ── neighbor_array: rows sorted by (distance, index) ──────────────
        neighbor_array = _DevNeighbors(N, max_n)
        if neigh_j.size:
            row   = cp.repeat(cp.arange(N, dtype=cp.int64), n_neighbors.astype(cp.int64))
            order = cp.lexsort(cp.stack([neigh_j.astype(cp.float64), neigh_d2,
                                         row.astype(cp.float64)]))
            j_s   = neigh_j[order]
            row_s = row[order]
            pos   = cp.arange(j_s.size, dtype=cp.int64) - neigh_offset[row_s]
            neighbor_array.xyz[row_s, pos]    = atoms.xyz[j_s]
            neighbor_array.radius[row_s, pos] = radius[j_s]
            neighbor_array.natom[row_s, pos]  = atoms.natom[j_s]
        neighbor_array.nneighbors[:] = n_neighbors

        # ── buried_array: ascending atom index (order doesn't matter) ─────
        buried_array = _DevNeighbors(N, max_b)
        if buried_j.size:
            row = cp.repeat(cp.arange(N, dtype=cp.int64), n_buried.astype(cp.int64))
            pos = cp.arange(buried_j.size, dtype=cp.int64) - buried_offset[row]
            buried_array.xyz[row, pos]    = atoms.xyz[buried_j]
            buried_array.radius[row, pos] = radius[buried_j]
            buried_array.natom[row, pos]  = atoms.natom[buried_j]
        buried_array.nneighbors[:] = n_buried

        self._dev.neighbor_array = neighbor_array
        self._dev.buried_array   = buried_array
        return good_atom

    # ── pair / triple loops ────────────────────────────────────────────────

    @time_range('second_loop')
    def second_loop(self, atoms):
        """
        second_loop over all atoms simultaneously.

        atoms : _AtomView — all good atoms to process (pre-filtered by good_atom mask)
        """
        if len(atoms) == 0:
            return

        dev = self._dev
        NA  = dev.neighbor_array
        rp  = self.settings.rp
        na1 = atoms.natom                                        # (N,) original indices

        # ── neighbour data for all atoms at once ──────────────────────────
        nneigh      = NA.nneighbors[na1]                         # (N,)
        neigh_xyz   = NA.xyz[na1]                                # (N, K, 3)
        neigh_rad   = NA.radius[na1]                             # (N, K)
        neigh_natom = NA.natom[na1]                              # (N, K)

        # ── forward-pair mask: only pairs where natom2 > natom1 ──────────
        fwd = neigh_natom > na1[:, None]                        # (N, K)

        # ── geometry for all (atom, forward-neighbor) pairs ───────────────
        a_xyz  = atoms.xyz
        diff   = neigh_xyz - a_xyz[:, None, :]                   # (N, K, 3)
        dij_sq = dot3_einsum(diff, diff)                         # (N, K)
        dij    = cp.sqrt(cp.where(fwd, dij_sq, 1.0))             # (N, K)

        eri = atoms.radius + rp                                  # (N,)
        erj = neigh_rad   + rp                                   # (N, K) — NaN for padding

        asymm   = cp.where(fwd, (cp.square(eri[:, None]) - cp.square(erj)) / dij, 0.0)  # (N, K)
        between = cp.abs(asymm) < dij                            # (N, K)
        tij     = ((a_xyz[:, None, :] + neigh_xyz) * 0.5
                   + (diff / dij[..., None]) * (asymm * 0.5)[..., None])  # (N, K, 3)

        far_sq  = cp.where(fwd, cp.square(eri[:, None] + erj) - dij_sq, -1.0)   # (N, K)
        cont_sq = cp.where(fwd, dij_sq - cp.square(atoms.radius[:, None] - neigh_rad), -1.0)  # (N, K)

        geom_ok = fwd & (far_sq > 0.0) & (cont_sq > 0.0)       # (N, K)

        # ── single-neighbour shortcut ─────────────────────────────────────
        shortcut = (nneigh <= 1) & geom_ok.any(axis=1)          # (N,)
        sc_rows  = cp.nonzero(shortcut)[0]
        if sc_rows.size:
            sc_na1 = na1[sc_rows]
            dev.atoms.access[sc_na1] = 1
            # First geom_ok forward neighbour for each shortcut atom
            first_k   = cp.argmax(geom_ok[sc_rows], axis=1)     # (S,)
            first_na2 = neigh_natom[sc_rows, first_k]
            dev.atoms.access[first_na2] = 1

        # ── flatten all non-shortcut, geom_ok pairs ───────────────────────
        pair_mask      = geom_ok & ~shortcut[:, None]           # (N, K)
        i_idx, k_idx   = cp.nonzero(pair_mask)

        if i_idx.size == 0:
            return

        na1_f  = na1[i_idx]                                      # (P,)
        na2_f  = neigh_natom[i_idx, k_idx]                       # (P,)
        dij_f  = dij[i_idx, k_idx]                               # (P,)
        uij_f  = diff[i_idx, k_idx] / dij_f[:, None]            # (P, 3)
        tij_f  = tij[i_idx, k_idx]                               # (P, 3)
        rij_f  = (0.5 * cp.sqrt(far_sq[i_idx, k_idx])
                      * cp.sqrt(cont_sq[i_idx, k_idx])
                      / dij_f)                                    # (P,)
        bet_f  = between[i_idx, k_idx]                           # (P,)

        # ── toroid queue (kept as arrays, same order as the CPU list) ─────
        a1_atten = atoms.atten[i_idx]                            # (P,)
        a2_atten = dev.atoms.atten[na2_f]                        # (P,)
        need_tor = (a1_atten > ATTEN_BLOCKER) | ((a2_atten > ATTEN_BLOCKER) & (rp > 0.0))
        tor = cp.nonzero(need_tor)[0]
        tq  = dev.toroid_queue
        tq.natom1  = cp.concatenate([tq.natom1,  na1_f[tor]])
        tq.natom2  = cp.concatenate([tq.natom2,  na2_f[tor]])
        tq.uij     = cp.concatenate([tq.uij,     uij_f[tor]])
        tq.tij     = cp.concatenate([tq.tij,     tij_f[tor]])
        tq.rij     = cp.concatenate([tq.rij,     rij_f[tor]])
        tq.between = cp.concatenate([tq.between, bet_f[tor]])

        # ── vec_third_loop (single call across all pairs) ─────────────────
        access_natoms = self.vec_third_loop(
            dev.atoms[na1_f],
            dev.atoms[na2_f],
            uij_f,
            tij_f,
            rij_f,
        )
        if access_natoms.size:
            dev.atoms.access[access_natoms] = 1

    @time_range('vec_third_loop')
    def vec_third_loop(self, atom1s, atom2s, uij, tij, rij):
        """
        third_loop over M (atom1, atom2) pairs simultaneously.

        atom1s : _AtomView  (M rows)
        atom2s : _AtomView  (M rows)
        uij    : (M, 3)  unit vector atom1→atom2
        tij    : (M, 3)  torus-circle centre
        rij    : (M,)    torus-circle radius

        Appends valid probes to the device probe array.
        Returns a 1-D int32 array of natom indices that should gain access = 1.
        """
        empty = cp.zeros(0, dtype=cp.int32)
        M  = len(atom1s)
        rp = self.settings.rp
        NA = self._dev.neighbor_array

        eri = atom1s.radius + rp   # (M,)
        erj = atom2s.radius + rp   # (M,)
        na1 = atom1s.natom         # (M,)  original atom indices
        na2 = atom2s.natom         # (M,)

        # ── neighbour data for every atom1 ───────────────────────────────
        neigh_xyz   = NA.xyz[na1]      # (M, K, 3)
        neigh_rad   = NA.radius[na1]   # (M, K)
        neigh_natom = NA.natom[na1]    # (M, K)

        # ── initial (pair, atom3) validity mask ──────────────────────────
        # atom3 must be ordered after atom2 and be a real neighbour
        valid = (neigh_natom > na2[:, None]) & (neigh_natom >= 0)  # (M, K)

        erk = cp.where(valid, neigh_rad + rp, cp.inf)   # (M, K)

        diff_jk = neigh_xyz - atom2s.xyz[:, None, :]    # (M, K, 3)
        djk     = norm3(diff_jk)                        # (M, K)
        valid  &= djk < (erj[:, None] + erk)

        diff_ik = neigh_xyz - atom1s.xyz[:, None, :]    # (M, K, 3)
        dik     = norm3(diff_ik)                        # (M, K)
        valid  &= dik < (eri[:, None] + erk)

        # all-three-blocked filter
        safe_na3    = cp.where(neigh_natom >= 0, neigh_natom, 0)
        a3_atten    = self._dev.atoms.atten[safe_na3]   # (M, K)
        all_blocked = (
            (atom1s.atten[:, None] <= ATTEN_BLOCKER) &
            (atom2s.atten[:, None] <= ATTEN_BLOCKER) &
            (a3_atten              <= ATTEN_BLOCKER)
        )
        valid &= ~all_blocked

        # ── flatten to Q valid triples ────────────────────────────────────
        pair_idx, k_idx = cp.nonzero(valid)   # (Q,)
        Q = int(pair_idx.size)
        if Q == 0:
            return empty

        a1_xyz_q = atom1s.xyz[pair_idx]              # (Q, 3)
        a3_xyz_q = neigh_xyz[pair_idx, k_idx]        # (Q, 3)
        a3_rad_q = neigh_rad[pair_idx, k_idx]        # (Q,)
        na1_q    = na1[pair_idx]                     # (Q,)
        na2_q    = na2[pair_idx]                     # (Q,)
        na3_q    = neigh_natom[pair_idx, k_idx]      # (Q,)
        eri_q    = eri[pair_idx]                     # (Q,)
        erk_q    = a3_rad_q + rp                     # (Q,)
        dik_q    = dik[pair_idx, k_idx]              # (Q,)
        uij_q    = uij[pair_idx]                     # (Q, 3)
        tij_q    = tij[pair_idx]                     # (Q, 3)
        rij_q    = rij[pair_idx]                     # (Q,)

        # ── uik, dt, wijk, swijk ─────────────────────────────────────────
        uik_q    = (a3_xyz_q - a1_xyz_q) / dik_q[:, None]            # (Q, 3)
        dt_q     = dot3_einsum(uij_q, uik_q)                         # (Q,)
        wijk_q   = cp.arccos(cp.clip(dt_q, -1.0, 1.0))               # (Q,)
        swijk_q  = cp.sin(wijk_q)                                    # (Q,)

        degenerate = (
            (dt_q >= 1.0) | (dt_q <= -1.0) |
            (wijk_q <= 0.0) | (swijk_q <= 0.0)
        )

        # ── degenerate triples: check which ones kill their pair ──────────
        # (computed for all triples, used only where degenerate)
        dtijk2 = sumsq3(tij_q - a3_xyz_q)   # squared: fixes a bug in the C++
        rkp2   = cp.square(erk_q) - cp.square(rij_q)
        kills_q = degenerate & (dtijk2 < rkp2)
        kill_pair_m = cp.zeros(M, dtype=bool)
        kill_pair_m[pair_idx[kills_q]] = True
        keep = ~degenerate & ~kill_pair_m[pair_idx]

        # ── apply keep filter ─────────────────────────────────────────────
        kq = cp.nonzero(keep)[0]
        if kq.size == 0:
            return empty
        a1_xyz_q = a1_xyz_q[kq];  a3_xyz_q = a3_xyz_q[kq]
        na1_q    = na1_q[kq];     na2_q    = na2_q[kq];   na3_q  = na3_q[kq]
        eri_q    = eri_q[kq];     erk_q    = erk_q[kq];   dik_q  = dik_q[kq]
        uij_q    = uij_q[kq];     tij_q    = tij_q[kq]
        uik_q    = uik_q[kq];     swijk_q  = swijk_q[kq]

        # ── probe geometry ────────────────────────────────────────────────
        uijk_q  = cross3(uij_q, uik_q) / swijk_q[:, None]           # (Q, 3)
        utb_q   = cross3(uijk_q, uij_q)                              # (Q, 3)

        asymm_q = (cp.square(eri_q) - cp.square(erk_q)) / dik_q     # (Q,)
        tik_q   = (a1_xyz_q + a3_xyz_q) * 0.5 + uik_q * (asymm_q * 0.5)[:, None]  # (Q, 3)

        dt_b_q  = dot3_einsum(uik_q, tik_q - tij_q)                  # (Q,)
        bijk_q  = tij_q + utb_q * (dt_b_q / swijk_q)[:, None]       # (Q, 3)

        bma       = bijk_q - a1_xyz_q
        hijk_sq_q = cp.square(eri_q) - dot3_einsum(bma, bma)         # (Q,)
        vh = cp.nonzero(hijk_sq_q > 0.0)[0]
        Q  = int(vh.size)
        if Q == 0:
            return empty

        bijk_q  = bijk_q[vh];   uijk_q  = uijk_q[vh]
        hijk_q  = cp.sqrt(hijk_sq_q[vh])
        na1_q   = na1_q[vh];    na2_q   = na2_q[vh];  na3_q = na3_q[vh]

        # ── two probe candidates per triple (isign = +1 and -1) ──────────
        # +1 half: a0=atom1, a1=atom2, a2=atom3
        # -1 half: a0=atom2, a1=atom1, a2=atom3
        bijk_2q  = cp.concatenate([bijk_q,  bijk_q ], axis=0)   # (2Q, 3)
        uijk_2q  = cp.concatenate([uijk_q,  uijk_q ], axis=0)   # (2Q, 3)
        hijk_2q  = cp.concatenate([hijk_q,  hijk_q ])            # (2Q,)
        na1_2q   = cp.concatenate([na1_q,   na1_q  ])            # (2Q,)
        na2_2q   = cp.concatenate([na2_q,   na2_q  ])            # (2Q,)
        na3_2q   = cp.concatenate([na3_q,   na3_q  ])            # (2Q,)
        isign_2q = cp.concatenate([cp.ones(Q), -cp.ones(Q)])    # (2Q,)

        pijk_2q  = bijk_2q + uijk_2q * (hijk_2q * isign_2q)[:, None]   # (2Q, 3)
        alt_2q   = uijk_2q * isign_2q[:, None]                          # (2Q, 3)

        probe_a0 = cp.where(isign_2q > 0, na1_2q, na2_2q)   # (2Q,)
        probe_a1 = cp.where(isign_2q > 0, na2_2q, na1_2q)   # (2Q,)
        probe_a2 = na3_2q                                     # (2Q,)

        # ── collision check against atom1's neighbours (minus atom2/atom3) ─
        collision = K.list_collision(
            pijk_2q, na1_2q, NA, rp,
            excl1=na2_2q, excl2=na3_2q, strict=False, order=ORDER_ALT)

        vp = cp.nonzero(~collision)[0]
        if vp.size == 0:
            return empty

        # ── write probes ──────────────────────────────────────────────────
        probe_a0_f = probe_a0[vp].astype(cp.int32)
        probe_a1_f = probe_a1[vp].astype(cp.int32)
        probe_a2_f = probe_a2[vp].astype(cp.int32)

        self._dev.probes.extend_from_arrays(
            probe_a0_f, probe_a1_f, probe_a2_f,
            hijk_2q[vp], pijk_2q[vp], alt_2q[vp],
        )

        # ── return natoms that gained access ─────────────────────────────
        return cp.unique(cp.concatenate([probe_a0_f, probe_a1_f, probe_a2_f]))

    # ── surfaces ───────────────────────────────────────────────────────────

    @time_range('generate_convex_surface')
    def generate_convex_surface(self, atoms):

        N  = len(atoms)
        if N == 0:
            return 1
        NA = self._dev.neighbor_array
        rp = self.settings.rp

        north = cp.zeros((N, 3))
        north[:] = cp.array([0., 0., 1.])

        south = cp.zeros((N, 3))
        south[:] = cp.array([0., 0., -1.])

        eqvec = cp.zeros((N, 3))
        eqvec[:] = cp.array([1., 0., 0.])

        ri  = atoms.radius                  # (N,)
        eri = atoms.radius + rp
        a_xyz = atoms.xyz

        nneigh = NA.nneighbors[atoms.natom]  # (N,)
        has_neigh = nneigh > 0

        if bool(has_neigh.any()):

            neigh_xyz = NA.xyz[atoms.natom, 0]      # (N,3)
            neigh_rad = NA.radius[atoms.natom, 0]   # (N,)

            # direction to neighbor
            north_vec = _normalize3(a_xyz - neigh_xyz)

            north = cp.where(has_neigh[:, None], north_vec, north)

            vtemp = cp.stack([
                cp.square(north[:, 1]) + cp.square(north[:, 2]),
                cp.square(north[:, 0]) + cp.square(north[:, 2]),
                cp.square(north[:, 0]) + cp.square(north[:, 1])
            ], axis=-1)

            vtemp = _normalize3(vtemp)

            dt = dot3_seq(vtemp, north)

            replace = cp.abs(dt) > 0.99
            vtemp = cp.where(replace[:, None], cp.array([1., 0., 0.]), vtemp)

            eq = _normalize3(cross3(north, vtemp))

            eqvec = cp.where(has_neigh[:, None], eq, eqvec)

            vql = cross3(eqvec, north)

            rj  = neigh_rad
            erj = neigh_rad + rp

            dij = norm3(neigh_xyz - a_xyz)
            uij = (neigh_xyz - a_xyz) / dij[:, None]

            asymm = (eri * eri - erj * erj) / dij
            tij = ((a_xyz + neigh_xyz) * 0.5) + uij * (asymm[:, None] * 0.5)

            far_sq = cp.square(eri + erj) - dij * dij
            contain_sq = dij * dij - cp.square(ri - rj)

            # A neighbour whose geometry is undefined relative to this atom
            # (far_sq <= 0 or contain_sq <= 0, e.g. a radius <= 0 neighbour
            # inside this atom's sphere) keeps the default south pole.
            geom_ok = has_neigh & (far_sq > 0.0) & (contain_sq > 0.0)

            far     = cp.sqrt(cp.where(geom_ok, far_sq, 1.0))
            contain = cp.sqrt(cp.where(geom_ok, contain_sq, 1.0))

            rij = 0.5 * far * contain / dij

            pij = tij + vql * rij[:, None]
            south_vec = (pij - a_xyz) / eri[:, None]

            south = cp.where(geom_ok[:, None], south_vec, south)

        # ---------------------------------------------------------
        # Latitude arcs for ALL atoms simultaneously
        # ---------------------------------------------------------

        o = cp.zeros((N, 3))

        cs, lats = self.vec_sub_arc(
            o,
            ri,
            eqvec,
            atoms.density,
            north,
            south
        )

        # dt per atom per latitude
        dt = dot3_seq(lats, north[:, None, :])  # (N, K)

        cen = a_xyz[:, None, :] + dt[..., None] * north[:, None, :]

        rad_sq = cp.square(ri[:, None]) - dt * dt
        valid_rad = rad_sq > 0

        rad = cp.where(valid_rad, cp.sqrt(cp.where(valid_rad, rad_sq, 0.0)), 0.0)

        # ---------------------------------------------------------
        # Generate ALL circle points, skipping zero-radius rows
        # ---------------------------------------------------------

        points, ps = self._circles(cen, rad, north, atoms.density)
        M = points.shape[1]

        valid_points = ~cp.isnan(points[..., 0])

        area = ps * cs[:, None]

        # ---------------------------------------------------------
        # Project outward
        # ---------------------------------------------------------

        atom_idx, lat_idx, circ_idx = cp.nonzero(valid_points)
        points_flat = points[atom_idx, lat_idx, circ_idx]

        pcen = a_xyz[atom_idx] + (
            points_flat - a_xyz[atom_idx]
        ) * (eri[atom_idx] / ri[atom_idx])[:, None]

        # ---------------------------------------------------------
        # Collision check (skips the first neighbour, like the CPU)
        # ---------------------------------------------------------

        collisions = K.list_collision(
            pcen, atoms.natom[atom_idx], NA, rp,
            start=1, strict=False, order=ORDER_SEQ)

        keep = cp.nonzero(~collisions)[0]

        if keep.size == 0:
            return 1

        points_keep = points_flat[keep]
        pcen_keep   = pcen[keep]
        atom_keep   = atom_idx[keep]
        lat_keep    = lat_idx[keep]

        self.run.results.dots.convex += int(keep.size)

        self.add_dots(
            atoms.molecule[atom_keep],
            1,
            points_keep,
            area[atom_keep, lat_keep],
            pcen_keep,
            atoms.natom[atom_keep],
        )

        return 1

    def _circles(self, cen, rad, axis, density):
        """
        Shared tail of the convex / concave surfaces: subdivide every
        (row, latitude) circle with rad > 0.

        cen (R, M, 3), rad (R, M), axis (R, 3), density (R,)
        Returns points (R, M, K, 3) NaN-padded and ps (R, M).
        """
        R, M     = rad.shape
        flat_rad = rad.reshape(-1)
        active   = cp.nonzero(flat_rad > 0)[0]

        if active.size:
            flat_cen     = cen.reshape(-1, 3)
            flat_axis    = cp.repeat(axis, M, axis=0)
            flat_density = cp.repeat(density, M)
            ps_a, pts_a  = self.vec_sub_cir(
                flat_cen[active], flat_rad[active],
                flat_axis[active], flat_density[active]
            )
            Kc               = pts_a.shape[1]
            full_pts         = cp.full((R * M, Kc, 3), cp.nan)
            full_pts[active] = pts_a
            full_ps          = cp.zeros(R * M)
            full_ps[active]  = ps_a
        else:
            Kc       = 0
            full_pts = cp.empty((R * M, 0, 3))
            full_ps  = cp.zeros(R * M)

        return full_pts.reshape(R, M, Kc, 3), full_ps.reshape(R, M)

    def vec_check_point_collision(self, pcen, xyzs, rads):
        """Same contract as the CPU method (kept for API parity)."""
        dists2 = sumsq3(xyzs - pcen)
        collision = (dists2 <= cp.square(rads + self.settings.rp)) & ~cp.isnan(rads)

        return collision[..., 1:].any(axis=-1)

    @time_range('generate_toroidal_surfaces')
    def generate_toroidal_surfaces(self):

        tq = self._dev.toroid_queue
        if len(tq) == 0:
            return

        atoms = self._dev.atoms
        natoms1 = tq.natom1
        natoms2 = tq.natom2

        atom1_has_access, atom2_has_access = self.generate_toroidal_surface(
                                                                atoms[natoms1],
                                                                atoms[natoms2],
                                                                tq.uij, tq.tij, tq.rij, tq.between)
        # NumPy's  a[idx] |= v  with repeated indices keeps the value computed
        # for the *last* occurrence; CuPy leaves that undefined, so pick the
        # last occurrence explicitly.
        _or_assign_last_wins(atoms.access, natoms1, atom1_has_access)
        _or_assign_last_wins(atoms.access, natoms2, atom2_has_access)

    @time_range('generate_toroidal_surface')
    def generate_toroidal_surface(
        self,
        atom1,
        atom2,
        uij,      # (N,3)
        tij,      # (N,3)
        rij,      # (N,)
        between   # (N,)
    ):

        N  = len(atom1)
        NA = self._dev.neighbor_array
        rp = self.settings.rp

        no_access = (cp.zeros(N, dtype=bool), cp.zeros(N, dtype=bool))

        density = (atom1.density + atom2.density) * 0.5

        eri = atom1.radius + rp
        erj = atom2.radius + rp

        rci = rij * atom1.radius / eri
        rcj = rij * atom2.radius / erj
        rb  = cp.maximum(rij - rp, 0.0)

        rs = (rci + 2 * rb + rcj) * 0.25
        e = rs / rij
        edens = e * e * density

        # ---------------------------------------------------------
        # Subdivide torus circle (batched)
        # ---------------------------------------------------------

        ts, subs = self.vec_sub_cir(
            tij,
            rij,
            uij,
            edens
        )

        # subs: (N, M, 3)
        valid_sub = ~cp.isnan(subs[..., 0])

        # Flatten valid subdivisions
        atom_idx, sub_idx = cp.nonzero(valid_sub)
        if atom_idx.size == 0:
            return no_access
        pij = subs[atom_idx, sub_idx]      # (Q,3)

        # ---------------------------------------------------------
        # Neighbor collision test (excluding atom2, strict <)
        # ---------------------------------------------------------

        a1_natom = atom1.natom[atom_idx]
        a2_natom = atom2.natom[atom_idx]
        too_close_any = K.list_collision(
            pij, a1_natom, NA, rp,
            excl1=a2_natom, strict=True, order=ORDER_ALT)

        valid_point = ~too_close_any

        valid_point &= ~ ((atom1.atten[atom_idx] == ATTEN_6)
                        & (atom2.atten[atom_idx] == ATTEN_6)
                        & (self._dev.buried_array.nneighbors[a1_natom] == 0)
                        )

        vp = cp.nonzero(valid_point)[0]
        if vp.size == 0:
            return no_access

        pij = pij[vp]
        atom_idx = atom_idx[vp]

        # Mark access
        atom1_has_access = cp.zeros(N, dtype=bool)
        atom2_has_access = cp.zeros(N, dtype=bool)

        atom1_has_access[atom_idx] = True
        atom2_has_access[atom_idx] = True

        # ---------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------

        pi = (atom1.xyz[atom_idx] - pij) / eri[atom_idx, None]
        pj = (atom2.xyz[atom_idx] - pij) / erj[atom_idx, None]

        axis = _normalize3(cross3(pi, pj))

        rij_a = rij[atom_idx]
        dtq = rp**2 - cp.square(rij_a)
        pcusp = (dtq > 0) & between[atom_idx]

        # cusp rows: pqi = (qij - pij) / rp, pqj = 0
        dtq_s = cp.sqrt(cp.where(pcusp, dtq, 0.0))
        qij = tij[atom_idx] - uij[atom_idx] * dtq_s[:, None]
        pqi_cusp = (qij - pij) / rp

        # other rows: pqi = pqj = normalised (pi + pj)
        pq_mid = _normalize3(pi + pj)

        pqi = cp.where(pcusp[:, None], pqi_cusp, pq_mid)
        pqj = cp.where(pcusp[:, None], 0.0, pq_mid)

        # Reject invalid dot cases
        dt1 = dot3_seq(pqi, pi)
        dt2 = dot3_seq(pqj, pj)

        va = cp.nonzero((cp.abs(dt1) < 1.0) & (cp.abs(dt2) < 1.0))[0]

        if va.size == 0:
            return atom1_has_access, atom2_has_access

        pij = pij[va]
        axis = axis[va]
        pi = pi[va]
        pj = pj[va]
        pqi = pqi[va]
        pqj = pqj[va]
        atom_idx = atom_idx[va]

        # ---------------------------------------------------------
        # Arc generation (batched)
        # ---------------------------------------------------------

        mol1 = atom1.molecule[atom_idx]

        # ---- atom1 arc ----
        m1 = cp.nonzero(atom1.atten[atom_idx] >= ATTEN_2)[0]
        if m1.size:
            self._toroidal_arc(pij[m1], axis[m1], density, pi[m1], pqi[m1],
                               atom_idx[m1], tij, uij, rij, ts,
                               mol1[m1], atom1.natom[atom_idx[m1]])

        # ---- atom2 arc ----
        m2 = cp.nonzero(atom2.atten[atom_idx] >= ATTEN_2)[0]
        if m2.size:
            self._toroidal_arc(pij[m2], axis[m2], density, pqj[m2], pj[m2],
                               atom_idx[m2], tij, uij, rij, ts,
                               mol1[m2], atom2.natom[atom_idx[m2]])

        return atom1_has_access, atom2_has_access

    def _toroidal_arc(self, pij, axis, density, x, v, aidx, tij, uij, rij, ts, mol, natom):
        """One of the two (identical-shaped) arc blocks of generate_toroidal_surface."""
        ps, points = self.vec_sub_arc(
            pij,
            cp.full(pij.shape[0], self.settings.rp),
            axis,
            density[aidx],
            x,
            v
        )

        dist = self.vec_distance_point_to_line(
            tij[aidx][..., None, :],
            uij[aidx][..., None, :],
            points
        )

        areas = ps[:, None] * ts[aidx, None] * dist / rij[aidx, None]
        rows, cols = cp.nonzero(~cp.isnan(areas))

        Kv = int(rows.size)
        self.run.results.dots.toroidal += Kv
        if Kv > 0:
            self.add_dots(mol[rows], 2, points[rows, cols], areas[rows, cols],
                          pij[rows], natom[rows])

    @time_range('generate_concave_surface')
    def generate_concave_surface(self):

        probes = self._dev.probes
        if not probes:
            return 1

        rp = self.settings.rp
        rp2 = rp * rp
        atoms = self._dev.atoms

        # ---------------------------------------------------------
        # Pull probe data into arrays
        # ---------------------------------------------------------

        P = len(probes)

        pijk = probes.point_xyz   # (P,3)
        uijk = probes.alt_xyz     # (P,3)
        hijk = probes.height      # (P,)

        atom_natom = [probes.atom_idx_0, probes.atom_idx_1, probes.atom_idx_2]

        atom_xyz      = cp.stack([atoms.xyz[n]      for n in atom_natom], axis=1)   # (P,3,3)
        atom_radius   = cp.stack([atoms.radius[n]   for n in atom_natom], axis=-1)  # (P,3)
        atom_density  = cp.stack([atoms.density[n]  for n in atom_natom], axis=-1)
        atom_atten    = cp.stack([atoms.atten[n]    for n in atom_natom], axis=-1)
        atom_molecule = cp.stack([atoms.molecule[n] for n in atom_natom], axis=-1)

        # np.mean over 3: sequential sum, then divide by 3
        density = ((atom_density[:, 0] + atom_density[:, 1]) + atom_density[:, 2]) / 3.0  # (P,)

        # ---------------------------------------------------------
        # Identify low probes
        # ---------------------------------------------------------

        low_mask    = hijk < rp
        low_indices = cp.nonzero(low_mask)[0]          # (L,) original probe indices
        low_points  = pijk[low_indices]

        # ---------------------------------------------------------
        # Skip fully attenuated probes
        # ---------------------------------------------------------

        skip = cp.all(atom_atten == ATTEN_6, axis=1)
        if not bool(cp.any(~skip)):
            return 1

        # ---------------------------------------------------------
        # Nearby low probes
        # ---------------------------------------------------------

        nears_mask = None
        if low_indices.size:

            d2 = sumsq3(pijk[:, None, :] - low_points[None, :, :])  # (P, L)

            nears_mask = d2 <= 4 * rp2
            nears_mask[low_indices, cp.arange(low_indices.size)] = False  # exclude self

        # ---------------------------------------------------------
        # Vectors from probe to atoms
        # ---------------------------------------------------------

        vp = _normalize3(atom_xyz - pijk[:, None, :])

        vectors = cp.stack([
            cross3(vp[:, 0], vp[:, 1]),
            cross3(vp[:, 1], vp[:, 2]),
            cross3(vp[:, 2], vp[:, 0])
        ], axis=1)

        vectors = _normalize3(vectors)

        # ---------------------------------------------------------
        # Highest vertex
        # ---------------------------------------------------------

        dt = dot3_seq(uijk[:, None, :], vp)   # (P,3)
        mm = cp.argmax(dt, axis=1)

        south = -uijk

        vtop = vp[cp.arange(P), mm]
        axis = _normalize3(cross3(vtop, south))

        # ---------------------------------------------------------
        # Latitude arcs (batched)
        # ---------------------------------------------------------

        o = cp.zeros_like(pijk)

        cs, lats = self.vec_sub_arc(
            o,
            cp.full(P, rp),
            axis,
            density,
            vtop,
            south
        )

        # ---------------------------------------------------------
        # Circle subdivisions
        # ---------------------------------------------------------

        dt_lat = dot3_seq(lats, south[:, None, :])
        cen = south[:, None, :] * dt_lat[..., None]

        rad_sq = rp2 - cp.square(dt_lat)
        rad = cp.sqrt(cp.where(rad_sq < 0, 0.0, rad_sq))   # np.clip(rad_sq, 0, None)

        points, ps = self._circles(cen, rad, south, density)

        valid_points = ~cp.isnan(points[..., 0])

        area = ps * cs[:, None]

        # ---------------------------------------------------------
        # Flatten all valid geometry
        # ---------------------------------------------------------

        idx_probe, idx_lat, idx_pt = cp.nonzero(valid_points)
        pts = points[idx_probe, idx_lat, idx_pt]

        # Vector rejection test
        vecs = vectors[idx_probe]
        bail = cp.any(
            dot3_seq(pts[:, None, :], vecs) >= 0,
            axis=1
        )

        nb = cp.nonzero(~bail)[0]
        idx_probe = idx_probe[nb]
        idx_lat = idx_lat[nb]

        pts = pts[nb] + pijk[idx_probe]

        # ---------------------------------------------------------
        # Low-probe collision
        # ---------------------------------------------------------

        if nears_mask is not None:

            near_sel = nears_mask[idx_probe]
            near_any = near_sel.any(axis=1)

            coll = self.check_probe_collision_vectorized(
                pts,
                low_points,
                near_sel,
                rp2
            )

            reject = (hijk[idx_probe] < rp) & near_any & coll

            ok = cp.nonzero(~reject)[0]
            pts = pts[ok]
            idx_probe = idx_probe[ok]
            idx_lat = idx_lat[ok]

        # ---------------------------------------------------------
        # Closest atom selection
        # ---------------------------------------------------------

        d = norm3(pts[:, None, :] - atom_xyz[idx_probe]) - atom_radius[idx_probe]

        mc = cp.argmin(d, axis=1)

        # ---------------------------------------------------------
        # Final dot creation
        # ---------------------------------------------------------

        self.run.results.dots.concave += int(pts.shape[0])

        atom_natom_stack = cp.stack(atom_natom, axis=1)   # (P, 3)
        self.add_dots(
            atom_molecule[idx_probe, mc],
            3,
            pts,
            area[idx_probe, idx_lat],
            pijk[idx_probe],
            atom_natom_stack[idx_probe, mc],
        )

        return 1

    def check_probe_collision_vectorized(
        self,
        points,        # (N,3)
        near_points,   # (M,3)
        near_mask,     # (N,M) bool
        r2             # scalar
        ):
        """
        Returns (N,) boolean array.
        True if point i collides with ANY allowed near_point.
        """

        N = points.shape[0]

        rows, cols = cp.nonzero(near_mask)

        collided_points = cp.zeros(N, dtype=bool)
        if rows.size == 0:
            return collided_points

        diff = points[rows] - near_points[cols]
        d2 = dot3_einsum(diff, diff)

        collided_points[rows[d2 < r2]] = True

        return collided_points

    def add_dots(self, molecule, type_, coor, area, pcen, atom_indices):
        """
        Batch version of add_dot.

        molecule     : (N,) int   — 0 or 1
        type_        : int scalar — 1=convex, 2=toroidal, 3=concave
        coor         : (N, 3)    — surface dot coordinates
        area         : (N,)      — area per dot
        pcen         : (N, 3)    — probe / atom centre (for normal and burial)
        atom_indices : (N,) int  — index into the atom arrays (== natom)
        """
        N = int(coor.shape[0])
        if N == 0:
            return

        pradius      = self.settings.rp
        atom_indices = atom_indices.astype(cp.int32, copy=False)
        molecule     = molecule.astype(cp.int8, copy=False)

        # ── outward normal ────────────────────────────────────────────────
        if pradius <= 0:
            outnml = coor - self._dev.atoms.xyz[atom_indices]   # (N, 3)
        else:
            outnml = (pcen - coor) / pradius                    # (N, 3)

        # ── buried determination ──────────────────────────────────────────
        buried = K.list_collision(
            pcen, atom_indices, self._dev.buried_array, pradius,
            strict=False, order=ORDER_SEQ).astype(cp.int8)

        # ── partition by molecule and batch-append ────────────────────────
        type_arr = cp.full(N, type_, dtype=cp.int8)
        for mol in (0, 1):
            sel = cp.nonzero(molecule == mol)[0]
            if sel.size == 0:
                continue
            self._dev.dots[mol].extend(
                coor[sel], outnml[sel],
                area[sel], buried[sel], type_arr[sel], atom_indices[sel],
            )

    # ── arc / circle subdivision ───────────────────────────────────────────

    def vec_distance_point_to_line(self, cen, axis, pnt):

        vec = pnt - cen
        dt = dot3_seq(vec, axis)
        d2 = sumsq3(vec) - dt * dt

        return cp.where(d2 < 0.0, 0.0, cp.sqrt(d2))

    def vec_sub_arc(self, cen, rad, axis, density, x, v):
        """
        cen, axis, x, v: (..., 3)
        rad, density: (...)
        Returns:
            ps: (...)
            points: (..., K, 3)  where K <= MAX_SUBDIV
        """

        # y = axis × x
        y = cross3(axis, x)

        dt1 = dot3_seq(v, x)
        dt2 = dot3_seq(v, y)

        angle = cp.arctan2(dt2, dt1)
        angle = cp.where(angle < 0.0, angle + 2 * np.pi, angle)

        # np.isclose(dt, 0) with the default tolerances is |dt| <= 1e-08
        angle = cp.where((cp.abs(dt1) <= 1e-08) & (cp.abs(dt2) <= 1e-08), cp.nan, angle)

        return self.vec_sub_div(cen, rad, x, y, angle, density)

    def vec_sub_div(self, cen, rad, x, y, angle, density):
        """
        cen, x, y: (..., 3)
        rad, angle, density: (...)

        Returns:
            ps: (...)
            points: (..., K, 3)  where K = min(MAX_SUBDIV, max subdivisions needed)
        """

        rad     = cp.asarray(rad)
        density = cp.asarray(density)
        angle   = cp.asarray(angle)

        base_shape = rad.shape

        # Angular spacing; invalid elements produce inf/nan like on the CPU
        delta = 1.0 / (cp.sqrt(density) * rad)
        raw   = angle / delta          # exact subdivisions needed per element

        # Trim the inner dimension to only as many slots as the worst-case element
        # needs, rather than always allocating MAX_SUBDIV=100.
        valid_raw = cp.isfinite(raw) & (raw > 0)
        raw_max = float(cp.where(valid_raw, raw, -cp.inf).max()) if raw.size else -math.inf
        if raw_max == -math.inf:
            return cp.zeros(base_shape), cp.full(base_shape + (0, 3), cp.nan)

        max_count = int(min(MAX_SUBDIV, math.ceil(raw_max)))
        if max_count == 0:
            return cp.zeros(base_shape), cp.full(base_shape + (0, 3), cp.nan)

        i = cp.arange(max_count)

        # a_i = delta*(i + 1/2)
        a = delta[..., None] * (i + 0.5)

        # Mask where subdivision exceeds angle
        mask = a <= angle[..., None]

        # cos/sin
        c = rad[..., None] * cp.cos(a)
        s = rad[..., None] * cp.sin(a)

        # Expand vectors
        cen_exp = cen[..., None, :]
        x_exp   = x[..., None, :]
        y_exp   = y[..., None, :]

        points = cen_exp + x_exp * c[..., None] + y_exp * s[..., None]

        # Apply mask → nan where invalid
        points = cp.where(mask[..., None], points, cp.nan)

        # Count valid points per element; ps = arc_length / count
        counts = cp.sum(mask, axis=-1)
        ps = cp.where(counts > 0,
                      rad * angle / cp.maximum(counts.astype(cp.float64), 0.01), 0.0)

        return ps, points

    def vec_sub_cir(self, cen, rad, axis, density):
        """
        cen, axis: (..., 3)
        rad, density: (...)

        Returns:
            ps: (...)
            points: (..., MAX_SUBDIV, 3)
        """

        axis = _normalize3(axis)

        # Build v1
        v1 = cp.stack([
            cp.square(axis[..., 1]) + cp.square(axis[..., 2]),
            cp.square(axis[..., 0]) + cp.square(axis[..., 2]),
            cp.square(axis[..., 0]) + cp.square(axis[..., 1])
        ], axis=-1)

        v1 = _normalize3(v1)

        dt = dot3_seq(v1, axis)

        # Replace near-parallel cases
        replace = cp.abs(dt) > 0.99
        v1 = cp.where(replace[..., None],
                      cp.array([1.0, 0.0, 0.0]),
                      v1)

        v2 = _normalize3(cross3(axis, v1))

        x = _normalize3(cross3(axis, v2))

        y = cross3(axis, x)

        angle = cp.full_like(rad, 2 * np.pi)

        return self.vec_sub_div(cen, rad, x, y, angle, density)


def _or_assign_last_wins(target, idx, values):
    """
    Reproduce NumPy's  target[idx] |= values  for repeated indices: NumPy
    evaluates  target[idx] | values  and then assigns element by element, so
    for a repeated index the value from its last occurrence is what remains.
    """
    n = int(idx.shape[0])
    if n == 0:
        return
    pos  = cp.arange(n, dtype=cp.int64)
    last = cp.full(target.shape[0], -1, dtype=cp.int64)
    cupyx.scatter_max(last, idx.astype(cp.int64), pos)
    sel  = cp.nonzero(last[idx] == pos)[0]
    uidx = idx[sel]
    target[uidx] = target[uidx] | values[sel].astype(target.dtype)
