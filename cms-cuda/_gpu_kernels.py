"""
CUDA kernels for the GPU (CuPy) implementation of contact molecular surface.

Everything here is float64 and compiled with ``--fmad=false`` so that
``a*b + c`` is never contracted into an FMA; each + - * / sqrt then rounds
exactly like the NumPy/SciPy code in ``_core.py``.

3-term sums are written in the same order NumPy/SciPy use on the CPU:

* ``np.sum(..., axis=-1)``, ``np.square(...).sum(-1)``, ``np.linalg.norm``,
  ``scipy cdist``                    ->  (p0 + p1) + p2   ("seq")
* ``np.einsum('...i,...i->...')``    ->  (p0 + p2) + p1   ("alt", NumPy's
  two-lane SIMD reduction on the x86-64 baseline build)

Set ``CMS_CUDA_KERNEL_DEBUG=1`` to compile the raw kernels with device
debug info and bounds assertions (use together with compute-sanitizer).
"""

import os
import threading

import cupy as cp


KERNEL_DEBUG = os.environ.get('CMS_CUDA_KERNEL_DEBUG', '') not in ('', '0')

_EW_OPTIONS  = ('--fmad=false',)
_RAW_OPTIONS = ('--fmad=false', '-std=c++17') + (
    ('-G', '-lineinfo', '-DCMS_DEBUG') if KERNEL_DEBUG else ())

ORDER_SEQ = 0
ORDER_ALT = 1

_BLOCK = 256


_RAW_SOURCE = r'''
#ifdef CMS_DEBUG
#define CMS_ASSERT(cond) do { if (!(cond)) { \
    printf("CMS_ASSERT failed: %s (line %d, block %d, thread %d)\n", \
           #cond, __LINE__, blockIdx.x, threadIdx.x); \
    __trap(); } } while (0)
#else
#define CMS_ASSERT(cond) do { } while (0)
#endif

#define CMS_TILE 256
#define CMS_FULL_MASK 0xffffffffu

// Sum of squares of a 3-vector difference in the order used by scipy cdist
// and np.square(...).sum(-1):  (d0*d0 + d1*d1) + d2*d2
__device__ __forceinline__ double sqdist_seq(double ax, double ay, double az,
                                             double bx, double by, double bz)
{
    const double dx = ax - bx;
    const double dy = ay - by;
    const double dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

// Same, in np.einsum order:  (d0*d0 + d2*d2) + d1*d1
__device__ __forceinline__ double sqdist_alt(double ax, double ay, double az,
                                             double bx, double by, double bz)
{
    const double dx = ax - bx;
    const double dy = ay - by;
    const double dz = az - bz;
    return (dx * dx + dz * dz) + dy * dy;
}


// For every query point, the minimum squared distance to any reference point
// (the exact brute-force nearest neighbour; equivalent to
// cdist(ref, query, 'sqeuclidean').min(axis=0)).  Reference points are staged
// through shared memory one tile at a time.  If out_idx is given it receives
// the index of the first reference point attaining the minimum, i.e.
// cdist(ref, query, 'sqeuclidean').argmin(axis=0) (inputs contain no NaN).
extern "C" __global__
void min_sqdist(const double* __restrict__ query, const int n_query,
                const double* __restrict__ ref,   const int n_ref,
                double* __restrict__ out, int* __restrict__ out_idx)
{
    __shared__ double tile[CMS_TILE * 3];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = i < n_query;

    double qx = 0.0, qy = 0.0, qz = 0.0;
    if (live) {
        qx = query[3 * i + 0];
        qy = query[3 * i + 1];
        qz = query[3 * i + 2];
    }
    double best = __longlong_as_double(0x7ff0000000000000LL);   // +inf
    int best_i = -1;

    for (int base = 0; base < n_ref; base += CMS_TILE) {
        const int n = min(CMS_TILE, n_ref - base);
        for (int t = threadIdx.x; t < 3 * n; t += blockDim.x) {
            CMS_ASSERT(3 * base + t < 3 * n_ref);
            tile[t] = ref[3 * base + t];
        }
        __syncthreads();
        if (live) {
            for (int k = 0; k < n; ++k) {
                const double d2 = sqdist_seq(tile[3 * k + 0], tile[3 * k + 1], tile[3 * k + 2],
                                             qx, qy, qz);
                if (d2 < best) { best = d2; best_i = base + k; }    // ascending index: first minimum
            }
        }
        __syncthreads();
    }
    if (live) {
        out[i] = best;
        if (out_idx) out_idx[i] = best_i;
    }
}


// Pass 1 of build_neighbor_arrays: per-atom counts of same-molecule
// neighbours, cross-molecule buried atoms, and cross-molecule atoms within
// bb2; plus the first coincident same-molecule atom (row-major order).
// One warp per atom i; the lanes sweep j in ascending chunks of 32.
extern "C" __global__
void neighbor_count(const double* __restrict__ xyz,
                    const double* __restrict__ radius,
                    const signed char* __restrict__ mol,
                    const signed char* __restrict__ atten,
                    const int n, const double two_rp, const double bb2,
                    int* __restrict__ n_neigh, int* __restrict__ n_buried,
                    int* __restrict__ n_bb, int* __restrict__ coincident_j)
{
    const int i    = (int)(((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (i >= n) return;                         // uniform across the warp

    int cn = 0, cb = 0, cbb = 0, cj = -1;
    if (atten[i] > 0) {                         // uniform across the warp
        const double xi = xyz[3 * i + 0], yi = xyz[3 * i + 1], zi = xyz[3 * i + 2];
        const double ri = radius[i];
        const signed char mi = mol[i];
        for (int base = 0; base < n; base += 32) {
            const int j = base + lane;
            bool nb = false, coin = false, bur = false, bbx = false;
            if (j < n) {
                const double d2 = sqdist_seq(xi, yi, zi, xyz[3 * j + 0], xyz[3 * j + 1], xyz[3 * j + 2]);
                const double t = (ri + radius[j]) + two_rp;
                const double bridge2 = t * t;
                if (mol[j] == mi) {
                    if (j != i && atten[j] > 0) {
                        coin = d2 <= 0.0001;
                        nb   = d2 < bridge2;
                    }
                } else if (atten[j] >= 5) {          // ATTEN_BURIED_FLAGGED
                    bur = d2 < bridge2;
                    bbx = d2 < bb2;
                }
            }
            cn  += __popc(__ballot_sync(CMS_FULL_MASK, nb));
            cb  += __popc(__ballot_sync(CMS_FULL_MASK, bur));
            cbb += __popc(__ballot_sync(CMS_FULL_MASK, bbx));
            const unsigned int mc = __ballot_sync(CMS_FULL_MASK, coin);
            if (cj < 0 && mc) cj = base + __ffs(mc) - 1;
        }
    }
    if (lane == 0) {
        n_neigh[i] = cn;
        n_buried[i] = cb;
        n_bb[i] = cbb;
        coincident_j[i] = cj;
    }
}


// Pass 2 of build_neighbor_arrays: write the neighbour candidates
// (index, squared distance) and buried partners of every atom into compact
// arrays at the offsets produced by an exclusive scan of pass 1.  One warp
// per atom; a ballot prefix count keeps the entries in ascending j, the
// order np.where uses on the CPU.
extern "C" __global__
void neighbor_emit(const double* __restrict__ xyz,
                   const double* __restrict__ radius,
                   const signed char* __restrict__ mol,
                   const signed char* __restrict__ atten,
                   const int n, const double two_rp,
                   const long long* __restrict__ neigh_offset,
                   const long long* __restrict__ buried_offset,
                   const long long n_neigh_total, const long long n_buried_total,
                   int* __restrict__ neigh_j, double* __restrict__ neigh_d2,
                   int* __restrict__ buried_j)
{
    const int i    = (int)(((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (i >= n || atten[i] <= 0) return;        // uniform across the warp

    const unsigned int below = (1u << lane) - 1u;
    const double xi = xyz[3 * i + 0], yi = xyz[3 * i + 1], zi = xyz[3 * i + 2];
    const double ri = radius[i];
    const signed char mi = mol[i];
    long long pn = neigh_offset[i];
    long long pb = buried_offset[i];
    for (int base = 0; base < n; base += 32) {
        const int j = base + lane;
        bool nb = false, bur = false;
        double d2 = 0.0;
        if (j < n) {
            d2 = sqdist_seq(xi, yi, zi, xyz[3 * j + 0], xyz[3 * j + 1], xyz[3 * j + 2]);
            const double t = (ri + radius[j]) + two_rp;
            const double bridge2 = t * t;
            if (mol[j] == mi) {
                nb = j != i && atten[j] > 0 && d2 < bridge2;
            } else {
                bur = atten[j] >= 5 && d2 < bridge2;
            }
        }
        const unsigned int mn = __ballot_sync(CMS_FULL_MASK, nb);
        if (nb) {
            const long long q = pn + __popc(mn & below);
            CMS_ASSERT(q < n_neigh_total);
            neigh_j[q] = j;
            neigh_d2[q] = d2;
        }
        pn += __popc(mn);
        const unsigned int mb = __ballot_sync(CMS_FULL_MASK, bur);
        if (bur) {
            const long long q = pb + __popc(mb & below);
            CMS_ASSERT(q < n_buried_total);
            buried_j[q] = j;
        }
        pb += __popc(mb);
    }
    CMS_ASSERT(pn == neigh_offset[i + 1]);
    CMS_ASSERT(pb == buried_offset[i + 1]);
}


// Axis-aligned bounding box of each CMS_TILE-point tile of `ref`
// (box[6t + 0..2] = min xyz, box[6t + 3..5] = max xyz).
extern "C" __global__
void tile_bounds(const double* __restrict__ ref, const int n_ref, const int n_tiles,
                 double* __restrict__ box)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tiles) return;
    const double inf = __longlong_as_double(0x7ff0000000000000LL);
    double lo0 = inf, lo1 = inf, lo2 = inf, hi0 = -inf, hi1 = -inf, hi2 = -inf;
    const int e = min((t + 1) * CMS_TILE, n_ref);
    for (int k = t * CMS_TILE; k < e; ++k) {
        lo0 = fmin(lo0, ref[3 * k + 0]);  hi0 = fmax(hi0, ref[3 * k + 0]);
        lo1 = fmin(lo1, ref[3 * k + 1]);  hi1 = fmax(hi1, ref[3 * k + 1]);
        lo2 = fmin(lo2, ref[3 * k + 2]);  hi2 = fmax(hi2, ref[3 * k + 2]);
    }
    box[6 * t + 0] = lo0;  box[6 * t + 1] = lo1;  box[6 * t + 2] = lo2;
    box[6 * t + 3] = hi0;  box[6 * t + 4] = hi1;  box[6 * t + 5] = hi2;
}


// Same result as min_sqdist, bit for bit, but skips reference tiles that
// cannot contain a closer point.  Both point sets are spatially sorted (Morton
// order) so a block's queries and a tile's points are compact; each block
// visits tiles outward from `start_tile[block]`.  A tile is skipped only when,
// for every thread, the squared distance to the tile's bounding box is already
// >= the best distance found.  Rounding is monotonic, so that box bound is
// never larger than the rounded d2 of any point in the tile: skipping cannot
// change the minimum.
// With out_idx (argmin mode) ref_index gives each sorted point's original
// index; ties keep the smallest original index, and a tile is skipped only if
// its bound is strictly greater than the best distance, so the result equals
// the brute-force first-index argmin.
extern "C" __global__
void min_sqdist_pruned(const double* __restrict__ query, const int n_query,
                       const double* __restrict__ ref,   const int n_ref,
                       const double* __restrict__ box,   const int n_tiles,
                       const int* __restrict__ start_tile,
                       const int* __restrict__ ref_index,
                       double* __restrict__ out, int* __restrict__ out_idx)
{
    __shared__ double tile[CMS_TILE * 3];
    __shared__ int tile_idx[CMS_TILE];
    const bool want_idx = out_idx != nullptr;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = i < n_query;

    double qx = 0.0, qy = 0.0, qz = 0.0;
    if (live) {
        qx = query[3 * i + 0];
        qy = query[3 * i + 1];
        qz = query[3 * i + 2];
    }
    double best = __longlong_as_double(0x7ff0000000000000LL);   // +inf
    int best_i = 0x7fffffff;

    const int t0 = start_tile[blockIdx.x];
    CMS_ASSERT(t0 >= 0 && t0 < n_tiles);
    for (int k = 0; k < 2 * n_tiles; ++k) {
        // t0, t0+1, t0-1, t0+2, t0-2, ...  (every tile exactly once)
        const int t = (k & 1) ? t0 + ((k + 1) >> 1) : t0 - (k >> 1);
        if (t < 0 || t >= n_tiles) continue;                   // uniform across the block

        const double* b = box + 6 * t;
        const double gx = fmax(fmax(b[0] - qx, qx - b[3]), 0.0);
        const double gy = fmax(fmax(b[1] - qy, qy - b[4]), 0.0);
        const double gz = fmax(fmax(b[2] - qz, qz - b[5]), 0.0);
        const double lb = gx * gx + gy * gy + gz * gz;
        const int need = live && (want_idx ? lb <= best : lb < best);
        if (!__syncthreads_or(need)) continue;

        const int base = t * CMS_TILE;
        const int n = min(CMS_TILE, n_ref - base);
        for (int s = threadIdx.x; s < 3 * n; s += blockDim.x) {
            CMS_ASSERT(3 * base + s < 3 * n_ref);
            tile[s] = ref[3 * base + s];
        }
        if (want_idx) {
            for (int s = threadIdx.x; s < n; s += blockDim.x) tile_idx[s] = ref_index[base + s];
        }
        __syncthreads();
        if (need) {
            for (int p = 0; p < n; ++p) {
                const double d2 = sqdist_seq(tile[3 * p + 0], tile[3 * p + 1], tile[3 * p + 2],
                                             qx, qy, qz);
                if (d2 < best || (want_idx && d2 == best && tile_idx[p] < best_i)) {
                    best = d2;
                    if (want_idx) best_i = tile_idx[p];
                }
            }
        }
        __syncthreads();
    }
    if (live) {
        out[i] = best;
        if (want_idx) out_idx[i] = best_i == 0x7fffffff ? -1 : best_i;
    }
}


// For every point p, test it against the padded neighbour row `row[p]` of a
// (R, K) neighbour table:  collide if any slot k in [start, nn[row]) whose
// atom id is not excl1[p] / excl2[p] satisfies  d2 <= (rad + rp)^2
// (or d2 < (rad + rp)^2 when strict).  `order` selects the 3-term summation
// order of d2 so it matches the NumPy expression being replaced.
extern "C" __global__
void list_collision(const double* __restrict__ pts, const long long n_pts,
                    const int* __restrict__ row,
                    const double* __restrict__ t_xyz,
                    const double* __restrict__ t_rad,
                    const int* __restrict__ t_natom,
                    const int* __restrict__ t_nn,
                    const int n_rows, const int K, const int start,
                    const int* __restrict__ excl1, const int* __restrict__ excl2,
                    const double rp, const int strict, const int order,
                    signed char* __restrict__ out)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pts) return;

    const int r = row[p];
    CMS_ASSERT(r >= 0 && r < n_rows);
    const int nn = t_nn[r];
    CMS_ASSERT(nn <= K);
    const int e1 = excl1 ? excl1[p] : -1;
    const int e2 = excl2 ? excl2[p] : -1;
    const double px = pts[3 * p + 0], py = pts[3 * p + 1], pz = pts[3 * p + 2];

    signed char hit = 0;
    for (int k = start; k < nn; ++k) {
        const long long s = (long long)r * K + k;
        const int nat = t_natom[s];
        if (nat == e1 || nat == e2) continue;
        const double d2 = order
            ? sqdist_alt(px, py, pz, t_xyz[3 * s + 0], t_xyz[3 * s + 1], t_xyz[3 * s + 2])
            : sqdist_seq(px, py, pz, t_xyz[3 * s + 0], t_xyz[3 * s + 1], t_xyz[3 * s + 2]);
        const double t = t_rad[s] + rp;
        const double lim = t * t;
        if (strict ? (d2 < lim) : (d2 <= lim)) { hit = 1; break; }
    }
    out[p] = hit;
}


// Deterministic per-segment sums: out[a] = sum of w[order[q]] for q in
// [offset[a], offset[a+1]), accumulated left to right starting from 0.0.
// With `order` a stable argsort of the bin indices this reproduces
// np.bincount(idx, weights=w) bit for bit.
extern "C" __global__
void segment_sum(const long long* __restrict__ order,
                 const double* __restrict__ w,
                 const long long* __restrict__ offset,
                 const int n_out, const long long n_w,
                 double* __restrict__ out)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_out) return;
    double s = 0.0;
    for (long long q = offset[a]; q < offset[a + 1]; ++q) {
        CMS_ASSERT(q < n_w);
        const long long src = order[q];
        CMS_ASSERT(src >= 0 && src < n_w);
        s += w[src];
    }
    out[a] = s;
}
'''


_module = None
_module_lock = threading.Lock()


def _get(name):
    global _module
    if _module is None:
        with _module_lock:
            if _module is None:
                _module = cp.RawModule(code=_RAW_SOURCE, options=_RAW_OPTIONS)
    return _module.get_function(name)


def _grid(n):
    return ((int(n) + _BLOCK - 1) // _BLOCK,)


# ── elementwise helpers (exact NumPy operation order, no FMA) ──────────────

_dot3_seq_k = cp.ElementwiseKernel(
    'float64 a0, float64 a1, float64 a2, float64 b0, float64 b1, float64 b2',
    'float64 y', 'y = a0 * b0 + a1 * b1 + a2 * b2',
    'cms_dot3_seq', options=_EW_OPTIONS)

_dot3_alt_k = cp.ElementwiseKernel(
    'float64 a0, float64 a1, float64 a2, float64 b0, float64 b1, float64 b2',
    'float64 y', 'y = (a0 * b0 + a2 * b2) + a1 * b1',
    'cms_dot3_alt', options=_EW_OPTIONS)

_norm3_k = cp.ElementwiseKernel(
    'float64 a0, float64 a1, float64 a2',
    'float64 y', 'y = sqrt(a0 * a0 + a1 * a1 + a2 * a2)',
    'cms_norm3', options=_EW_OPTIONS)

_cross3_k = cp.ElementwiseKernel(
    'float64 a0, float64 a1, float64 a2, float64 b0, float64 b1, float64 b2',
    'float64 c0, float64 c1, float64 c2',
    '''
    c0 = a1 * b2 - a2 * b1;
    c1 = a2 * b0 - a0 * b2;
    c2 = a0 * b1 - a1 * b0;
    ''',
    'cms_cross3', options=_EW_OPTIONS)


def dot3_seq(a, b):
    """np.sum(a * b, axis=-1) for 3-vectors."""
    return _dot3_seq_k(a[..., 0], a[..., 1], a[..., 2], b[..., 0], b[..., 1], b[..., 2])


def dot3_einsum(a, b):
    """np.einsum('...i,...i->...', a, b) for 3-vectors."""
    return _dot3_alt_k(a[..., 0], a[..., 1], a[..., 2], b[..., 0], b[..., 1], b[..., 2])


def sumsq3(a):
    """np.square(a).sum(axis=-1) for 3-vectors."""
    return dot3_seq(a, a)


def norm3(a):
    """np.linalg.norm(a, axis=-1) for 3-vectors."""
    return _norm3_k(a[..., 0], a[..., 1], a[..., 2])


def cross3(a, b):
    """np.cross(a, b) for 3-vectors."""
    c0, c1, c2 = _cross3_k(a[..., 0], a[..., 1], a[..., 2], b[..., 0], b[..., 1], b[..., 2])
    return cp.stack([c0, c1, c2], axis=-1)


# ── raw kernel wrappers ────────────────────────────────────────────────────

_morton_k = cp.ElementwiseKernel(
    'float64 x, float64 y, float64 z, raw float64 lo, raw float64 scale',
    'uint32 code',
    'code = (cms_spread(cms_cell(x, lo[0], scale[0])) << 2)'
    '     | (cms_spread(cms_cell(y, lo[1], scale[1])) << 1)'
    '     |  cms_spread(cms_cell(z, lo[2], scale[2]));',
    'cms_morton',
    preamble=r'''
    __device__ unsigned int cms_cell(double v, double lo, double scale) {
        const double t = (v - lo) * scale;
        if (!(t > 0.0)) return 0u;
        if (t >= 1023.0) return 1023u;
        return (unsigned int)t;
    }
    __device__ unsigned int cms_spread(unsigned int v) {
        v &= 0x3ffu;
        v = (v | (v << 16)) & 0x030000FFu;
        v = (v | (v << 8))  & 0x0300F00Fu;
        v = (v | (v << 4))  & 0x030C30C3u;
        v = (v | (v << 2))  & 0x09249249u;
        return v;
    }
    ''')

_TILE = 256
_PRUNE_MIN_REF = 2048


def min_sqdist(query, ref, prune=None, return_index=False):
    """
    cdist(ref, query, 'sqeuclidean').min(axis=0) without the matrix; with
    return_index=True also the .argmin(axis=0) (first index on ties).

    Large reference sets use the spatially pruned kernel, which returns
    exactly the same values (and indices) as the brute-force one (see the
    CUDA source).
    """
    query = cp.ascontiguousarray(query, dtype=cp.float64)
    ref   = cp.ascontiguousarray(ref,   dtype=cp.float64)
    nq, nr = query.shape[0], ref.shape[0]
    out = cp.empty(nq, dtype=cp.float64)
    idx = cp.empty(nq, dtype=cp.int32) if return_index else None
    if nq == 0:
        return (out, idx) if return_index else out
    if prune is None:
        prune = nr >= _PRUNE_MIN_REF
    if not prune or nr == 0:
        _get('min_sqdist')(_grid(nq), (_BLOCK,),
                           (query, cp.int32(nq), ref, cp.int32(nr), out,
                            idx if return_index else 0))
        return (out, idx) if return_index else out

    # Morton-sort both point sets (ordering only; values are untouched)
    lo    = cp.minimum(query.min(axis=0), ref.min(axis=0))
    hi    = cp.maximum(query.max(axis=0), ref.max(axis=0))
    scale = 1023.0 / cp.maximum(hi - lo, 1e-12)
    code_q = _morton_k(query[:, 0], query[:, 1], query[:, 2], lo, scale)
    code_r = _morton_k(ref[:, 0], ref[:, 1], ref[:, 2], lo, scale)
    order_q = cp.argsort(code_q)
    order_r = cp.argsort(code_r)
    q_s    = query[order_q]
    r_s    = ref[order_r]
    code_r = code_r[order_r]

    n_tiles = (nr + _TILE - 1) // _TILE
    box = cp.empty((n_tiles, 6), dtype=cp.float64)
    _get('tile_bounds')(_grid(n_tiles), (_BLOCK,), (r_s, cp.int32(nr), cp.int32(n_tiles), box))

    # each block starts at the reference tile nearest (in Morton order) to its middle query
    n_blocks = (nq + _BLOCK - 1) // _BLOCK
    mid   = cp.minimum(cp.arange(n_blocks) * _BLOCK + _BLOCK // 2, nq - 1)
    start = cp.searchsorted(code_r, code_q[order_q][mid]) // _TILE
    start = cp.clip(start, 0, n_tiles - 1).astype(cp.int32)

    out_s = cp.empty(nq, dtype=cp.float64)
    idx_s = cp.empty(nq, dtype=cp.int32) if return_index else None
    ref_index = order_r.astype(cp.int32) if return_index else None
    _get('min_sqdist_pruned')((n_blocks,), (_BLOCK,),
                              (q_s, cp.int32(nq), r_s, cp.int32(nr),
                               box, cp.int32(n_tiles), start,
                               ref_index if return_index else 0,
                               out_s, idx_s if return_index else 0))
    out[order_q] = out_s
    if not return_index:
        return out
    idx[order_q] = idx_s
    return out, idx


def neighbor_count(xyz, radius, mol, atten, two_rp, bb2):
    n = xyz.shape[0]
    n_neigh  = cp.empty(n, dtype=cp.int32)
    n_buried = cp.empty(n, dtype=cp.int32)
    n_bb     = cp.empty(n, dtype=cp.int32)
    coin_j   = cp.empty(n, dtype=cp.int32)
    _get('neighbor_count')(_grid(32 * n), (_BLOCK,),
                           (xyz, radius, mol, atten, cp.int32(n),
                            cp.float64(two_rp), cp.float64(bb2),
                            n_neigh, n_buried, n_bb, coin_j))
    return n_neigh, n_buried, n_bb, coin_j


def neighbor_emit(xyz, radius, mol, atten, two_rp, neigh_offset, buried_offset):
    n = xyz.shape[0]
    n_neigh_total  = int(neigh_offset[-1])
    n_buried_total = int(buried_offset[-1])
    neigh_j  = cp.empty(max(n_neigh_total, 1),  dtype=cp.int32)
    neigh_d2 = cp.empty(max(n_neigh_total, 1),  dtype=cp.float64)
    buried_j = cp.empty(max(n_buried_total, 1), dtype=cp.int32)
    _get('neighbor_emit')(_grid(32 * n), (_BLOCK,),
                          (xyz, radius, mol, atten, cp.int32(n), cp.float64(two_rp),
                           neigh_offset, buried_offset,
                           cp.int64(n_neigh_total), cp.int64(n_buried_total),
                           neigh_j, neigh_d2, buried_j))
    return neigh_j[:n_neigh_total], neigh_d2[:n_neigh_total], buried_j[:n_buried_total]


def list_collision(pts, row, table, rp, start=0, excl1=None, excl2=None,
                   strict=False, order=ORDER_SEQ):
    """
    Boolean (n_pts,) array: does point p collide with any atom of neighbour
    row `row[p]` of `table` (a _DevNeighbors)?  See the CUDA source for the
    exact rule.
    """
    pts = cp.ascontiguousarray(pts, dtype=cp.float64)
    n = pts.shape[0]
    out = cp.zeros(n, dtype=cp.int8)
    if n == 0:
        return out.astype(bool)
    row = cp.ascontiguousarray(row, dtype=cp.int32)
    e1 = cp.ascontiguousarray(excl1, dtype=cp.int32) if excl1 is not None else None
    e2 = cp.ascontiguousarray(excl2, dtype=cp.int32) if excl2 is not None else None
    n_rows, K = table.natom.shape
    _get('list_collision')(_grid(n), (_BLOCK,),
                           (pts, cp.int64(n), row,
                            table.xyz, table.radius, table.natom, table.nneighbors,
                            cp.int32(n_rows), cp.int32(K), cp.int32(start),
                            e1 if e1 is not None else 0, e2 if e2 is not None else 0,
                            cp.float64(rp), cp.int32(1 if strict else 0), cp.int32(order),
                            out))
    return out.astype(bool)


def bincount_weights(idx, w, n_out):
    """Deterministic np.bincount(idx, weights=w, minlength=n_out)."""
    idx = cp.ascontiguousarray(idx)
    w   = cp.ascontiguousarray(w, dtype=cp.float64)
    n_w = idx.shape[0]
    if n_w:
        n_out = max(int(n_out), int(idx.max()) + 1)
    out = cp.zeros(n_out, dtype=cp.float64)
    if n_w == 0 or n_out == 0:
        return out
    order  = cp.argsort(idx).astype(cp.int64)          # CuPy's argsort is stable
    counts = cp.bincount(idx, minlength=n_out)
    offset = cp.zeros(n_out + 1, dtype=cp.int64)
    cp.cumsum(counts, out=offset[1:])
    _get('segment_sum')(_grid(n_out), (_BLOCK,),
                        (order, w, offset, cp.int32(n_out), cp.int64(n_w), out))
    return out
