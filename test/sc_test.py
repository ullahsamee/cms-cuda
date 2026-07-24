#!/usr/bin/env python
"""
Regression test for MolecularSurfaceCalculator's shape complementarity (SC) path.

This mirrors cms_test.py exactly, but drives calculate_shape_complementarity()
(CalcLoadedSC()) instead of calculate_contact_ms() (CalcLoaded()). Because SC
is the path that populates run.results.surface[*].{d_mean, d_median, s_mean,
s_median, trimmedArea, nAllDots, nTrimmedDots} and run.trimmed_dots[0]/[1]
(all left at their zero/empty defaults by the CMS path), this test exercises
those fields in addition to everything cms_test.py already covers.

Usage:
    Run from the project root so that sc_radii.lib is on the working directory:
        python test/sc_test.py

    To save the golden file:
        Set save_golden = True below and run once, then set it back to False.
"""
import os
import sys
import pickle

# Run from project root so sc_radii.lib can be found
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TEST_DIR)
os.chdir(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

# Importing py_contact_ms initialises PyRosetta at module level
from py_contact_ms import partition_pose, calculate_shape_complementarity
from pyrosetta import pose_from_file
from pyrosetta import init
init('-mute all')

# ── Configuration ─────────────────────────────────────────────────────────────

save_golden = False
TOLERANCE   = 0.01          # 1 %

TEST_PDB    = os.path.join(_TEST_DIR, 'sc_example.pdb')
GOLDEN_PKL  = os.path.join(_TEST_DIR, 'sc_golden.pkl')


# ── TestCalculatorRun ─────────────────────────────────────────────────────────

class TestCalculatorRun:
    """
    Serialisable snapshot of all self.run variables from a
    MolecularSurfaceCalculator, ready for pickle round-trips.

    Because run.atoms / run.dots / run.probes are SoA containers backed by
    numpy arrays, this class reads those arrays directly — no object traversal
    or id-to-index mapping needed.
    """

    def __init__(self, calc, sc_value):
        run = calc.run

        # ── top-level scalars ──────────────────────────────────────────────
        self.sc         = sc_value
        self.radmax     = run.radmax

        # ── results ───────────────────────────────────────────────────────
        r = run.results
        self.results = {
            'sc':            r.sc,
            'area':          r.area,
            'distance':      r.distance,
            'perimeter':     r.perimeter,
            'nAtoms':        r.nAtoms,
            'valid':         r.valid,
            'dots_convex':   r.dots.convex,
            'dots_toroidal': r.dots.toroidal,
            'dots_concave':  r.dots.concave,
        }
        for i, surf in enumerate(r.surface):
            p = f'surface_{i}'
            self.results.update({
                f'{p}_nAtoms':          surf.nAtoms,
                f'{p}_nBuriedAtoms':    surf.nBuriedAtoms,
                f'{p}_nBlockedAtoms':   surf.nBlockedAtoms,
                f'{p}_nAllDots':        surf.nAllDots,
                f'{p}_nTrimmedDots':    surf.nTrimmedDots,
                f'{p}_nBuriedDots':     surf.nBuriedDots,
                f'{p}_nAccessibleDots': surf.nAccessibleDots,
                f'{p}_trimmedArea':     surf.trimmedArea,
                f'{p}_d_mean':          surf.d_mean,
                f'{p}_d_median':        surf.d_median,
                f'{p}_s_mean':          surf.s_mean,
                f'{p}_s_median':        surf.s_median,
            })

        # ── atoms ─────────────────────────────────────────────────────────
        # Read directly from AtomArray numpy arrays; per-atom neighbor / buried
        # lists are on the cached AtomView objects.
        a = run.atoms
        n = len(a)
        self.atoms = [
            {
                'natom':       int(a.natom[i]),
                'molecule':    int(a.molecule[i]),
                'radius':      float(a.radius[i]),
                'density':     float(a.density[i]),
                'atten':       int(a.atten[i]),
                'access':      int(a.access[i]),
                'x':           float(a.xyz[i,0]),
                'y':           float(a.xyz[i,1]),
                'z':           float(a.xyz[i,2]),
                'n_neighbors': int(calc.run.neighbor_array.nneighbors[i]),
                'n_buried':    int(calc.run.buried_array.nneighbors[i]),
            }
            for i in range(n)
        ]

        # ── dots ──────────────────────────────────────────────────────────
        # DotArray stores atom_idx directly; no id() lookup needed.
        self.dots = [[], []]
        for mol in range(2):
            d = run.dots[mol]
            nd = len(d)
            self.dots[mol] = [
                {
                    'coor_x':   float(d.coor_xyz[i, 0]),
                    'coor_y':   float(d.coor_xyz[i, 1]),
                    'coor_z':   float(d.coor_xyz[i, 2]),
                    'area':     float(d.area[i]),
                    'buried':   int(d.buried[i]),
                    'type':     int(d.type_[i]),
                    'atom_idx': int(d.atom_idx[i]),
                }
                for i in range(nd)
            ]

        # ── trimmed_dots ──────────────────────────────────────────────────
        self.trimmed_dots = [[], []]
        for mol in range(2):
            d = run.trimmed_dots[mol]
            nd = len(d)
            self.trimmed_dots[mol] = [
                {
                    'coor_x':   float(d.coor_xyz[i, 0]),
                    'coor_y':   float(d.coor_xyz[i, 1]),
                    'coor_z':   float(d.coor_xyz[i, 2]),
                    'area':     float(d.area[i]),
                    'buried':   int(d.buried[i]),
                    'type':     int(d.type_[i]),
                    'atom_idx': int(d.atom_idx[i]),
                }
                for i in range(nd)
            ]

        # ── probes ────────────────────────────────────────────────────────
        # ProbeArray stores atom indices directly in atom_idx_{0,1,2}.
        p  = run.probes
        np_ = len(p)
        self.probes = [
            {
                'atom_indices': [int(p.atom_idx_0[i]),
                                 int(p.atom_idx_1[i]),
                                 int(p.atom_idx_2[i])],
                'height':  float(p.height[i]),
                'point_x': float(p.point_xyz[i,0]),
                'point_y': float(p.point_xyz[i,1]),
                'point_z': float(p.point_xyz[i,2]),
                'alt_x':   float(p.alt_xyz[i,0]),
                'alt_y':   float(p.alt_xyz[i,1]),
                'alt_z':   float(p.alt_xyz[i,2]),
            }
            for i in range(np_)
        ]


# ── Comparison helpers ────────────────────────────────────────────────────────

def _rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-10)
    return abs(float(a) - float(b)) / denom


def _check_scalar(name, cur, gold, tol, out):
    if isinstance(cur, str) or isinstance(gold, str):
        if cur != gold:
            out.append(f"  {name}: {gold!r} -> {cur!r}")
        return
    rd = _rel_diff(cur, gold)
    if rd > tol:
        sign = '+' if cur > gold else ''
        pct  = 100 * (cur - gold) / max(abs(gold), 1e-10)
        out.append(f"  {name}: {gold:.6g} -> {cur:.6g} ({sign}{pct:.2f}%)")


def _check_dict_list(name, cur_list, gold_list, numeric_keys, tol, out,
                     sort_keys=None):
    """
    Compare two parallel lists of dicts.  Reports count mismatches and, for
    each numeric key, how many elements exceed the tolerance and the worst
    relative difference seen.

    If *sort_keys* is provided (a sequence of dict keys), both lists are
    sorted by those keys before comparison so element order is irrelevant.
    """
    if len(cur_list) != len(gold_list):
        out.append(f"  {name} count: {len(gold_list)} -> {len(cur_list)}")
        return

    if sort_keys:
        cur_list  = sorted(cur_list,  key=lambda d: tuple(d[k] for k in sort_keys))
        gold_list = sorted(gold_list, key=lambda d: tuple(d[k] for k in sort_keys))

    n = len(cur_list)
    for key in numeric_keys:
        n_diff  = 0
        max_rd  = 0.0
        for c, g in zip(cur_list, gold_list):
            if key not in c or key not in g:
                continue
            rd = _rel_diff(c[key], g[key])
            if rd > tol:
                n_diff += 1
                max_rd  = max(max_rd, rd)
        if n_diff:
            out.append(
                f"  {name}[*].{key}: "
                f"{n_diff}/{n} elements differ (max {100*max_rd:.2f}%)"
            )


def compare_runs(current, golden, tol=TOLERANCE):
    """
    Return a list of human-readable difference strings.
    An empty list means the two runs agree within *tol*.
    """
    diffs = []

    # top-level scalars
    _check_scalar('sc',         current.sc,         golden.sc,         tol, diffs)
    _check_scalar('radmax',     current.radmax,     golden.radmax,     tol, diffs)

    # results dict
    all_keys = sorted(set(golden.results) | set(current.results))
    for key in all_keys:
        if key not in current.results:
            diffs.append(f"  results.{key}: missing in current run")
        elif key not in golden.results:
            diffs.append(f"  results.{key}: missing in golden")
        else:
            _check_scalar(f'results.{key}',
                          current.results[key], golden.results[key], tol, diffs)

    # atoms
    _check_dict_list(
        'atoms', current.atoms, golden.atoms,
        ['natom', 'molecule', 'radius', 'density',
         'atten', 'access', 'x', 'y', 'z', 'n_neighbors', 'n_buried'],
        tol, diffs,
    )

    # dots
    dot_keys      = ['coor_x', 'coor_y', 'coor_z', 'area', 'buried', 'type', 'atom_idx']
    dot_sort_keys = ['coor_x', 'coor_y', 'coor_z']
    for mol in range(2):
        _check_dict_list(f'dots[{mol}]',
                         current.dots[mol], golden.dots[mol],
                         dot_keys, tol, diffs,
                         sort_keys=dot_sort_keys)

    # trimmed_dots
    for mol in range(2):
        _check_dict_list(f'trimmed_dots[{mol}]',
                         current.trimmed_dots[mol], golden.trimmed_dots[mol],
                         dot_keys, tol, diffs,
                         sort_keys=dot_sort_keys)

    # probes
    _check_dict_list(
        'probes', current.probes, golden.probes,
        ['height', 'point_x', 'point_y', 'point_z', 'alt_x', 'alt_y', 'alt_z'],
        tol, diffs,
        sort_keys=['point_x', 'point_y', 'point_z'],
    )

    return diffs


# ── Main ──────────────────────────────────────────────────────────────────────

def run_test():
    # 1. Load PDB
    print(f"Loading {TEST_PDB} ...")
    pose = pose_from_file(TEST_PDB)

    # 2. Run calculator
    binder_xyz, binder_radii, target_xyz, target_radii = partition_pose(pose)
    sc, sc_int_area, median_dist, calc = calculate_shape_complementarity(binder_xyz, binder_radii, target_xyz, target_radii)
    print(f"SC value: {sc:.6f}")
    print(f"SC interface area: {sc_int_area:.6f}")
    print(f"SC median distance: {median_dist:.6f}")

    # 3. Snapshot run state
    current_run = TestCalculatorRun(calc, sc)

    # 6. Optionally save golden and exit
    if save_golden:
        with open(GOLDEN_PKL, 'wb') as f:
            pickle.dump(current_run, f)
        print(f"Golden data saved to {GOLDEN_PKL}")
        return

    # 4. Load golden
    if not os.path.exists(GOLDEN_PKL):
        print(
            f"No golden file found at {GOLDEN_PKL}.\n"
            f"Set save_golden = True in {__file__} and run once to create it."
        )
        return

    with open(GOLDEN_PKL, 'rb') as f:
        golden_run = pickle.load(f)

    # 5. Compare
    diffs = compare_runs(current_run, golden_run)

    if diffs:
        print(f"REGRESSION: {len(diffs)} variable(s) differ by more than "
              f"{100*TOLERANCE:.0f}%:")
        for d in diffs:
            print(d)
    else:
        print(f"PASS: all variables match within {100*TOLERANCE:.0f}% tolerance.")


if __name__ == '__main__':
    run_test()
