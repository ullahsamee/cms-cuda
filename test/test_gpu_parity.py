#!/usr/bin/env python
"""
CPU vs GPU parity test (and benchmark) for cms-cuda.

Runs the original NumPy implementation (device="cpu", i.e. the upstream
bcov77/py_contact_ms code in _core.py) and the CuPy/CUDA implementation
(device="cuda") on the same inputs and compares everything the calculators
produce: CMS totals and per-atom arrays (target side, binder side, maximum
possible), shape complementarity (SC, interface area, median distance and every
per-surface statistic), dot / trimmed-dot / probe arrays, atom flags, neighbour
lists and result counters.  It also checks the shared surface cache
(CalcLoaded + CalcLoadedSC on one calculator), and that two GPU runs are
bit-identical.

Inputs:
  * the PDBs in the repository root (parsed with BioPython; chain B = binder,
    chain A = target, hydrogens removed, as in cms_compute_visualize.py)
  * the golden regression sets golden.pkl / sc_golden.pkl from the upstream
    test/ directory, if they can be found (--golden-dir, ./test, or git
    history); their atoms carry the exact input xyz / radii
  * a variant of the first input with some radii set to 0 (upstream keeps
    such atoms so per-atom outputs stay aligned with the input)

Usage:
    python tests/test_gpu_parity.py            # full parity check
    python tests/test_gpu_parity.py --quick    # golden set only (for compute-sanitizer)
    python tests/test_gpu_parity.py --bench    # parity + timings
"""
import argparse
import glob
import io
import os
import pickle
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cms_cuda import (calculate_contact_ms, calculate_shape_complementarity,
                           calculate_maximum_possible_contact_ms,
                           get_radii_from_names, gpu_available, MolecularSurfaceCalculator)

TOL = 1e-9


# ── inputs ────────────────────────────────────────────────────────────────────

def load_pdb(pdb):
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("complex", pdb)
    chains = {}
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent().id
        name = atom.name.strip()
        if atom.element == "H" or name.startswith("H") or (
                len(name) > 1 and name[0].isdigit() and name[1] == "H"):
            continue
        c = chains.setdefault(chain, ([], [], []))
        c[0].append(atom.coord)
        c[1].append(residue.resname)
        c[2].append(name)
    b, t = chains["B"], chains["A"]
    return (np.array(b[0], dtype=np.float64), get_radii_from_names(b[1], b[2]),
            np.array(t[0], dtype=np.float64), get_radii_from_names(t[1], t[2]))


class TestCalculatorRun:
    """Placeholder so the golden pickles (written by upstream test/*.py) load."""


def load_golden(name, golden_dir=None):
    """Find test/<name> on disk or in git history; return (inputs, golden) or None."""
    blob = None
    for d in (golden_dir, os.path.join(_ROOT, 'test')):
        if d and os.path.exists(os.path.join(d, name)):
            blob = open(os.path.join(d, name), 'rb').read()
            break
    if blob is None:
        for ref in ('HEAD', 'upstream/main', 'upstream/master'):
            try:
                blob = subprocess.run(['git', 'show', f'{ref}:test/{name}'], cwd=_ROOT,
                                      capture_output=True, check=True).stdout
                break
            except Exception:
                continue
    if blob is None:
        return None
    import __main__
    __main__.TestCalculatorRun = TestCalculatorRun
    g = pickle.load(io.BytesIO(blob))
    mol = np.array([a['molecule'] for a in g.atoms])
    xyz = np.array([[a['x'], a['y'], a['z']] for a in g.atoms])
    rad = np.array([a['radius'] for a in g.atoms])
    return (xyz[mol == 0], rad[mol == 0], xyz[mol == 1], rad[mol == 1]), g


def with_zero_radii(inputs, every=11):
    bx, br, tx, tr = inputs
    br, tr = br.copy(), tr.copy()
    br[::every] = 0.0
    tr[5::every] = 0.0
    return bx, br, tx, tr


# ── comparison helpers ────────────────────────────────────────────────────────

class Report:
    def __init__(self, name):
        self.name = name
        self.fail = []
        self.info = []

    def check(self, label, ok, detail=''):
        if not ok:
            self.fail.append(f'{label} {detail}'.rstrip())

    def exact(self, label, a, b):
        a, b = np.asarray(a), np.asarray(b)
        ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
        self.check(label, ok, f'(shapes {a.shape} vs {b.shape})' if a.shape != b.shape else '(values differ)')

    def close(self, label, a, b, tol=TOL):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            self.check(label, False, f'(shapes {a.shape} vs {b.shape})')
            return
        if a.size == 0:
            return
        nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
        d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
        scale = np.maximum(1.0, np.maximum(np.abs(np.nan_to_num(a)), np.abs(np.nan_to_num(b))))
        worst = float((d / scale).max())
        self.check(label, nan_ok and worst <= tol, f'(max rel/abs diff {worst:.3g}, nan layout equal {nan_ok})')
        self.info.append((label, worst))


def compare_dots(rep, label, c, g):
    rep.exact(f'len({label})', len(c), len(g))
    if len(c) != len(g):
        return
    n = len(c)
    for f in ('buried', 'type_', 'atom_idx'):
        rep.exact(f'{label}.{f}', getattr(c, f)[:n], getattr(g, f)[:n])
    for f in ('coor_xyz', 'outnml_xyz', 'area'):
        rep.close(f'{label}.{f}', getattr(c, f)[:n], getattr(g, f)[:n])


def compare_calcs(rep, cpu, gpu):
    rc, rg = cpu.run, gpu.run
    rep.exact('radmax', rc.radmax, rg.radmax)
    rep.exact('surfaces_generated', (rc.surfaces_generated, rc.surfaces_generated_all_atoms),
              (rg.surfaces_generated, rg.surfaces_generated_all_atoms))

    for attr in ('nAtoms', 'valid'):
        rep.exact(f'results.{attr}', getattr(rc.results, attr), getattr(rg.results, attr))
    for attr in ('sc', 'area', 'distance', 'perimeter'):
        rep.close(f'results.{attr}', getattr(rc.results, attr), getattr(rg.results, attr))
    for attr in ('convex', 'toroidal', 'concave'):
        rep.exact(f'results.dots.{attr}', getattr(rc.results.dots, attr), getattr(rg.results.dots, attr))
    for i in range(3):
        sc_, sg_ = rc.results.surface[i], rg.results.surface[i]
        for attr in ('nAtoms', 'nBuriedAtoms', 'nBlockedAtoms', 'nAllDots', 'nTrimmedDots',
                     'nBuriedDots', 'nAccessibleDots'):
            rep.exact(f'results.surface[{i}].{attr}', getattr(sc_, attr), getattr(sg_, attr))
        for attr in ('trimmedArea', 'd_mean', 'd_median', 's_mean', 's_median'):
            rep.close(f'results.surface[{i}].{attr}', getattr(sc_, attr), getattr(sg_, attr))

    ac, ag = rc.atoms, rg.atoms
    for f in ('natom', 'molecule', 'atten', 'access', 'radius', 'density', 'xyz'):
        rep.exact(f'atoms.{f}', getattr(ac, f), getattr(ag, f))

    for nm in ('neighbor_array', 'buried_array'):
        c, g = getattr(rc, nm), getattr(rg, nm)
        if c is None or g is None:
            rep.check(nm, c is None and g is None, '(one side missing)')
            continue
        for f in ('nneighbors', 'natom', 'radius', 'xyz'):
            rep.exact(f'{nm}.{f}', getattr(c, f), getattr(g, f))

    for m in (0, 1):
        compare_dots(rep, f'dots[{m}]', rc.dots[m], rg.dots[m])
        compare_dots(rep, f'trimmed_dots[{m}]', rc.trimmed_dots[m], rg.trimmed_dots[m])

    pc, pg = rc.probes, rg.probes
    rep.exact('len(probes)', len(pc), len(pg))
    if len(pc) == len(pg):
        n = len(pc)
        for f in ('atom_idx_0', 'atom_idx_1', 'atom_idx_2'):
            rep.exact(f'probes.{f}', getattr(pc, f)[:n], getattr(pg, f)[:n])
        for f in ('height', 'point_xyz', 'alt_xyz'):
            rep.close(f'probes.{f}', getattr(pc, f)[:n], getattr(pg, f)[:n])

    tc, tg = rc.toroid_queue, rg.toroid_queue
    rep.exact('len(toroid_queue)', len(tc), len(tg))
    if len(tc) == len(tg) and len(tc):
        rep.exact('toroid_queue.natoms', [(t[0], t[1], t[5]) for t in tc], [(t[0], t[1], t[5]) for t in tg])
        rep.close('toroid_queue.uij', np.array([t[2] for t in tc]), np.array([t[2] for t in tg]))
        rep.close('toroid_queue.tij', np.array([t[3] for t in tc]), np.array([t[3] for t in tg]))
        rep.close('toroid_queue.rij', np.array([t[4] for t in tc]), np.array([t[4] for t in tg]))


def gpu_calculator():
    from cms_cuda import MolecularSurfaceCalculatorGPU
    return MolecularSurfaceCalculatorGPU()


def run_case(name, inputs, expect=None, bench=False):
    """expect: dict with any of cms / sc / sc_area / sc_distance reference values."""
    bx, br, tx, tr = inputs
    rep = Report(name)
    sub = []

    # ── CMS (+ binder side) and max possible ──────────────────────────────
    cms_c, pa_c, calc_c = calculate_contact_ms(bx, br, tx, tr, device='cpu')
    cms_g, pa_g, calc_g = calculate_contact_ms(bx, br, tx, tr, device='cuda')
    bcms_c, pb_c = calc_c.calc_contact_molecular_surface(target_side=False)
    bcms_g, pb_g = calc_g.calc_contact_molecular_surface(target_side=False)
    max_c, mpa_c, mcalc_c = calculate_maximum_possible_contact_ms(tx, tr, device='cpu')
    max_g, mpa_g, mcalc_g = calculate_maximum_possible_contact_ms(tx, tr, device='cuda')

    rep.check('return types', type(cms_g) is float and type(bcms_g) is float and type(max_g) is float
              and all(isinstance(a, np.ndarray) for a in (pa_g, pb_g, mpa_g)))
    rep.exact('per-atom lengths == inputs', (len(pa_g), len(pb_g), len(mpa_g)), (len(tx), len(bx), len(tx)))
    rep.close('cms (target)', cms_c, cms_g)
    rep.close('cms (binder)', bcms_c, bcms_g)
    rep.close('cms (max possible)', max_c, max_g)
    rep.close('per-atom cms (target)', pa_c, pa_g)
    rep.close('per-atom cms (binder)', pb_c, pb_g)
    rep.close('per-atom area (max possible)', mpa_c, mpa_g)
    compare_calcs(rep, calc_c, calc_g)
    r = Report('max possible'); compare_calcs(r, mcalc_c, mcalc_g); sub.append(r)

    # ── shape complementarity ─────────────────────────────────────────────
    sc_c = calculate_shape_complementarity(bx, br, tx, tr, device='cpu')
    sc_g = calculate_shape_complementarity(bx, br, tx, tr, device='cuda')
    rep.check('SC return types', all(type(v) is float for v in sc_g[:3]))
    for label, a, b in zip(('sc', 'sc interface area', 'sc median distance'), sc_c[:3], sc_g[:3]):
        rep.close(label, a, b)
    r = Report('SC'); compare_calcs(r, sc_c[3], sc_g[3]); sub.append(r)

    # ── one calculator, CalcLoaded + CalcLoadedSC (shared cached surface) ─
    for dev, make in (('cpu', MolecularSurfaceCalculator), ('cuda', gpu_calculator)):
        calc = make()
        calc.add_binder_and_target(bx, br, tx, tr)
        cms, pa = calc.CalcLoaded()
        sc3 = calc.CalcLoadedSC()
        cms2, pa2 = calc.CalcLoaded()
        ref_cms, ref_sc = (cms_c, sc_c) if dev == 'cpu' else (cms_g, sc_g)
        rep.exact(f'[{dev}] shared calc: CMS == separate run', (cms, cms2), (ref_cms, ref_cms))
        rep.exact(f'[{dev}] shared calc: SC == separate run', sc3, ref_sc[:3])
        try:
            calc.CalcLoadedMaxPossibleCMS()
            rep.check(f'[{dev}] cached surface reuse with all_atoms=True raises', False)
        except RuntimeError:
            pass

    # ── determinism: a second GPU run must be bit-identical ───────────────
    cms_g2, pa_g2, calc_g2 = calculate_contact_ms(bx, br, tx, tr, device='cuda')
    sc_g2 = calculate_shape_complementarity(bx, br, tx, tr, device='cuda')
    rep.exact('GPU determinism: cms', cms_g, cms_g2)
    rep.exact('GPU determinism: per-atom', pa_g, pa_g2)
    rep.exact('GPU determinism: sc', sc_g[:3], sc_g2[:3])
    for m in (0, 1):
        rep.exact(f'GPU determinism: dots[{m}].coor_xyz', calc_g.run.dots[m].coor_xyz,
                  calc_g2.run.dots[m].coor_xyz)

    for key, value in (expect or {}).items():
        got = {'cms': cms_g, 'sc': sc_g[0], 'sc_area': sc_g[1], 'sc_distance': sc_g[2]}[key]
        rep.close(f'{key} vs golden', got, value)

    for r in sub:
        rep.fail += [f'[{r.name}] {f}' for f in r.fail]
        rep.info += r.info

    print(f'\n== {name}: {len(bx)} binder + {len(tx)} target atoms')
    print(f'   CMS     cpu {cms_c:.10f}   gpu {cms_g:.10f}   rel diff {abs(cms_c - cms_g) / max(abs(cms_c), 1e-30):.2e}')
    print(f'   binder  cpu {bcms_c:.10f}   gpu {bcms_g:.10f}')
    print(f'   max     cpu {max_c:.10f}   gpu {max_g:.10f}')
    print(f'   SC      cpu {sc_c[0]:.10f}   gpu {sc_g[0]:.10f}   (area {sc_g[1]:.4f}, median dist {sc_g[2]:.4f})')
    print(f'   dots {len(calc_g.run.dots[0])}/{len(calc_g.run.dots[1])}  probes {len(calc_g.run.probes)}  '
          f'trimmed {len(sc_g[3].run.trimmed_dots[0])}/{len(sc_g[3].run.trimmed_dots[1])}  '
          f'max float diff over all compared values {max([w for _, w in rep.info] or [0.0]):.2e}')

    if bench:
        timings(bx, br, tx, tr)

    if rep.fail:
        print('   FAIL:')
        for f in rep.fail:
            print('     -', f)
    else:
        print('   PASS')
    return not rep.fail


def timings(bx, br, tx, tr, repeat=5):
    from cupyx.profiler import benchmark

    def cpu_time(fn):
        ts = []
        for _ in range(2):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return min(ts)

    def gpu_time(fn):
        # The calculators synchronise before returning numpy results, so the
        # host-side time is the full end-to-end time.
        r = benchmark(fn, n_repeat=repeat, n_warmup=2)
        return float(np.mean(r.cpu_times))

    _, _, calc_g = calculate_contact_ms(bx, br, tx, tr, device='cuda')
    _, _, calc_c = calculate_contact_ms(bx, br, tx, tr, device='cpu')

    rows = [
        ('calculate_contact_ms',
         cpu_time(lambda: calculate_contact_ms(bx, br, tx, tr, device='cpu')),
         gpu_time(lambda: calculate_contact_ms(bx, br, tx, tr, device='cuda'))),
        ('binder side (reuse calc)',
         cpu_time(lambda: calc_c.calc_contact_molecular_surface(target_side=False)),
         gpu_time(lambda: calc_g.calc_contact_molecular_surface(target_side=False))),
        ('calculate_shape_complementarity',
         cpu_time(lambda: calculate_shape_complementarity(bx, br, tx, tr, device='cpu')),
         gpu_time(lambda: calculate_shape_complementarity(bx, br, tx, tr, device='cuda'))),
        ('calculate_maximum_possible_contact_ms',
         cpu_time(lambda: calculate_maximum_possible_contact_ms(tx, tr, device='cpu')),
         gpu_time(lambda: calculate_maximum_possible_contact_ms(tx, tr, device='cuda'))),
    ]
    print('   timings (warm):')
    for label, c, g in rows:
        print(f'     {label:40s} cpu {c * 1e3:9.1f} ms   gpu {g * 1e3:8.1f} ms   x{c / g:6.1f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='golden sets only')
    ap.add_argument('--bench', action='store_true', help='also time CPU vs GPU')
    ap.add_argument('--golden-dir', default=None,
                    help='directory with golden.pkl / sc_golden.pkl (default: ./test or git history)')
    ap.add_argument('pdbs', nargs='*', help='PDB files (default: *.pdb in the repository root)')
    args = ap.parse_args()

    if not gpu_available():
        print('No usable CUDA GPU (CuPy missing or no device) — nothing to compare.')
        return 1

    ok = True
    cases = []

    golden = load_golden('golden.pkl', args.golden_dir)
    if golden is not None:
        inputs, g = golden
        cases.append(('golden.pkl (CMS golden)', inputs, {'cms': g.cms}))
    else:
        print('golden.pkl not found — skipping the CMS golden set')

    sc_golden = load_golden('sc_golden.pkl', args.golden_dir)
    if sc_golden is not None:
        inputs, g = sc_golden
        expect = {'sc': g.sc, 'sc_area': g.results['area'], 'sc_distance': g.results['distance']}
        if cases and all(np.array_equal(a, b) for a, b in zip(cases[0][1], inputs)):
            # upstream's SC golden uses the same structure as the CMS golden
            cases[0] = ('golden.pkl + sc_golden.pkl', cases[0][1], {**cases[0][2], **expect})
        else:
            cases.append(('sc_golden.pkl (SC golden)', inputs, expect))
    else:
        print('sc_golden.pkl not found — skipping the SC golden set (use --golden-dir)')

    if not args.quick:
        pdbs = args.pdbs or sorted(glob.glob(os.path.join(_ROOT, '*.pdb')))
        for pdb in pdbs:
            cases.append((os.path.basename(pdb), load_pdb(pdb), None))

    if not cases:
        print('Nothing to test: pass PDB files (binder = chain B, target = chain A) and/or '
              '--golden-dir pointing at the upstream bcov77/py_contact_ms test/ folder.')
        return 1

    name, inputs, _ = cases[0]
    cases.append((name + ' with some radii = 0', with_zero_radii(inputs), None))

    for name, inputs, expect in cases:
        ok &= run_case(name, inputs, expect=expect, bench=args.bench and 'radii = 0' not in name)

    print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
