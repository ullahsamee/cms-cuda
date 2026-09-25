#!/usr/bin/env python
"""
Calculate contact molecular surface (CMS) and shape complementarity (SC) for
one or more complex structures (PDB or mmCIF).

Each structure is split into a *binder* and a *target* group of heavy atoms,
chosen by chain ID and/or residue name.  By convention CMS is reported on the
target side; the binder-side CMS is reported too.  SC is symmetric.

Examples
--------
  # peptide / mini-protein / antibody (chains H+L) binding a protein
  python examples/calculate_cms_sc.py complex.pdb --binder-chains B --target-chains A
  python examples/calculate_cms_sc.py fab.cif --binder-chains H,L --target-chains A

  # protein binding DNA or RNA (both strands as the target)
  python examples/calculate_cms_sc.py 1AAY.pdb --binder-chains A --target-chains B,C

  # protein designed to bind a small molecule: the ligand is the target
  python examples/calculate_cms_sc.py 1STP.pdb --binder-chains A --binder-exclude BTN \\
          --target-resnames BTN --max-cms

  # many designs at once, results to CSV, per-residue CMS tables
  python examples/calculate_cms_sc.py designs/*.pdb --binder-chains B --target-chains A \\
          --csv cms_sc.csv --per-residue per_residue/

See docs/TUTORIAL.md for what the numbers mean.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

from cms_cuda import (MolecularSurfaceCalculator, calculate_maximum_possible_contact_ms,
                           get_radii_from_names, gpu_available)

WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}


def load_atoms(path, chains=None, resnames=None, exclude_resnames=(),
               keep_water=False, keep_ions=False, keep_hydrogens=False):
    """
    Heavy atoms of ONE side of an interface, ready for cms_cuda.

    path             : .pdb / .ent / .cif / .mmcif file
    chains           : chain IDs to include, e.g. "A" or ["H", "L"] (None = all chains)
    resnames         : only these residue names, e.g. {"BTN"} for a ligand (None = all)
    exclude_resnames : residue names to drop, e.g. {"BTN"} to take the protein without its ligand
    keep_water       : keep HOH/WAT/... (default: drop)
    keep_ions        : keep single-atom HETATM residues such as ZN, MG, CA, NA, CL (default: drop)
    keep_hydrogens   : keep H/D atoms (default: drop -- CMS/SC are defined on heavy atoms)

    Returns xyz (N, 3) float64, radii (N,) float64, and info: one
    (chain, residue number + insertion code, residue name, atom name, element)
    tuple per atom.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    if isinstance(chains, str):
        chains = [c for c in chains.split(",") if c] if "," in chains else list(chains)
    is_cif = str(path).lower().endswith((".cif", ".mmcif"))
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    model = parser.get_structure("s", str(path))[0]          # first model only (NMR / multi-model files)

    xyz, res_names, atom_names, info = [], [], [], []
    for chain in model:
        if chains is not None and chain.id not in chains:
            continue
        for residue in chain:                                # one altloc per atom (Biopython's default choice)
            resname = residue.get_resname().strip()
            hetflag = residue.id[0]
            if resnames is not None and resname not in resnames:
                continue
            if resname in exclude_resnames:
                continue
            if not keep_water and (hetflag == "W" or resname in WATER_NAMES):
                continue
            if not keep_ions and hetflag.startswith("H_") and len(residue) == 1:
                continue
            for atom in residue:
                name = atom.get_name().strip()
                element = (atom.element or "").strip().upper()
                is_h = element in ("H", "D") or (
                    element in ("", "X") and name.lstrip("0123456789").startswith(("H", "D")))
                if is_h and not keep_hydrogens:
                    continue
                xyz.append(atom.coord)
                res_names.append(resname)
                atom_names.append(name)
                info.append((chain.id, f"{residue.id[1]}{residue.id[2].strip()}", resname, name, element))

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    radii = get_radii_from_names(res_names, atom_names)
    return xyz, radii, info


def unmatched_atoms(info, radii):
    """(residue name, atom name) pairs that got radius 0 from the CMS radius table."""
    return sorted({(i[2], i[3]) for i, r in zip(info, radii) if r == 0.0})


def per_residue(info, per_atom):
    """Sum a per-atom array over residues -> list of (chain, resnum, resname, value)."""
    sums, order = {}, []
    for (chain, resnum, resname, _, _), v in zip(info, per_atom):
        key = (chain, resnum, resname)
        if key not in sums:
            sums[key] = 0.0
            order.append(key)
        sums[key] += float(v)
    return [(*k, sums[k]) for k in order]


def make_calculator(device):
    if device == "cpu" or (device == "auto" and not gpu_available()):
        return MolecularSurfaceCalculator(), "cpu"
    if not gpu_available():
        sys.exit("--device cuda requested but no usable CUDA GPU / CuPy was found")
    from cms_cuda import MolecularSurfaceCalculatorGPU
    return MolecularSurfaceCalculatorGPU(), "cuda"


def split(s):
    return [x for x in s.split(",") if x] if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structures", nargs="+", help="PDB / mmCIF files of the complex")
    for side in ("binder", "target"):
        ap.add_argument(f"--{side}-chains", help=f"comma-separated chain IDs of the {side} (default: all)")
        ap.add_argument(f"--{side}-resnames", help=f"only these residue names in the {side}, e.g. BTN")
        ap.add_argument(f"--{side}-exclude", help=f"residue names to drop from the {side}, e.g. BTN")
    ap.add_argument("--keep-water", action="store_true", help="keep water molecules")
    ap.add_argument("--keep-ions", action="store_true", help="keep single-atom ions (most get radius 0)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--no-sc", action="store_true", help="skip shape complementarity")
    ap.add_argument("--max-cms", action="store_true",
                    help="also compute the maximum possible CMS (total molecular surface) of each side")
    ap.add_argument("--csv", help="write one row per structure to this CSV file")
    ap.add_argument("--per-residue", metavar="DIR", help="write per-residue CMS tables to this directory")
    args = ap.parse_args()

    if not (args.binder_chains or args.binder_resnames) or not (args.target_chains or args.target_resnames):
        ap.error("select both sides, e.g. --binder-chains B --target-chains A")

    def sel(side):
        return dict(chains=split(getattr(args, f"{side}_chains")),
                    resnames=set(split(getattr(args, f"{side}_resnames")) or []) or None,
                    exclude_resnames=set(split(getattr(args, f"{side}_exclude")) or []),
                    keep_water=args.keep_water, keep_ions=args.keep_ions)

    if args.per_residue:
        os.makedirs(args.per_residue, exist_ok=True)

    rows = []
    for path in args.structures:
        name = os.path.basename(path)
        try:
            t0 = time.perf_counter()
            bx, br, binfo = load_atoms(path, **sel("binder"))
            tx, tr, tinfo = load_atoms(path, **sel("target"))
            if len(bx) == 0 or len(tx) == 0:
                raise ValueError(f"empty selection (binder {len(bx)} atoms, target {len(tx)} atoms)")
            for side, info, radii in (("binder", binfo, br), ("target", tinfo, tr)):
                bad = unmatched_atoms(info, radii)
                if bad:
                    print(f"  warning: {name} {side}: {int((radii == 0).sum())} atoms have radius 0 "
                          f"(no entry in the radius table, they are invisible): {bad[:8]}", file=sys.stderr)

            calc, device = make_calculator(args.device)
            calc.add_binder_and_target(bx, br, tx, tr)
            cms, per_atom_target = calc.CalcLoaded()
            cms_binder, per_atom_binder = calc.calc_contact_molecular_surface(target_side=False)
            row = dict(structure=name, device=device, binder_atoms=len(bx), target_atoms=len(tx),
                       cms=round(cms, 4), cms_binder=round(cms_binder, 4))
            if not args.no_sc:
                sc, sc_area, median_dist = calc.CalcLoadedSC()
                row.update(sc=round(sc, 4), sc_area=round(sc_area, 4), sc_median_dist=round(median_dist, 4))
            if args.max_cms:
                max_t = calculate_maximum_possible_contact_ms(tx, tr, device=device)[0]
                max_b = calculate_maximum_possible_contact_ms(bx, br, device=device)[0]
                row.update(max_cms_target=round(max_t, 4), max_cms_binder=round(max_b, 4),
                           cms_fraction_target=round(cms / max_t, 4) if max_t else 0.0,
                           cms_fraction_binder=round(cms_binder / max_b, 4) if max_b else 0.0)
            row["seconds"] = round(time.perf_counter() - t0, 3)

            if args.per_residue:
                base = os.path.splitext(name)[0]
                for side, info, values in (("target", tinfo, per_atom_target), ("binder", binfo, per_atom_binder)):
                    with open(os.path.join(args.per_residue, f"{base}_{side}_cms.csv"), "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["chain", "resnum", "resname", "cms"])
                        w.writerows((c, n, r, round(v, 4)) for c, n, r, v in per_residue(info, values))
        except Exception as exc:                 # keep going through a batch
            row = dict(structure=name, error=str(exc))
            print(f"  error: {name}: {exc}", file=sys.stderr)
        rows.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()))

    if args.csv:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
