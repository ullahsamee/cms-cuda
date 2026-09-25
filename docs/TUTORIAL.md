# Calculating Contact Molecular Surface (CMS) and Shape Complementarity (SC)

A step-by-step guide to computing CMS and Lawrence & Colman SC with `cms-cuda`. It covers what the numbers mean, how to prepare structures, which kinds of complexes work, and how to run many structures on CPU or GPU.

Every number shown below was produced with this package on public PDB entries, so you can reproduce them.

**Contents**

1. [What CMS and SC measure](#1-what-cms-and-sc-measure)
2. [Installation](#2-installation)
3. [Preparing the input (the step that matters most)](#3-preparing-the-input)
4. [Calculating CMS](#4-calculating-cms)
5. [Calculating SC](#5-calculating-sc)
6. [CMS and SC from one calculation](#6-cms-and-sc-from-one-calculation)
7. [Many structures at once](#7-many-structures-at-once)
8. [Which complexes can be analysed?](#8-which-complexes-can-be-analysed)
9. [Settings](#9-settings)
10. [Speed, GPU and large systems](#10-speed-gpu-and-large-systems)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What CMS and SC measure

Both numbers start from the same step: each molecule's surface is turned into **dots**.

### What is a dot?

A dot is one sample point on a molecule's surface. Instead of treating the surface as a smooth sheet, the program covers it evenly with tiny points and does every calculation on them.

1. **Atoms become spheres**, with radii from the radius table (e.g. carbon ≈ 1.85 Å).
2. **A probe ball rolls over the molecule.** Its radius is 1.7 Å, about the size of a water molecule. The skin it traces is the **molecular (solvent-excluded) surface**, with three kinds of patches:
   - *convex:* the probe touches one atom;
   - *toroidal:* it rolls in the groove between two atoms;
   - *concave:* it sits in a pocket touching three atoms.
3. **The skin is covered with dots, 15 per Å².** Each dot stands for a patch of about 0.07 Å², so a surface area is simply the sum of its dots' areas. For example, 1YCR has 16,648 dots on the p53 peptide and 24,138 on MDM2.

Each dot carries:

| property | meaning |
|---|---|
| position | where it sits on the surface (x, y, z) |
| area | how much surface it stands for (~0.07 Å²) |
| normal | an arrow pointing straight out of the surface at that spot |
| atom | the atom it belongs to. Summing an atom's dots gives per-atom (and per-residue) values. |
| **buried** | yes if the probe at this dot would bump into the *partner* molecule, meaning the dot is part of the interface |

The surface is only built near the partner, around atoms within 8 Å of the other molecule; the rest of the molecule still shapes it.

```text
  binder surface   ·  ·  ·  ·  ·  ·     ← buried dots on the binder
                         ↕ d  (gap between the two surfaces)
  target surface   ·  ·  ·  ·  ·  ·     ← buried dots on the target
```

Don't confuse these surface dots with the "·" in `n_A · n_B` below. That is the *dot product* of two normal arrows, a different meaning.

The two scores then read the dots differently:

| | **CMS** — contact molecular surface | **SC** — shape complementarity (Lawrence & Colman 1993) |
|---|---|---|
| Question it answers | *How much* surface is in close contact? | *How well* do the two surfaces fit, regardless of size? |
| Uses | every buried dot on one side | buried dots with the 1.5 Å rim trimmed off, on both sides |
| Per dot | `area × exp(−0.5 · d²)` | `n_A · n_B × exp(−0.5 · d²)`, clipped to ±0.999 |
| Combined by | **sum** over the target's buried dots | **median** per side (negated), then the mean of the two sides |
| Units / range | Å², grows with interface size | unitless, about 0–1 (1 = perfect fit) |
| Per-atom values | yes (per-atom, so also per-residue) | no (whole-interface statistic) |

In the table, `d` is the distance from a dot to the nearest (buried or trimmed) dot on the other molecule. `n_A` and `n_B` are the outward normals at the two dots; complementary surfaces face each other, so `n_A·n_B ≈ −1`.

What the Gaussian `exp(−0.5·d²)` means in practice for CMS: a dot counts fully at `d = 0`, 61 % at 1 Å, 14 % at 2 Å and 1 % at 3 Å. CMS therefore measures the area of the interface that sits within roughly 2 Å of the partner's surface.

**Why both?** SC ignores size: a tiny interface with a perfect fit scores as high as a large one. CMS combines size and tightness of fit, which is why it is the better score for ranking binder designs. SC is still the classic, widely reported fit statistic, and it is useful next to CMS.

---

## 2. Installation

Requirements:
- **CPU:** Python ≥ 3.8, `numpy`, `scipy`. `biopython` is only needed to read structure files, as in the examples below.
- **GPU (optional):** an NVIDIA GPU plus [CuPy](https://cupy.dev); current CuPy needs Python ≥ 3.10.

```bash
# CPU only
pip install "git+https://github.com/ullahsamee/cms-cuda"

# GPU: first check which CUDA version your NVIDIA driver supports
nvidia-smi                      # look for "CUDA Version: 13.x" or "12.x" in the header

# driver supports CUDA 13 -> the [gpu] extra installs cupy-cuda13x
pip install "cms-cuda[gpu] @ git+https://github.com/ullahsamee/cms-cuda"

# driver supports CUDA 12 -> install the CUDA 12 build of CuPy instead
pip install "git+https://github.com/ullahsamee/cms-cuda" cupy-cuda12x
```

Once version 0.2.0 or newer is on PyPI, `pip install cms-cuda` and `pip install "cms-cuda[gpu]"` do the same.

Check the installation:

```bash
python -c "import cms_cuda as p; print(p.__version__, 'GPU available:', p.gpu_available())"
```

To develop, or to run the test:

```bash
git clone https://github.com/ullahsamee/cms-cuda
cd cms-cuda
python -m venv venv && source venv/bin/activate
pip install -e ".[gpu]"                     # or: pip install -e .   (CPU only)
python test/test_gpu_parity.py             # CPU vs GPU comparison (needs a GPU)
```

**PyRosetta is not needed.** It is only used by the optional helper `partition_pose()`, for people who already have a PyRosetta pose.

**The first GPU call** in a fresh environment compiles the CUDA kernels, which takes a few seconds. They are cached in `~/.cupy/kernel_cache`, so later runs start quickly.

---

## 3. Preparing the input

`cms-cuda` needs two groups of atoms, a **binder** and a **target**. Each group is an array of coordinates plus an array of radii:

```python
binder_xyz   # (N0, 3) float, Å
binder_radii # (N0,)   float, from get_radii_from_names()
target_xyz   # (N1, 3)
target_radii # (N1,)
```

The calculation is purely geometric: it only sees atom positions and radii, not bonds or chain topology. Almost everything that can go wrong therefore happens while you build these arrays. Rules:

1. **Use the coordinates of the complex** (the bound pose), from a crystal structure or a prediction (AF2/AF3, Boltz, Chai, RFdiffusion + MPNN, …). Binder and target must be in the same coordinate frame.
2. **Heavy atoms only; remove hydrogens.** The radius table gives H atoms 0.5 Å, which would change the surface.
3. **One model, one conformer.** Take the first model of NMR / multi-model files, and one alternate location (altloc) per atom. Duplicated atoms raise `RuntimeError: Coincident atoms`.
4. **Remove water and ions** unless you really want them (see [§8](#8-which-complexes-can-be-analysed) for the radii ions would get).
5. **Always get radii from `get_radii_from_names(residue_names, atom_names)`.** CMS and SC are defined with this specific radius table (Rosetta's SC radii). Pass stripped names: `"CA"`, not `" CA "`.
6. **Check for atoms with radius 0.** A name that matches nothing in the table gets radius 0. Such an atom is kept, so per-atom arrays stay aligned with your input, but it has no surface and blocks nothing: it is invisible.
7. **Decide which side is the target.** CMS is reported on the target side by convention. SC is symmetric.

### A loader that follows these rules

Copy this function into your script. It is also included in [`examples/calculate_cms_sc.py`](../examples/calculate_cms_sc.py), where you can import it directly. It reads PDB or mmCIF files (for example AlphaFold 3 `.cif` output).

```python
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from cms_cuda import get_radii_from_names

WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}


def load_atoms(path, chains=None, resnames=None, exclude_resnames=(),
               keep_water=False, keep_ions=False, keep_hydrogens=False):
    """
    Heavy atoms of ONE side of an interface, ready for cms_cuda.

    chains           : chain IDs to include, e.g. "A" or ["H", "L"] (None = all chains)
    resnames         : only these residue names, e.g. {"BTN"} for a ligand (None = all)
    exclude_resnames : residue names to drop, e.g. {"BTN"} to take the protein without its ligand
    Returns xyz (N, 3), radii (N,), info = [(chain, resnum, resname, atom, element), ...]
    """
    if isinstance(chains, str):
        chains = [c for c in chains.split(",") if c] if "," in chains else list(chains)
    is_cif = str(path).lower().endswith((".cif", ".mmcif"))
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    model = parser.get_structure("s", str(path))[0]          # first model only

    xyz, res_names, atom_names, info = [], [], [], []
    for chain in model:
        if chains is not None and chain.id not in chains:
            continue
        for residue in chain:                                # one altloc per atom
            resname = residue.get_resname().strip()
            hetflag = residue.id[0]
            if resnames is not None and resname not in resnames:
                continue
            if resname in exclude_resnames:
                continue
            if not keep_water and (hetflag == "W" or resname in WATER_NAMES):
                continue
            if not keep_ions and hetflag.startswith("H_") and len(residue) == 1:
                continue                                     # single-atom HETATM = ion
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
```

With a string, each character is one chain ID (`"HL"` = chains H and L); `"H,L"` also works. Chain IDs longer than one character (common in mmCIF) must be given as a list: `chains=["AA", "AB"]`.

### Choosing binder and target

| System | binder | target |
|---|---|---|
| peptide / cyclic peptide / mini-protein → protein | `chains="B"` | `chains="A"` |
| antibody Fab / scFv / Fv → antigen | `chains="HL"` (heavy + light together) | antigen chain(s) |
| VHH / nanobody → antigen | VHH chain | antigen chain(s) |
| protein → DNA or RNA | protein chain(s) | **both** nucleic-acid strands, e.g. `chains="BC"` |
| protein designed to bind a small molecule | `chains="A", exclude_resnames={"LIG"}` | `resnames={"LIG"}` (ligand as target) |
| small molecule (drug) → protein / DNA / RNA | `resnames={"LIG"}` | the macromolecule, `exclude_resnames={"LIG"}` |
| homodimer | one chain | the other chain |

Crystal structures often contain several copies of a complex. Select one binder–target pair, as in the table.

Download the example structures used below:

```bash
curl -O https://files.rcsb.org/download/1YCR.pdb   # p53 peptide (chain B) bound to MDM2 (chain A)
curl -O https://files.rcsb.org/download/1STP.pdb   # streptavidin (chain A) with biotin (BTN)
```

---

## 4. Calculating CMS

```python
from cms_cuda import calculate_contact_ms

bx, br, binfo = load_atoms("1YCR.pdb", chains="B")    # binder: p53 peptide
tx, tr, tinfo = load_atoms("1YCR.pdb", chains="A")    # target: MDM2

print("radius-0 atoms:", int((br == 0).sum()), int((tr == 0).sum()))   # 0 0

cms, per_atom_target, calc = calculate_contact_ms(bx, br, tx, tr)
print(f"CMS (target side): {cms:.1f} A^2")            # 485.4
print(per_atom_target.shape)                           # (705,) -> one value per target atom, same order as tx
```

`calculate_contact_ms` returns:

| return value | meaning |
|---|---|
| `cms` | contact molecular surface on the **target** side, in Å² (Python `float`) |
| `per_atom_target` | NumPy array, one value per target atom, in the order you passed them. It sums to `cms`. |
| `calc` | the calculator; reuse it for the binder side and for SC without recomputing the surface |

### Binder side

```python
cms_binder, per_atom_binder = calc.calc_contact_molecular_surface(target_side=False)
print(f"CMS (binder side): {cms_binder:.1f} A^2")      # 460.1
```

The two sides differ a little (485.4 vs 460.1) because each is measured on its own surface. Report the target side unless you have a reason not to, and compare like with like.

### Per-residue CMS (hot spots)

```python
import collections

def per_residue(info, per_atom):
    total = collections.defaultdict(float)
    for (chain, resnum, resname, atom, element), value in zip(info, per_atom):
        total[(chain, resnum, resname)] += value
    return sorted(total.items(), key=lambda kv: -kv[1])

for (chain, resnum, resname), value in per_residue(binfo, per_atom_binder)[:4]:
    print(chain, resnum, resname, round(value, 1))
# B 19 PHE 90.4
# B 23 TRP 73.8
# B 29 ASN 62.9
# B 26 LEU 44.4
```

p53's known hot-spot triad (Phe19, Trp23, Leu26) comes out on top. Per-residue CMS is a quick way to see which residues carry an interface.

### Normalising by the maximum possible CMS

`calculate_maximum_possible_contact_ms(xyz, radii)` returns the total molecular surface area of one molecule on its own. No partner is needed: every dot counts with full weight. CMS can never exceed it, so the ratio is the fraction of that molecule's surface in close contact:

```python
from cms_cuda import calculate_maximum_possible_contact_ms

max_target, per_atom_max, _ = calculate_maximum_possible_contact_ms(tx, tr)
max_binder, _, _            = calculate_maximum_possible_contact_ms(bx, br)
print(f"target: {cms / max_target:.1%}   binder: {cms_binder / max_binder:.1%}")
# target: 11.4%   binder: 40.6%
```

This is most useful for small molecules. For a protein designed to bind a ligand, make the ligand the target: CMS / max CMS is then the fraction of the ligand's surface the protein covers. For biotin in streptavidin (1STP) it is 79.9 % (see [§8](#8-which-complexes-can-be-analysed)).

### Reading CMS values

- CMS is an area in Å². It grows with both the size and the tightness of an interface, so compare structures of the same kind (same target, similar binder type).
- It is deterministic: the same input always gives the same number.
- If binder and target are more than 8 Å apart everywhere, no interface surface is built and CMS is 0.

---

## 5. Calculating SC

```python
from cms_cuda import calculate_shape_complementarity

sc, sc_area, median_dist, calc = calculate_shape_complementarity(bx, br, tx, tr)
print(f"SC = {sc:.3f}, interface area = {sc_area:.1f} A^2, median distance = {median_dist:.2f} A")
# SC = 0.764, interface area = 957.5 A^2, median distance = 0.48 A
```

| return value | meaning |
|---|---|
| `sc` | shape complementarity, about 0–1. 1 would be a perfect fit; well-packed protein–protein interfaces are usually about 0.5–0.75. |
| `sc_area` | trimmed interface area, binder + target (Å²): the buried surface minus the 1.5 Å rim |
| `median_dist` | median gap between the two surfaces (Å), averaged over both sides |
| `calc` | the calculator, holding the detailed statistics |

The per-side details are in `calc.run.results.surface[i]`. Index 0 is the binder, 1 the target, and 2 the average of the two, which is what the function returns:

```python
for i, side in enumerate(["binder", "target", "average"]):
    s = calc.run.results.surface[i]
    print(f"{side:8s} S_median={s.s_median:.3f}  S_mean={s.s_mean:.3f}  d_median={s.d_median:.2f}  "
          f"trimmed_area={s.trimmedArea:.1f}  dots={s.nTrimmedDots}/{s.nAllDots}")
# binder   S_median=0.778  S_mean=0.710  d_median=0.45  trimmed_area=446.9  dots=6661/16648
# target   S_median=0.750  S_mean=0.663  d_median=0.51  trimmed_area=510.6  dots=7742/24138
# average  S_median=0.764  S_mean=0.687  d_median=0.48  trimmed_area=957.5  dots=14403/40786
```

`sc` is `surface[2].s_median`. The median is a binned (0.02-wide bins), interpolated median, as in Rosetta's implementation.

### Reading SC values

- SC measures fit only, not size. A small interface can score as high as a large one. Look at `sc_area` or CMS alongside it.
- The familiar reference ranges come from protein–protein interfaces. In the original paper, protease–inhibitor and oligomer interfaces scored about 0.70–0.76 and antibody–antigen interfaces somewhat lower, about 0.64–0.68. For DNA, RNA or small-molecule interfaces, compare with values computed the same way for similar complexes rather than with those ranges.
- `sc = 0` together with `sc_area = 0` means there was no interface: nothing within 8 Å, or no buried dots left after trimming.
- The algorithm and its defaults follow Rosetta's implementation, but this is an independent port. Expect values very close to Rosetta's `ShapeComplementarityCalculator`, not necessarily identical to the last digit. Compare numbers computed with the same tool.

---

## 6. CMS and SC from one calculation

CMS and SC are two readouts of the same surface. With one calculator the surface is built once:

```python
from cms_cuda import MolecularSurfaceCalculator      # CPU
# from cms_cuda import MolecularSurfaceCalculatorGPU as MolecularSurfaceCalculator   # GPU

calc = MolecularSurfaceCalculator()
calc.add_binder_and_target(bx, br, tx, tr)

cms, per_atom_target = calc.CalcLoaded()                                    # CMS, target side
cms_binder, per_atom_binder = calc.calc_contact_molecular_surface(target_side=False)
sc, sc_area, median_dist = calc.CalcLoadedSC()                              # SC (reuses the surface)
```

`CalcLoaded()` and `CalcLoadedSC()` can be called in either order. The maximum possible CMS uses a different set-up (one molecule, every atom active), so compute it with `calculate_maximum_possible_contact_ms()` or a separate calculator. Calling `CalcLoadedMaxPossibleCMS()` on a calculator that already built a two-molecule surface raises `RuntimeError`, deliberately.

The convenience functions pick the device for you:

```python
calculate_contact_ms(bx, br, tx, tr)                  # device="auto": GPU if available, else CPU
calculate_contact_ms(bx, br, tx, tr, device="cpu")    # always the original NumPy code
calculate_contact_ms(bx, br, tx, tr, device="cuda")   # GPU, error if unavailable
```

`calculate_shape_complementarity` and `calculate_maximum_possible_contact_ms` take the same `device=` argument. CPU and GPU give the same results: identical dot counts, and floating-point values within about 1e-13.

---

## 7. Many structures at once

### Command line

[`examples/calculate_cms_sc.py`](../examples/calculate_cms_sc.py) wraps everything above:

```bash
# designs from a binder campaign: binder = chain B, target = chain A
python examples/calculate_cms_sc.py designs/*.pdb --binder-chains B --target-chains A --csv results.csv

# antibody (heavy + light) against its antigen, from AlphaFold 3 mmCIF files
python examples/calculate_cms_sc.py af3_models/*.cif --binder-chains H,L --target-chains A --csv results.csv

# protein designed to bind a small molecule (ligand as target), with max-CMS normalisation
python examples/calculate_cms_sc.py 1STP.pdb --binder-chains A --binder-exclude BTN --target-resnames BTN --max-cms

# per-residue CMS tables (one CSV per structure and side)
python examples/calculate_cms_sc.py designs/*.pdb --binder-chains B --target-chains A --per-residue per_residue/
```

Output (one line per structure, the same columns in the CSV):

```
structure=1STP.pdb  device=cuda  binder_atoms=901  target_atoms=16  cms=181.3543  cms_binder=201.8963  sc=0.8008  sc_area=363.3935  sc_median_dist=0.3665  max_cms_target=226.8553  max_cms_binder=5635.5122  cms_fraction_target=0.7994  cms_fraction_binder=0.0358  seconds=0.655
```

Other options:
- `--device cpu|cuda|auto`: choose the device.
- `--no-sc`: skip SC.
- `--keep-water`, `--keep-ions`: keep what the loader drops by default.

Atoms with radius 0 are reported as warnings. A structure that fails is reported and the batch continues.

### In Python

```python
import csv, glob
from cms_cuda import MolecularSurfaceCalculator, gpu_available

if gpu_available():
    from cms_cuda import MolecularSurfaceCalculatorGPU as MolecularSurfaceCalculator

rows = []
for path in sorted(glob.glob("designs/*.pdb")):
    bx, br, _ = load_atoms(path, chains="B")
    tx, tr, _ = load_atoms(path, chains="A")
    calc = MolecularSurfaceCalculator()
    calc.add_binder_and_target(bx, br, tx, tr)
    cms, _ = calc.CalcLoaded()
    sc, sc_area, median_dist = calc.CalcLoadedSC()
    rows.append(dict(structure=path, cms=cms, sc=sc, sc_area=sc_area, median_dist=median_dist))

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
```

---

## 8. Which complexes can be analysed?

**Any two groups of atoms can be analysed.** Proteins, peptides (linear or cyclic), antibodies, nucleic acids and small molecules are all fine. The only requirement is that every atom gets a sensible radius.

The radius table has residue-specific entries for amino-acid side-chain atoms and water. Every other atom, including protein backbone atoms, gets a radius from the first letter(s) of its atom name:

| atom name starts with | radius (Å) | covers |
|---|---|---|
| `C` (other than `C`, `CA`, `CB`) | 1.85 | carbons in nucleic acids, ligands, glycans, non-standard residues |
| `N` | 1.65 | nitrogens |
| `O` | 1.60 | oxygens, including phosphate `OP1`/`O1P` |
| `S` | 1.90 | sulfur; also selenium in MSE (`SE`) |
| `P` | 2.15 | phosphorus (backbone of DNA/RNA, phosphorylated residues, ligands) |
| `F` | 1.50 | fluorine |

Name-based radii go wrong for these atoms:

| atom | radius it gets | recommendation |
|---|---|---|
| Br, I, B | 0 (no match: invisible) | if they touch the interface, assign a radius yourself (see below) |
| Cl (`CL`, `CL1`, …) | 1.85 (read as a carbon) | usually acceptable (Cl van der Waals ≈ 1.75 Å) |
| Zn, Mg, K, Mn and most metals | 0 | drop ions (the default in `load_atoms`) |
| Ca²⁺ ion named `CA` | 1.85 (read as a Cα carbon) | drop |
| Na⁺ `NA`, Fe `FE` | 1.65 / 1.50 (read as N / F) | drop |
| hydrogens | 0.5 | remove (the default in `load_atoms`) |

### Results for each type of complex

Crystal structures from the PDB, computed with the defaults. CPU and GPU agree on all of them.

| complex | PDB | binder | target | CMS target (Å²) | CMS binder (Å²) | SC |
|---|---|---|---|---|---|---|
| peptide → protein | 1YCR (p53 / MDM2) | B | A | 485.4 | 460.1 | 0.764 |
| Fv (VH + VL) → protein | 1VFB (D1.3 / lysozyme) | A + B | C | 383.9 | 378.3 | 0.728 |
| VHH → protein | 1MEL (cAb-Lys3 / lysozyme) | A | L | 515.3 | 508.9 | 0.774 |
| protein → DNA | 1AAY (Zif268 zinc fingers) | A | B + C | 675.2 | 679.3 | 0.584 |
| protein → RNA | 1URN (U1A) | A | P | 547.4 | 530.3 | 0.813 |
| protein binder, small-molecule target | 1STP (streptavidin / biotin) | A without BTN | BTN | 181.4 (79.9 % of max) | 201.9 | 0.801 |
| small molecule → protein | 3PTB (benzamidine / trypsin) | BEN | A without BEN | 123.6 | 111.8 (83.5 % of max) | 0.862 |
| small molecule → DNA | 1D30 (DAPI / DNA) | DAP | A + B | 201.4 | 187.2 (69.0 % of max) | 0.823 |

After the default clean-up (no water, ions or hydrogens), none of these structures had an atom with radius 0.

Notes by type:

- **Peptides, cyclic peptides, mini-proteins, de novo binders.** Handled exactly like proteins. Cyclisation, D-amino acids and other non-standard residues are fine: non-standard residues get name-based radii.
- **Antibodies (Fab, scFv, Fv, VHH).** Put heavy and light chains together in the binder. Constant domains far from the epitope do not affect the result.
- **DNA and RNA.** Include both strands of a duplex in the target. Radii are the name-based values above. Phosphorus is only handled correctly from version 0.2.0 on: before the upstream fix of 2026-07-17, P got radius 0, so older nucleic-acid numbers differ.
- **Small molecules.** Atom names must start with the element symbol (`C12`, `N3`, `O2`, …), which is normal in PDB/mmCIF files. SC is well defined for buried ligands, e.g. 1600–2900 trimmed dots per side in the table above. The protein-based SC reference ranges do not apply, so compare ligands with ligands.
- **Glycans, cofactors, waters.** Include or exclude them deliberately, the same way for every structure you compare.

### Assigning a radius to an unmatched atom

Only do this if an atom with radius 0 matters for your interface. Numbers computed this way are no longer directly comparable with the standard radius table, so do it consistently for a whole study.

```python
lig_xyz, lig_radii, lig_info = load_atoms("complex.pdb", resnames={"LIG"})
fallback = {"BR": 1.85, "I": 1.98, "B": 1.92}             # Bondi-type van der Waals radii
for k, (chain, resnum, resname, atom, element) in enumerate(lig_info):
    if lig_radii[k] == 0 and element in fallback:
        lig_radii[k] = fallback[element]
```

---

## 9. Settings

The defaults are the standard values (the same as Rosetta's SC defaults). Keep them if you want numbers that are comparable with other work.

| setting | default | meaning |
|---|---|---|
| `rp` | 1.7 Å | probe radius used to build the molecular surface |
| `density` | 15 dots/Å² | surface dot density |
| `sep` | 8.0 Å | atoms farther than this from the partner are not part of the interface |
| `weight` | 0.5 Å⁻² | the Gaussian `exp(−weight·d²)` in both CMS and SC |
| `band` | 1.5 Å | SC only: width of the interface rim that is trimmed away |
| `binwidth_dist`, `binwidth_norm` | 0.02 | SC only: bin widths of the binned medians |

To change a setting, build the calculator yourself and set it **before** adding atoms (the dot density is stored per atom when atoms are added):

```python
calc = MolecularSurfaceCalculator()
calc.settings.density = 30.0          # finer surface
calc.add_binder_and_target(bx, br, tx, tr)
cms, _ = calc.CalcLoaded()
```

---

## 10. Speed, GPU and large systems

Measured on one machine (NumPy CPU path vs an RTX 5090 GPU, warm):

| structure | CMS | SC | max possible CMS |
|---|---|---|---|
| 120 + 764 atoms | 1.23 s → 17 ms | 1.43 s → 24 ms | 2.41 s → 14 ms |
| 79 + 2304 atoms | 2.45 s → 21 ms | 2.86 s → 27 ms | 8.28 s → 22 ms |

- **GPU:** install CuPy and the GPU is used automatically. Memory use grows linearly with the number of atoms.
- **CPU and large complexes:** the CPU code builds atom × atom distance matrices, so memory grows with the square of the atom count (about 3 GB for each 20,000 × 20,000 float64 matrix). For very large targets, crop to atoms near the interface first. In test on four complexes (a designed peptide binder, 1VFB, 1AAY and 1MEL), keeping only atoms within 10–25 Å of the partner gave the same CMS and SC as the full structure (identical or within 1e-16). Use **15 Å or more** to keep a margin, and check it once on your own system:

```python
from scipy.spatial.distance import cdist
keep_t = cdist(tx, bx).min(axis=1) <= 15.0
keep_b = cdist(bx, tx).min(axis=1) <= 15.0
cms, per_atom, calc = calculate_contact_ms(bx[keep_b], br[keep_b], tx[keep_t], tr[keep_t])
# per_atom now lines up with tx[keep_t]
```

- **Batches on a GPU:** create one calculator per structure. GPU memory is pooled and reused between structures.

---

## 11. Troubleshooting

| symptom | cause and fix |
|---|---|
| `RuntimeError: Coincident atoms` | Two atoms of the same side sit on top of each other: duplicated chains, several models, or alternate locations read twice. Use one model and one altloc per atom (as `load_atoms` does). |
| warnings / atoms with radius 0 | The atom name is not in the radius table (ions, Br, I, …). Drop them or assign radii ([§8](#8-which-complexes-can-be-analysed)). |
| CMS = 0, SC = 0 | The two selections are not in contact (nothing within 8 Å), or a selection is empty / wrong chain. Print `len(bx), len(tx)`. |
| `RuntimeError: ... already generated its molecular surface with all_atoms=...` | `CalcLoadedMaxPossibleCMS()` was called on a calculator used for CMS/SC (or the other way round). Use a new calculator, or `calculate_maximum_possible_contact_ms()`. |
| per-atom array has a different length than expected | It always has one entry per atom you passed, including radius-0 atoms. Check your selection. |
| numbers differ from an old run | Versions before 0.2.0 dropped radius-0 atoms (shifting per-atom arrays) and gave phosphorus radius 0. Hydrogens, waters or a different binder/target split also change results. |
| `device='cuda' requested but no CUDA GPU is usable` | CuPy is not installed, is the wrong CUDA build (`cupy-cuda12x` vs `cupy-cuda13x`), or no GPU is visible (`nvidia-smi`, `CUDA_VISIBLE_DEVICES`). |
| first GPU call takes seconds | One-time CUDA kernel compilation. It is cached afterwards. |

---

**References.** Lawrence, M. C. & Colman, P. M. (1993). Shape complementarity at protein/protein interfaces. *J. Mol. Biol.* 234, 946–950. Contact molecular surface: Longxing Cao's C++ implementation, ported to Python by Brian Coventry ([bcov77/py_contact_ms](https://github.com/bcov77/py_contact_ms)).
