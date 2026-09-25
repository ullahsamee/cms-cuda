# cms-cuda

GPU-accelerated (CUDA) **contact molecular surface (CMS)** and **Lawrence & Colman shape complementarity (SC)** for protein design.

cms-cuda is built on [bcov77/py_contact_ms](https://github.com/bcov77/py_contact_ms) by Brian Coventry, the Python port of Longxing Cao's C++ contact molecular surface. The original NumPy code is included unchanged and used on the CPU. The GPU version gives the same results (identical dot counts, values within ~1e-13) and is 36–165× faster per complex.

```bash
pip install "cms-cuda[gpu] @ git+https://github.com/ullahsamee/cms-cuda"   # GPU (CUDA 13 driver)
pip install "git+https://github.com/ullahsamee/cms-cuda"                    # CPU only
```

## About contact molecular surface

Longxing Cao's Contact Molecular Surface has been ported to python to allow the next generation of protein designers to use it with ease. Contact Molecular Surface (contact ms) is based on Lawrence and Colman's 1993 paper where they calculate Shape Complementarity. The difference is that instead of returning a singular value denoting the shape complementarity, contact ms instead returns a distance-weighted surface area of the target molecule.

At it's core, contact ms is based on the following formula:
`contact_ms = area * exp( -0.5 * distance**2)`

Where area is the interfacial area on the target and distance is the distance between the binder and the target (from the molecular surfaces) at that point.

> [!NOTE]
> **What is a "dot"?** The docs often talk about surface *dots*. The program covers each molecule's surface with tiny, evenly spaced sample points. The surface here is the skin traced by a 1.7 Å probe ball rolling over the atoms, and it gets about 15 dots per Å², each standing for ~0.07 Å².
>
> Every dot knows its position, its area, its outward direction (normal), which atom it belongs to, and whether it is *buried*, i.e. touching the partner and part of the interface.
>
> The formula above is applied per dot: `area` is the dot's area, and `distance` is the gap to the nearest dot on the binder's surface. Summing over the target's buried dots gives CMS. More in [docs/TUTORIAL.md](docs/TUTORIAL.md#what-is-a-dot).

Here's an image from (Brian Coventry dissertation) explaining why contact ms is better than SASA or Shape Complementarity:
<img width="784" height="442" alt="image" src="https://github.com/user-attachments/assets/a05ec752-9b92-4e95-a6dc-14b60129693a" />

(Look up Brian Coventry Dissertation if you want the full story about it's pros and cons)

**Documentation:** [docs/TUTORIAL.md](docs/TUTORIAL.md) is a step-by-step guide to calculating CMS and shape complementarity. It covers installation, preparing PDB/mmCIF inputs, per-residue CMS, normalisation, batch runs, and which complexes work (peptides, cyclic peptides, mini-proteins, antibodies/VHH, DNA/RNA, small molecules). [examples/calculate_cms_sc.py](examples/calculate_cms_sc.py) does it all from the command line:
```bash
python examples/calculate_cms_sc.py complex.pdb --binder-chains B --target-chains A
```


In terms of using this library, there are really only two functions you need:
```python
from cms_cuda import calculate_contact_ms, get_radii_from_names

# You'll have to figure out how to generate the following arrays
binder_xyz = xyz of binder heavy-atoms (non-hydrogen)
binder_res_names = list of residue name3 for each xyz (so like [ARG, ARG, ARG, LYS])
binder_atom_names = list of atom names for each xyz, stripped (so like [N, CA, C, O])
target_xyz = ...
target_res_names = ...
target_atom_names = ...

# Do not supply your own radii! CMS requires specific radii
binder_radii = get_radii_from_names(binder_res_names, binder_atom_names)
target_radii = get_radii_from_names(target_res_names, target_atom_names)

# Remember, contact_ms is only on the target side by convention
contact_ms, per_target_atom_cms, calc = calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii)

# If you also want the binder-side, you can do this (avoids recomputing everything)
binder_cms, per_binder_atom_cms = calc.calc_contact_molecular_surface(target_side=False)

# If you are doing small-molecule design, you may also want to know the maximum CMS possible (basically the surface area)
from cms_cuda import calculate_maximum_possible_contact_ms
max_target_cms, max_target_cms_per_atom, _ = calculate_maximum_possible_contact_ms(target_xyz, target_radii)
```

If you actually want the original Lawrence & Coleman shape complementarity statistic itself (rather than contact ms's distance-weighted surface area), that's in here too now. Unlike contact_ms, SC is a single whole-interface statistic, so there's no such thing as a sensible per-atom SC value:
```python
from cms_cuda import calculate_shape_complementarity, get_radii_from_names

# Same xyz/radii setup as above
sc, sc_int_area, median_dist, calc = calculate_shape_complementarity(binder_xyz, binder_radii, target_xyz, target_radii)

# sc          -- the shape complementarity statistic (usually 0-1; well-packed interfaces are ~0.5-0.75)
# sc_int_area -- summed trimmed interface area of both molecules (A^2)
# median_dist -- median interface separation distance (A)
```

SC isn't a separate calculator class -- it's the same `MolecularSurfaceCalculator` used for contact ms. So if you'd rather build the calc yourself (same pattern as `calculate_contact_ms` above) you can just call `CalcLoadedSC()` directly instead of going through the convenience function:
```python
from cms_cuda import MolecularSurfaceCalculator

calc = MolecularSurfaceCalculator()
calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)
sc, sc_int_area, median_dist = calc.CalcLoadedSC()
```
CMS and SC are both just different read-only readouts of the same underlying molecular surface, so you can freely call `CalcLoaded()` and `CalcLoadedSC()` (in either order) on the same `calc` instance to get both -- surface generation only actually happens once and is cached/reused, it won't be recomputed or corrupted by calling the other:
```python
calc = MolecularSurfaceCalculator()
calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)

cms, per_target_atom_cms = calc.CalcLoaded()
sc, sc_int_area, median_dist = calc.CalcLoadedSC()
```

## GPU acceleration (NVIDIA / CUDA)

If [CuPy](https://cupy.dev) is installed and a CUDA GPU is visible, the functions above (`calculate_contact_ms`, `calculate_shape_complementarity`, `calculate_maximum_possible_contact_ms`) run on the GPU automatically; otherwise they use the original NumPy code. Nothing else changes: same arguments, same return values (Python `float` and NumPy arrays), same `calc` interface. If you build the calculator yourself, `MolecularSurfaceCalculatorGPU` is the GPU drop-in for `MolecularSurfaceCalculator` (same methods, including `CalcLoaded()` / `CalcLoadedSC()` sharing one generated surface).

```bash
pip install "cms-cuda[gpu] @ git+https://github.com/ullahsamee/cms-cuda"   # or add cupy-cuda13x (cupy-cuda12x for CUDA 12 drivers) to a CPU install
```

```python
from cms_cuda import calculate_contact_ms, gpu_available

gpu_available()                                    # True if CuPy + a CUDA device are usable
calculate_contact_ms(bx, br, tx, tr)               # device="auto": GPU if available, else CPU
calculate_contact_ms(bx, br, tx, tr, device="cpu")   # always the original NumPy code
calculate_contact_ms(bx, br, tx, tr, device="cuda")  # require the GPU (error if unavailable)
```

The GPU path is a method-for-method port of the CPU calculator in float64, so the results are the same: identical dot, probe and atom counts, flags and neighbour lists, and floating-point values that agree with the CPU to about 1e-13 (the only differences come from the last bit of `sin`/`cos`/`exp`/... in CUDA vs NumPy). GPU results are deterministic run to run. `test/test_gpu_parity.py` checks all of this against the CPU code, CMS and SC alike (`--bench` also prints timings; `--golden-dir path/to/test` also checks the upstream `golden.pkl` / `sc_golden.pkl` regression sets).

Typical warm timings (RTX 5090 GPU vs the NumPy CPU path on the same machine):

| structure | `calculate_contact_ms` | `calculate_shape_complementarity` | `calculate_maximum_possible_contact_ms` |
|---|---|---|---|
| 120 + 764 atoms | 1.23 s → 17 ms | 1.43 s → 24 ms | 2.41 s → 14 ms |
| 79 + 2304 atoms | 2.45 s → 21 ms | 2.86 s → 27 ms | 8.28 s → 22 ms |

The first GPU call in a fresh environment compiles the CUDA kernels (several seconds); they are cached in `~/.cupy/kernel_cache`, so later processes start quickly.

For debugging and profiling:

```bash
# kernel bounds assertions + memory/race checking
CMS_CUDA_KERNEL_DEBUG=1 compute-sanitizer --tool memcheck python test/test_gpu_parity.py --quick
# per-stage timeline (every stage is an NVTX range named after the CPU method)
nsys profile -t cuda,nvtx -o cms_profile python test/test_gpu_parity.py --quick
```

