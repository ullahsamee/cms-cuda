Longxing Cao's Contact Molecular Surface has been ported to python to allow the next generation of protein designers to use it with ease. Contact Molecular Surface (contact ms) is based on Lawrence and Colman's 1993 paper where they calculate Shape Complementarity. The difference is that instead of returning a singular value denoting the shape complementarity, contact ms instead returns a distance-weighted surface area of the target molecule.

At it's core, contact ms is based on the following formula:
`contact_ms = area * exp( -0.5 * distance**2)`

Where area is the interfacial area on the target and distance is the distance between the binder and the target (from the molecular surfaces) at that point.

Perhaps the coolest new thing the python version brings is per-atom contact ms! I always thought this was a dubious idea but it actually turns out its very straightforward based on the math.


Here's an image from my PhD explaining why contact ms is better than SASA or Shape Complementarity:
<img width="784" height="442" alt="image" src="https://github.com/user-attachments/assets/a05ec752-9b92-4e95-a6dc-14b60129693a" />

(Look up Brian Coventry Dissertation if you want the full story about it's pros and cons)


In terms of using this library, there are really only two functions you need:
```python
from py_contact_ms import calculate_contact_ms, get_radii_from_names

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
from py_contact_ms import calculate_maximum_possible_contact_ms
max_target_cms, max_target_cms_per_atom = calculate_maximum_possible_contact_ms(target_xyz, target_radii)
```

If you actually want the original Lawrence & Coleman shape complementarity statistic itself (rather than contact ms's distance-weighted surface area), that's in here too now. Unlike contact_ms, SC is a single whole-interface statistic, so there's no such thing as a sensible per-atom SC value:
```python
from py_contact_ms import calculate_shape_complementarity, get_radii_from_names

# Same xyz/radii setup as above
sc, sc_int_area, median_dist, calc = calculate_shape_complementarity(binder_xyz, binder_radii, target_xyz, target_radii)

# sc          -- the shape complementarity statistic (usually 0-1; well-packed interfaces are ~0.5-0.75)
# sc_int_area -- summed trimmed interface area of both molecules (A^2)
# median_dist -- median interface separation distance (A)
```

SC isn't a separate calculator class -- it's the same `MolecularSurfaceCalculator` used for contact ms. So if you'd rather build the calc yourself (same pattern as `calculate_contact_ms` above) you can just call `CalcLoadedSC()` directly instead of going through the convenience function:
```python
from py_contact_ms import MolecularSurfaceCalculator

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


Many thanks the Claude and ChatGPT for making this herculean effort possible (in like 20 hours which is insane)
