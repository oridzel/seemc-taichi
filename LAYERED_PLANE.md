# Film on substrate (layered plane)

`seemc_taichi/layered_plane.py` adds a planar target made of a thin film on a
semi-infinite substrate, e.g. 70 nm of carbon on tungsten for the RFA grid wires.

## Generate a sampler library

Same command as for a single material, plus the substrate and the thickness:

```bash
python -m seemc_taichi.sweep_cli ../MaterialDatabase.pkl \
  --material C --substrate W --film-thickness-nm 70 \
  --arch cuda --precision f64 \
  --energies <same list as the C library> --angles <same list> \
  --n 100000 --capacity 5000000 \
  --save-emissions --emissions-dir rfa_emissions/C70nm_on_W \
  --output rfa_emissions/C70nm_on_W_yields.csv

python sampler_adapter/taichi_to_sampler_library.py \
  rfa_emissions/C70nm_on_W sampler_library/C70nm_on_W
```

The material label in the CSV/NPZ files is `C70nm_on_W` (override with
`--label`).  The adapter needs no change.  In the RFA model, point the mesh
wires at the new folder:

```python
load_default_surface_models(..., grid_mesh_sampler_dir="sampler_library/C70nm_on_W")
```

## Check it first

```bash
python3 examples/validate_layered_plane.py ../MaterialDatabase.pkl \
  --film C --substrate W --film-thickness-nm 70 --arch cuda --energy 1000 --n 20000
```

It verifies two identities against the validated single-material engine (film
on the same material; very thick film) and prints the C-on-W result.

## Physics

* Each layer uses the `plane.py` collision physics unchanged (the code block is
  copied verbatim and instantiated once per material).
* Interface at depth t: the energy `E_s` changes by `dV` when the electron
  changes layer, parallel momentum is conserved (refraction), and the electron
  is reflected when it cannot supply the step or, stochastically, with the
  quantum reflection probability of the step.
  * `--interface-alignment vacuum` (default): `dV = Ui(substrate) - Ui(film)`,
    i.e. both materials share the vacuum level.
  * `--interface-alignment fermi`: `dV = E_F(substrate) - E_F(film)`, i.e. the
    Fermi levels are aligned, as for two metals in contact.
  * `--interface-model` defaults to the surface barrier model.
* Only the film surface emits; the surface barrier is the film's.
* A leg cut at the interface has no collision; the next leg draws a new free
  path in the new layer (same rule as at the vacuum surface).
* Extra diagnostics in the CSV: `interface_encounters`,
  `interface_transmissions`, `interface_reflections`, `film_to_substrate`,
  `substrate_to_film`, `terminated_below_vacuum_level`.

## Limits

* One film on one substrate; no multilayers, no roughness, no interface mixing.
* A planar film is used for a cylindrical wire: valid while the film is thin
  compared with the wire radius.
* The film thickness is a run-time value: one compiled engine serves any
  thickness (`engine.set_film_thickness_nm`).  The first compilation takes
  several minutes because the collision kernel exists once per material.
