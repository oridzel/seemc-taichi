# seemc-taichi

Experimental Taichi acceleration backend for SEEMC. It remains separate from
`seemc-imaging`: the validated Python transport is the reference implementation,
and each accelerated stage is checked statistically before it is used for
production simulation.

## v0.5.0 scope

v0.5 is the first **complete planar transport slice**. It combines:

- v0.2 elastic free flight and angular scattering;
- v0.3 inelastic `omega/q` sampling and dynamic secondary cascades;
- v0.4 incoming/outgoing surface-barrier physics; and
- a z=0 plane transport loop that returns real TEY/SEY/BSEY and event-level
  emissions.

The plane convention matches current SEEMC: solid `z > 0`, vacuum `z < 0`,
outward normal `(0, 0, -1)`. A sampled free path is truncated when the plane is
reached first. **No collision is applied to a surface-truncated leg.** An
internally reflected electron starts the next leg with a newly sampled free path.

`steps_per_chunk` is only a GPU scheduling bound. Reaching the chunk bound does
not kill an electron; the host relaunches that allocation wave until every member
physically terminates. This replaces the v0.3 benchmark's temporary step-cap
termination.

The emission buffer stores final vacuum energy/direction plus electron, parent,
root-primary and generation IDs. Direct incoming-barrier reflections are included
as immediate primary emissions, matching current Python SEEMC accounting.

## Install

```bash
python3 -m pip install -e '.[reference,test]'
python3 -m pytest -q
```

## 1. Re-run the already validated component checks

Elastic:

```bash
python3 examples/validate_elastic_sampling.py \
  ../MaterialDatabase.pkl --material Si \
  --arch cpu --precision f64 --energy 1000 --n 200000
```

Inelastic:

```bash
python3 examples/validate_inelastic_sampling.py \
  ../MaterialDatabase.pkl --material Si \
  --arch metal --precision f32 --energy 1000 --n 20000
```

Barrier:

```bash
python3 examples/validate_surface_barrier.py \
  ../MaterialDatabase.pkl --material Si \
  --arch metal --precision f32 \
  --energy-solid 100 --energy-vac 1000 --angle 0 --n 200000
```

## 2. First full planar parity test

Start with CPU/f64 and modest statistics:

```bash
python3 examples/validate_plane_yield.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --arch cpu \
  --precision f64 \
  --energy 1000 \
  --angle 0 \
  --n 2000 \
  --capacity 200000
```

Then Metal/f32:

```bash
python3 examples/validate_plane_yield.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energy 1000 \
  --angle 0 \
  --n 2000 \
  --capacity 200000
```

The validator reports Python-reference and Taichi TEY/SEY/BSEY, combined
Monte-Carlo pulls, emission-energy histogram TV distance, emitted-direction norm
error, allocation size and all transport diagnostics. Increase `--capacity` if
there is any particle or emission overflow; an overflowed run is not a yield test.

## 3. Full planar Metal benchmark

```bash
python3 examples/plane_yield_benchmark.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energy 1000 \
  --angle 0 \
  --n 10000 \
  --capacity 500000 \
  --steps-per-chunk 256
```

This reports actual planar yields as well as warmed transport throughput. The
benchmark calls `ti.sync()` before stopping the timer.

## Physics conventions preserved

- Inside-solid energy is `E_s`, referenced to the valence-band bottom.
- Incident CLI energy is vacuum energy `E0`; transmitted primaries enter with
  `E_s = E0 + U_i` and the reciprocal barrier refraction.
- Elastic and inelastic tables use the same configurable energy references as
  current Python SEEMC.
- Semiconductor inelastic thresholds, `omega` window and q kinematics match the
  current model.
- Channel DIIMFP CDFs are truncated before loss sampling.
- Secondary children preserve immediate parent, root primary and generation.
- With `track_subbarrier=False`, inside-solid electrons at/below `U_i` are
  terminal and newly born secondaries at/below `U_i` are not queued.
- On an outgoing surface encounter, escape uses the configured
  `abrupt`/`classical`/`expqm` barrier and preserves parallel momentum.
- Conventional SE/BSE counters use the configured 50 eV compatibility cutoff;
  ancestry-based cascade/primary yields are reported separately.

## Roadmap

- **v0.1:** SoA particle pool and atomic cascade allocation — complete.
- **v0.2:** real elastic transport — complete/validated.
- **v0.3:** real inelastic transport and secondary creation — complete/validated
  on Si CPU/Metal single-event tests.
- **v0.4:** standalone incoming/outgoing surface barrier — complete/validated.
- **v0.5:** coupled planar transport + TEY/SEY/BSEY + emission buffer — current.
- **next:** establish planar yield/spectrum parity over energy and angle, then add
  a sampler-compatible accelerated batch API.
- **later:** trapezoid/scene geometry and imaging batches.

## v0.6 batch planar yield sweeps

Run a complete normal-incidence energy sweep with one Taichi initialization:

```bash
seemc-taichi-sweep ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energies 100 200 500 1000 2000 5000 10000 \
  --angles 0 \
  --n 10000 \
  --capacity 5000000 \
  --output si_yield_taichi.csv
```

The same runner accepts an energy-angle grid, for example:

```bash
seemc-taichi-sweep ../MaterialDatabase.pkl \
  --material Si --arch metal --precision f32 \
  --energies 100 200 500 1000 2000 5000 10000 \
  --angles 0 30 45 60 75 80 85 \
  --n 10000 --capacity 5000000 \
  --output si_yield_map.csv --resume
```

`--resume` skips valid points already in the CSV and retries points previously marked invalid.  The CSV is rewritten atomically after every completed point, so an interrupted sweep keeps its prior results.  Overflowed or otherwise incomplete runs are retained through `raw_*` columns but the primary `tey`, `sey`, `bsey`, `cascade_yield`, and `primary_yield` columns are written as NaN and `valid=False`.

Use `--save-emissions` to write one compressed NPZ per energy/angle containing emitted energy, direction, lineage, generation, and mechanism arrays.  The example script `examples/plane_yield_sweep.py` exposes the same interface if you prefer to invoke it with Python directly.
\n\n## v0.6.1 Python-reference emission export\n\nGenerate reference `seemc-imaging` emission events in an NPZ layout that mirrors\nthe Taichi sweep output.  This is intended for direct spectrum, angle and lineage\nparity checks.\n\n```bash\nseemc-reference-emissions ../MaterialDatabase.pkl \\\n  --material Si \\\n  --energies 100 500 1000 5000 \\\n  --angle 0 \\\n  --n 5000 \\\n  --workers 1 \\\n  --output-dir si_reference_emissions \\\n  --resume\n```\n\nThe equivalent example-script invocation is:\n\n```bash\npython3 examples/reference_emission_sweep.py \\\n  ../MaterialDatabase.pkl \\\n  --material Si \\\n  --energies 100 500 1000 5000 \\\n  --angle 0 --n 5000 \\\n  --output-dir si_reference_emissions\n```\n\nEach point writes, for example,\n`Si_100eV_a0deg_reference_emissions.npz`.  The file contains the same core\narrays as the accelerated output (`emission_energy_ev`, `emission_ux/uy/uz`,\n`emission_electron_id`, `emission_parent_id`, `emission_root_primary_id`,\n`emission_generation`, `emission_is_cascade`, and numeric\n`emission_mechanism`) plus reference-only mechanism labels, birth depth and\nbarrier-reflection probability.\n\nThe exporter derives a deterministic seed independently for every energy/angle\npoint, writes a summary `reference_yields.csv` after each completed point, and\nsupports `--resume`.  It verifies that the raw exported event counts reproduce\nthe reference solver's SEY/BSEY before committing each NPZ.\n
## v0.6.2: rare conditional-q retry

The inelastic kernel now keeps the selected channel fixed and retries the
`omega -> q` conditional draw up to four times when a numerically empty q CDF
is encountered.  `q_cdf_empty` is incremented only if all retries fail.  This
is intended for extremely rare f32/table-interpolation edge cases and does not
change ordinary events.

## v0.7.0: Taichi trapezoidal line scan

v0.7.0 extends the validated planar Taichi transport to the analytic
`TrapezoidalLine` geometry used by SEEMC imaging.  The line is infinite in y,
vacuum is toward negative z, the top is at `z=-height`, and the bulk substrate
starts at `z=0`.  Free flights are truncated at the first exposed top,
sidewall, or substrate surface before barrier physics is applied.

The line-scan runner keeps one material/table set and one Taichi engine alive
for the whole scan.  A finite Gaussian beam is sampled separately at each
pixel, so primaries close to an edge may land on different faces naturally.
The output includes yield and Monte-Carlo SEM for TEY, conventional SE/BSE,
all cascade emissions, all emitted original primaries, and the current SE1/SE2
classification:

- **SE1:** generation 1 and no inelastic collision experienced by that emitted
  secondary itself (elastic collisions are allowed).
- **SE2:** every other emitted cascade electron.

A typical Apple-Metal run is:

```bash
seemc-taichi-trapezoid ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energy-ev 1000 \
  --top-width-nm 50 \
  --bottom-width-nm 70 \
  --height-nm 50 \
  --scan-width-nm 150 \
  --pixels 201 \
  --primaries-per-pixel 1000 \
  --beam-fwhm-nm 2 \
  --capacity 500000 \
  --output-prefix si_trapezoid_1keV \
  --plot
```

This writes `si_trapezoid_1keV.csv`, `si_trapezoid_1keV.npz`, and (with
`--plot`) `si_trapezoid_1keV.png`.  Any pixel that hits particle/emission
capacity or a fatal transport diagnostic is marked invalid and its primary
yield columns are written as NaN in the CSV; the raw diagnostics remain
available for debugging.
\n\n## v0.7.2 trajectory animation\n\nTrapezoid scans can record a small, selected set of root primaries and all of\ntheir descendants.  This is intended for visualization/debugging; do not trace\nall high-statistics primaries because trajectory records are much larger than\nyield counters.\n\nExample scan with one traced sidewall pixel:\n\n```bash\nseemc-taichi-trapezoid ../MaterialDatabase.pkl \\\n  --material Si --arch metal --precision f32 \\\n  --energy-ev 1000 --top-width-nm 50 --bottom-width-nm 70 --height-nm 50 \\\n  --scan-width-nm 150 --pixels 201 --primaries-per-pixel 1000 \\\n  --beam-fwhm-nm 2 --capacity 500000 \\\n  --trace-pixel 60 --trace-primaries 20 --trajectory-capacity 200000 \\\n  --output-prefix si_trapezoid_1keV --plot\n```\n\nThe traced-pixel file can then be animated and linked to the full line scan:\n\n```bash\nseemc-taichi-animate-trapezoid \\\n  si_trapezoid_1keV_pixel060_trajectories.npz \\\n  --scan-npz si_trapezoid_1keV.npz \\\n  --color-by lineage \\\n  --output si_trapezoid_pixel060_lineage.gif\n```\n\n`--trace-x-nm` is usually easier than calculating a pixel index; it selects the nearest scan pixel.  `--color-by` accepts `lineage`, `generation`, `event`, or `energy`.  Lineage\nuses the current SEEMC imaging rule: Primary; SE1 = generation 1 and zero own\ninelastic collisions; SE2 = all other cascade electrons.  With `--scan-npz`,\nthe right panel shows TEY, Cascade all, Primary all, SE1 and SE2 and marks the\nanimated pixel.  Geometry is read from trajectory metadata, so width/height\narguments are normally unnecessary for files made by v0.7.1 or later.\n