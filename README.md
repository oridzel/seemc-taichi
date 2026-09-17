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
- **v0.5:** coupled planar transport + TEY/SEY/BSEY + emission buffer — complete.
- **v0.6:** reusable planar sweeps and reference-emission export — complete.
- **v0.7:** one-line trapezoidal scan and diagnostic trajectories — complete.
- **v0.8:** SEEMC-imaging-compatible multi-line scan/archive/animation — current.
- **next:** complete production Python-vs-Taichi parity maps and add a
  sampler-compatible accelerated batch API.

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
## v0.6.1 Python-reference emission export

Generate reference `seemc-imaging` emission events in an NPZ layout that
mirrors the Taichi sweep output. This is intended for direct spectrum, angle,
and lineage parity checks.

```bash
seemc-reference-emissions ../MaterialDatabase.pkl \
  --material Si \
  --energies 100 500 1000 5000 \
  --angle 0 \
  --n 5000 \
  --workers 1 \
  --output-dir si_reference_emissions \
  --resume
```

The equivalent example-script invocation is:

```bash
python3 examples/reference_emission_sweep.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --energies 100 500 1000 5000 \
  --angle 0 --n 5000 \
  --output-dir si_reference_emissions
```

Each point writes, for example, `Si_100eV_a0deg_reference_emissions.npz`. The
file contains the same core arrays as the accelerated output
(`emission_energy_ev`, `emission_ux/uy/uz`, `emission_electron_id`,
`emission_parent_id`, `emission_root_primary_id`, `emission_generation`,
`emission_is_cascade`, and numeric `emission_mechanism`) plus reference-only
mechanism labels, birth depth, and barrier-reflection probability.

The exporter derives a deterministic seed independently for every energy/angle
point, writes a summary `reference_yields.csv` after each completed point, and
supports `--resume`. It verifies that the raw exported event counts reproduce
the reference solver's SEY/BSEY before committing each NPZ.
## v0.6.2: rare conditional-q retry

The inelastic kernel now keeps the selected channel fixed and retries the
`omega -> q` conditional draw up to four times when a numerically empty q CDF
is encountered.  `q_cdf_empty` is incremented only if all retries fail.  This
is intended for extremely rare f32/table-interpolation edge cases and does not
change ordinary events.

## v0.8.0: SEEMC-imaging-compatible trapezoidal line scan

v0.8 changes the Taichi workflow to match the established `seemc-imaging`
example scripts. The geometry can contain one or more identical trapezoidal
lines, centered symmetrically at the requested pitch. The line/substrate union
has no buried interface. After an electron crosses an outward surface barrier,
its vacuum ray is tested against every other line and the exposed substrate.
If it hits one, the reciprocal incoming barrier is applied and transport can
continue in that solid. The electron is counted as emitted only after it really
escapes the complete array.

The coordinate convention remains:

- vacuum is negative z;
- line tops are at `z=-height`;
- the substrate begins at `z=0`; and
- every line is infinite along y.

### Run a multi-line scan

This example is the Taichi equivalent of `trapezoidal_line_scan.py` in
`seemc-imaging`:

```bash
seemc-taichi-trapezoid ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energy-ev 1000 \
  --top-width-nm 50 \
  --bottom-width-nm 70 \
  --height-nm 50 \
  --n-lines 3 \
  --pitch-nm 100 \
  --pixels 201 \
  --trajectories 1000 \
  --beam-fwhm-nm 2 \
  --record-primaries-per-pixel 10 \
  --capacity 500000 \
  --trajectory-capacity 200000 \
  --output three_lines.csv \
  --plot
```

When `--field-width-nm` is omitted, the scan automatically covers the entire
array plus 40 nm of substrate on each side. For the geometry above, the line
array spans 270 nm and the default scan width is therefore 350 nm.

The command writes:

- `three_lines.csv` — the wide per-pixel table;
- `three_lines.npz` — the compact raster and per-primary counts;
- `three_lines.trajectories.npz` — one combined, animation-ready archive; and
- `three_lines.png` when `--plot` is present.

Trajectory recording is on by default at every pixel. Only the first ten root
primaries per pixel are retained unless `--record-primaries-per-pixel` is
changed; all 1000 primaries in this example still contribute to every yield.
Use `--record-all-trajectories` to retain every root or
`--no-record-trajectories` to skip the animation archive.

`--capacity` and `--trajectory-capacity` control different storage:

- `--capacity` is the physical particle/cascade pool for one pixel. Overflow
  makes that pixel invalid.
- `--trajectory-capacity` stores raw animation event points for one pixel. A
  trajectory overflow does not alter the yields, but the movie is incomplete.

`--trajectory-stride N` and `--trajectory-max-points N` reduce the saved movie
data while preserving each electron's endpoints. `--beam-fwhm-nm` controls the
Gaussian beam spot; use zero for the historical point beam.

The v0.7 names `--primaries-per-pixel`, `--scan-width-nm`, and
`--output-prefix` remain accepted as aliases. Sparse diagnostic recording via
`--trace-pixel`, `--trace-x-nm`, or `--trace-every` also remains available, but
the normal animation-ready workflow records every pixel.

### Animate the complete scan

The animation command now consumes the combined trajectory archive directly:

```bash
seemc-taichi-animate-trapezoid \
  three_lines.trajectories.npz \
  --fps 30 \
  --frames-per-pixel 8 \
  --pause-frames 1 \
  --color-by energy \
  --vacuum-flight-nm 35 \
  --profile-channels cascade_all,primary_all,tey \
  --output three_lines.gif
```

`seemc-taichi-animate-scan` is an alias for the same whole-scan animator. The
movie follows the SEEMC-imaging presentation: the commanded beam moves across
all recorded pixels, multiple recorded cascades at the current pixel evolve
together with fading tails, and the lower panel builds the three physical
profiles:

- `cascade_all` — **Full SE**;
- `primary_all` — **Full BSE**, including LLE and non-LLE primaries; and
- `tey` — **Total measured** signal.

The substrate and every trapezoidal line use the same fill and have no buried
base boundary. Geometry is read from archive metadata, including all line
centers. `--n-lines`, `--pitch-nm`, and `--line-centers-nm` are display
overrides for old archives.

`--color-by energy` uses a logarithmic instantaneous-energy scale.
`--color-by population` distinguishes SE1, SE2, low-loss primaries, non-LLE
primaries, and absorbed tracks. SE1 follows the current rule: generation 1 and
zero inelastic collisions experienced by that secondary itself; elastic
collisions do not disqualify it.

MP4 output requires ffmpeg. GIF output uses Pillow. The legacy one-pixel
animator remains available as `seemc-taichi-animate-pixel` for old per-pixel
v0.7 trajectory files.
