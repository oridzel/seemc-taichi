# seemc-taichi

Experimental Taichi acceleration backend for SEEMC electron Monte Carlo transport.

`seemc-taichi` is intentionally separate from `seemc-imaging`: the validated Python implementation remains the physics reference, while the Taichi backend provides accelerated CPU/GPU execution and is checked statistically against the reference before new features are used for production calculations.

Current package version: **v0.7.2**.

## Current capabilities

The accelerated backend currently includes:

- elastic free-flight sampling and ELSEPA angular scattering;
- inelastic SE/plasmon channel selection;
- truncated energy-loss sampling and conditional momentum-transfer sampling;
- dynamic secondary-electron cascade creation;
- semiconductor transport conventions used by the current SEEMC Si model;
- incoming and outgoing surface-barrier physics;
- validated planar TEY/SEY/BSEY simulations;
- energy/angle yield sweeps and emission-event export;
- analytic raised-trapezoid transport for SEM line scans;
- current SE1/SE2 classification;
- selected trajectory recording and animation;
- Apple Metal `f32`, CPU `f32/f64`, and other Taichi backends supported by the local installation.

The Taichi planar implementation has been statistically validated against current Python SEEMC for Si at multiple energies, including 100 eV, 500 eV, 1 keV, and 5 keV.

---

# Installation

From the package directory:

```bash
python3 -m pip install -e '.[reference,test]'
python3 -m pytest -q
```

If console commands such as `pytest` or `seemc-taichi-trapezoid` are installed but not found by `zsh`, either run Python entry points with `python3 -m ...` where applicable or add your Python scripts directory to `PATH`.

For the Python.org macOS install used during development this is typically:

```bash
export PATH="/Library/Frameworks/Python.framework/Versions/3.13/bin:$PATH"
```

Add that line to `~/.zshrc` if you want it to persist.

---

# Coordinate and energy conventions

The trapezoid and planar geometry follow the current SEEMC convention:

```text
vacuum:       z < exposed surface
solid:        z > local surface
substrate:    z >= 0
trapezoid top z = -height
```

For the trapezoid:

```text
             top width
          ┌────────────┐       z = -height
         /              \
        /                \
───────/──────────────────\──── substrate z = 0
       <---- bottom ---->
```

The line is infinite in `y`.

Important transport rules preserved from Python SEEMC:

- incident CLI energy is vacuum energy `E0`;
- inside-solid energy is `E_s`;
- a sampled free path is truncated if an exposed surface is reached first;
- **no collision is applied to a surface-truncated flight**;
- after internal reflection a new free path is sampled;
- parallel momentum is conserved across the surface barrier;
- secondary parent/root/generation lineage is preserved.

---

# Signal definitions

The line-scan output contains both detector-style energy channels and lineage channels. These are deliberately kept separate.

## Conventional energy split

```text
SEY   = emitted electrons with vacuum energy <= BSE cutoff
BSEY  = emitted electrons with vacuum energy >  BSE cutoff
```

The default compatibility cutoff is 50 eV.

## Lineage split

```text
TEY          = every emitted electron
Cascade all  = every emitted cascade-origin electron
Primary all  = every emitted original incident primary
```

These satisfy:

```text
TEY = SEY + BSEY
TEY = Cascade all + Primary all
```

## Current SE1 / SE2 definition

The current imaging definition is:

**SE1**
- generation = 1; and
- that secondary escapes without experiencing any inelastic collision of its own;
- elastic collisions are allowed.

**SE2**
- every other emitted cascade electron.

Therefore:

```text
Cascade all = SE1 + SE2
```

This classification is independent of the 50 eV SE/BSE cutoff.

---

# Planar validation

## Elastic sampling

```bash
python3 examples/validate_elastic_sampling.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --arch cpu \
  --precision f64 \
  --energy 1000 \
  --n 200000
```

## Inelastic sampling

```bash
python3 examples/validate_inelastic_sampling.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energy 1000 \
  --n 20000
```

## Surface barrier

```bash
python3 examples/validate_surface_barrier.py \
  ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energy-solid 100 \
  --energy-vac 1000 \
  --angle 0 \
  --n 200000
```

## Full planar yield parity

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

The validator reports Python-reference and Taichi TEY/SEY/BSEY, Monte Carlo pulls, spectrum-distance diagnostics, allocation size, and transport counters.

Never use yields from a run with particle or emission-buffer overflow.

---

# Batch planar energy/angle sweeps

A complete energy sweep can be run with one Taichi initialization:

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

A full energy-angle grid works the same way:

```bash
seemc-taichi-sweep ../MaterialDatabase.pkl \
  --material Si \
  --arch metal \
  --precision f32 \
  --energies 100 200 500 1000 2000 5000 10000 \
  --angles 0 30 45 60 75 80 85 \
  --n 10000 \
  --capacity 5000000 \
  --output si_yield_map.csv \
  --resume
```

Useful options:

```text
--resume          skip already valid points and retry invalid points
--save-emissions  write emitted-electron NPZ files for spectrum/angle analysis
```

Invalid points retain raw diagnostic values but their main physical yield columns are written as `NaN` so they are not accidentally used as valid results.

---

# Python-reference emission export

To generate emission distributions from the Python reference solver in a format comparable with Taichi:

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

Each NPZ contains emitted energy, direction, electron/parent/root IDs, generation, cascade flag, and mechanism information.

---

# Trapezoidal line scan

The main command is:

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
  --steps-per-chunk 256 \
  --output-prefix si_trapezoid_1keV \
  --plot
```

The finite Gaussian beam is sampled primary-by-primary. Near a geometric edge, primaries belonging to one nominal raster pixel can therefore land on different physical surfaces naturally.

Typical outputs:

```text
si_trapezoid_1keV.csv
si_trapezoid_1keV.npz
si_trapezoid_1keV.png
```

The default quick-look plot shows:

```text
TEY
Cascade all
Primary all
SE1
SE2
```

with Monte Carlo confidence bands.

The CSV/NPZ also contain conventional `sey_50ev` and `bsey_50ev` channels.

---

# `--capacity` vs `--trajectory-capacity`

These control **different memory pools**.

## `--capacity`

`--capacity` is the size of the **actual transport particle pool** for each raster pixel.

It stores:

```text
original primaries
+ generation-1 secondaries
+ generation-2 secondaries
+ generation-3 secondaries
+ ...
```

Example:

```bash
--primaries-per-pixel 1000 \
--capacity 500000
```

means each pixel starts with 1,000 primaries but has room for up to 500,000 total particle records after secondary generation.

If this pool fills:

```text
particle_overflow = True
```

The transport cascade is truncated and **that pixel is physically invalid**.

A too-small `--capacity` affects the simulation result itself.

### Recommended strategy

For a new energy/material/geometry:

1. start with a generous value;
2. inspect `allocated`, `stored`, and `particle_overflow`;
3. reduce later if memory matters.

For the current 1 keV Si trapezoid with 1,000 primaries/pixel:

```bash
--capacity 500000
```

has been a comfortable development value.

Higher incident energies can produce much larger cascades and may require substantially larger pools.

---

## `--trajectory-capacity`

`--trajectory-capacity` is only the size of the **optional trajectory-recording buffer** used for visualization/debugging.

A recorded trajectory point can represent events such as:

```text
launch
elastic collision
inelastic collision
secondary birth
surface hit
internal reflection
escape
termination
```

One electron can therefore contribute many trajectory points.

Example:

```bash
--trace-primaries 20 \
--trajectory-capacity 200000
```

means:

- all simulation primaries are still transported normally;
- only the selected 20 root-primary lineages are recorded;
- up to 200,000 trajectory points may be saved for that traced pixel.

If the trajectory buffer fills, the **physics calculation can still be valid**, but the saved animation is incomplete.

So the distinction is:

| Option | Stores | Too small means |
|---|---|---|
| `--capacity` | actual transported electrons | simulation/pixel invalid |
| `--trajectory-capacity` | visualization points | animation incomplete |

Trajectory tracing should normally use only a small number of primaries because event records are much larger than yield counters.

---

# Recording trajectories

There are two convenient ways to choose traced pixels.

## By pixel index

```bash
--trace-pixel 60
```

The option may be repeated:

```bash
--trace-pixel 60 \
--trace-pixel 100 \
--trace-pixel 140
```

## By physical beam position

Usually easier:

```bash
--trace-x-nm -30 \
--trace-x-nm 0 \
--trace-x-nm 30
```

The nearest raster pixel is selected automatically.

A useful 1 keV diagnostic run is:

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
  --steps-per-chunk 256 \
  --trace-x-nm -30 \
  --trace-x-nm 0 \
  --trace-x-nm 30 \
  --trace-primaries 20 \
  --trajectory-capacity 200000 \
  --output-prefix si_trapezoid_1keV \
  --plot
```

Each traced pixel writes a file such as:

```text
si_trapezoid_1keV_pixel060_trajectories.npz
```

The file contains event positions, energy, electron/parent/root IDs, generation, inelastic count, event code, surface code, and step index.

---

# Trajectory animation

Use:

```text
seemc-taichi-animate-trapezoid
```

The animation can optionally be linked to the full scan NPZ. In that mode the trajectory panel is shown beside the full line profile with a marker at the traced pixel.

## Color by lineage

```bash
seemc-taichi-animate-trapezoid \
  si_trapezoid_1keV_pixel060_trajectories.npz \
  --scan-npz si_trapezoid_1keV.npz \
  --color-by lineage \
  --output left_sidewall_lineage.gif \
  --fps 15
```

Lineage colors distinguish:

```text
Primary
SE1
SE2
```

## Color by generation

```bash
seemc-taichi-animate-trapezoid \
  si_trapezoid_1keV_pixel060_trajectories.npz \
  --scan-npz si_trapezoid_1keV.npz \
  --color-by generation \
  --output left_sidewall_generation.gif
```

## Color by event

```bash
seemc-taichi-animate-trapezoid \
  si_trapezoid_1keV_pixel060_trajectories.npz \
  --scan-npz si_trapezoid_1keV.npz \
  --color-by event \
  --output left_sidewall_events.gif
```

Useful event categories include launch, elastic collision, inelastic collision, spawn, surface hit, reflection, escape, and termination.

## Color by energy

```bash
seemc-taichi-animate-trapezoid \
  si_trapezoid_1keV_pixel060_trajectories.npz \
  --scan-npz si_trapezoid_1keV.npz \
  --color-by energy \
  --output left_sidewall_energy.gif
```

GIF and MP4 outputs are supported when the required Matplotlib writer is available.

### Choosing `--trace-primaries`

Start with:

```bash
--trace-primaries 5
```

to inspect individual cascades clearly.

For a richer visualization:

```bash
--trace-primaries 10
```

to:

```bash
--trace-primaries 20
```

is usually reasonable.

Tracing hundreds or thousands of primaries is not recommended for animation because the plot becomes visually saturated and the trajectory buffer becomes large.

---

# Main trapezoid outputs

## CSV

The CSV contains one row per raster pixel, including:

- beam `x`;
- nominal/local surface information;
- TEY;
- conventional SEY/BSEY;
- Cascade all / Primary all;
- SE1 / SE2;
- generation-resolved channels;
- LLE / non-LLE primary channels;
- Monte Carlo SEMs;
- elapsed time and throughput;
- particle allocation;
- transport diagnostics.

Diagnostic counter names are kept separate from physical yield names, so the scalar `sey_50ev` field is the actual yield rather than the raw integer counter.

## NPZ

The line-scan NPZ stores arrays for all line-profile channels plus per-primary counts. Per-primary counts make covariance-aware signal combinations and metrology analysis possible later.

## Trajectory NPZ

A traced-pixel NPZ stores sparse event points only for selected root-primary lineages. It is intended for visualization and transport diagnostics, not as the main high-statistics simulation output.

---

# Buffer and validity diagnostics

A normal valid pixel should have:

```text
particle_overflow          False
emission_buffer_overflow   0
omega_cdf_empty            0
q_cdf_empty                0
step_limit_hit             0
generation_limit_hit       0
no_scattering_rate         0
```

`step_chunk_continuations` may be nonzero. It is a scheduling counter, not a physics failure.

`q_window_clipped` may also be nonzero for rare kinematic edge cases. The current inelastic implementation retries the conditional `omega -> q` sample when a numerically empty conditional-q distribution is encountered. A surviving `q_cdf_empty` still invalidates the result and should be investigated.

---

# Precision and backend notes

During development the useful validation chain has been:

```text
Python SEEMC reference
        ↓
Taichi CPU f64
        ↓
Taichi CPU f32
        ↓
Taichi Metal f32
```

On the tested Apple Metal backend, `f64` is not supported, so use:

```bash
--arch metal --precision f32
```

CPU `f64` remains useful as the closest accelerated numerical comparison to the Python reference.

---

# Version roadmap

- **v0.1** — structure-of-arrays particle pool and atomic cascade allocation.
- **v0.2** — real elastic transport.
- **v0.3** — real inelastic transport and secondary generation.
- **v0.4** — standalone incoming/outgoing barrier physics.
- **v0.5** — coupled planar TEY/SEY/BSEY solver.
- **v0.6** — energy/angle sweeps and emission export.
- **v0.6.1** — Python-reference emission export.
- **v0.6.2** — robust retry for rare empty conditional-q draws.
- **v0.7.0** — analytic Taichi trapezoidal line scan.
- **v0.7.1** — trajectory recording and corrected line-scan serialization/labels.
- **v0.7.2** — enhanced trajectory animation with lineage/generation/event/energy coloring and linked line-profile panel.

Likely next steps are geometry parity validation against Python SEEMC, multi-line scenes, richer trajectory diagnostics, and eventually integration of the accelerated backend into the main imaging workflow.

---

# Recommended first trapezoid workflow

For a new user or new material:

1. Validate the material with the planar solver.
2. Run a low-statistics trapezoid smoke test.
3. Verify that every raster pixel is valid.
4. Increase primaries/pixel for the production line profile.
5. Trace only a few primaries at representative locations such as top center, sidewall, and substrate.
6. Use linked trajectory animations to understand why the signal changes near edges.
7. Increase `--capacity` if the physical particle pool overflows; increase `--trajectory-capacity` only if the saved animation is truncated.
