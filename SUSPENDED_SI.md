# Suspended Si linescan — v0.9.2

This update adds `--suspended` to the v0.9.1 linescan/raster driver. With this
flag, each trapezoid is a finite solid with vacuum below it and in the gaps.
Without the flag, the original bulk-substrate geometry remains the default.

## Geometry from the supplied document

- Si, nominal width 50 nm, thickness 30 nm.
- Initial approximation: top = bottom = 50 nm (vertical walls).
- The document says the taper is less than 2 degrees from vertical. For an
  outward taper of exactly 2 degrees and a 50 nm top, bottom width is
  `50 + 2*30*tan(2 degrees) = 52.095 nm`. This is a sensitivity limit, not a
  measured bottom width. Use `--bottom-width-nm 52.095` for that comparison.
- Width is provisionally interpreted as top width; verify against independent
  dimensional measurements before using the scan for absolute metrology.
- Commands below use one isolated line, infinite along y. The document does
  not supply the line-and-space pitch, so these commands do not reproduce the
  complete patterned array. When known, add `--n-lines 3 --pitch-nm VALUE` and
  a field wide enough to span the array. Electrons may re-enter adjacent lines.
- No supporting film remains between lines in `--suspended` mode. If the real
  structure has a continuous residual film, that is a different geometry.

## Install

Unzip the package, then run from the extracted `seemc-taichi-v0.9.2` directory
in the same Python environment used for v0.9.1:

```bash
python -m pip install -e .
python -c "import seemc_taichi; print(seemc_taichi.__version__)"
```

The version should print `0.9.2`. Keep your existing `seemc-imaging` installation
and material database. The database must contain Si transport tables covering
the energies being simulated, including 30 keV. This package contains code, not
the user's Si material database.

## Run both energies (Bash / zsh)

Replace `/path/to/MaterialDatabase.pkl` with the actual database path. `metal`
is for a compatible Mac GPU; use `cuda` for an NVIDIA GPU or `cpu` for CPU.
The existing precision defaults are f32 on Metal and f64 otherwise.

```bash
for energy in 900 30000; do
  seemc-taichi-trapezoid /path/to/MaterialDatabase.pkl \
    --material Si --arch metal \
    --energy-ev "$energy" \
    --suspended \
    --top-width-nm 50 --bottom-width-nm 50 --height-nm 30 \
    --n-lines 1 \
    --field-width-nm 120 --nx 1201 --ny 1 \
    --beam-fwhm-nm 0.1 \
    --primaries-per-pixel 10000 \
    --no-record-trajectories --no-per-primary-counts \
    --plot --plot-channels back_sey_50ev,forward_sey_50ev \
    --output "si_suspended_50w_30h_0p1nm_${energy}eV.csv"
done
```

This is 12.01 million incident primaries per energy. For an initial short check,
use `--nx 121 --primaries-per-pixel 1000` and different output names. The final
reference scan uses 0.1 nm spacing; the short check has 1 nm spacing and is not
an adequate reference for fitting narrow edge features.

Each run writes CSV, NPZ and PNG files. Counts, yields and Monte Carlo standard
errors are retained with `--no-per-primary-counts`; only the large individual
primary count arrays are omitted. No trajectories are recorded with these
commands. Keep the `valid` and `status` columns: invalid pixels must not enter
a profile fit. For particle overflow, increase `--capacity` or reduce primaries
per pixel; do not interpret a truncated cascade as a valid yield.

## Signals and interpretation

Coordinates: top at z = -30 nm, bottom at z = 0; incident beam travels in +z.

| Output channel | Meaning |
| --- | --- |
| `back_sey_50ev` | Electrons of vacuum energy <=50 eV escaping with final uz<0 |
| `forward_sey_50ev` | Same energy range, final uz>=0 |
| `back_bsey_50ev` | Energy >50 eV, final uz<0 |
| `forward_primary_all` | Original primaries escaping toward +z after encountering the sample |
| `back_se1`, `back_se2` | Upward SE1/SE2 populations, without a 50 eV cut |
| `gap_primary_fraction` | Fraction of incident primaries that miss all solid |
| `sey_50ev` | SE energy window summed over all final directions |

CSV also contains `yield__CHANNEL` and `sem__CHANNEL`; NPZ stores the same
channels. SE1 retains the current definition: generation 1 with zero own
inelastic collisions. For suspended samples the historical `bsey_50ev` is an
all-direction energy-window count and includes transmitted electrons; it must
not be interpreted as backward BSE. Likewise, `tey` is an all-direction emitted
count, not a modeled detector current. Direct gap primaries are excluded from
all emission channels; add their fraction to `forward_primary_all` if an ideal
forward primary detector should also include the unobstructed beam.

Use `back_sey_50ev` as an initial comparison channel for the SEM measurement,
while retaining the forward signal for diagnosis. A hemisphere cut is not an
in-lens detector model: electron-optical fields and detector acceptance are
not included. The finite slab also does not add crystal-orientation-dependent
coherent transport, charging, oxide, roughness, or beam convergence. The beam
has a Gaussian lateral distribution and parallel normal-incidence directions.

## Subsequent convolution

For Gaussian beams, convolve the 0.1 nm-FWHM reference with a Gaussian having
FWHM `sqrt(target_fwhm_nm**2 - 0.1**2)`. Its standard deviation in scan pixels
is that value divided by `2.354820045 * 0.1` for the final reference grid.
Use additional scan margin if the fitted beam has wide tails. This operation
changes the spatial beam distribution; it does not introduce angular
convergence or change the material geometry.

## Validation scope

The included tests use synthetic transport tables to isolate geometry and
surface handling. They check bottom transmission at 900 eV and 30 keV, vacuum
gap misses, Gaussian launches straddling an edge, and host-side re-entry
intersections. Transport tests run on CPU in f64 and f32; Metal/CUDA hardware is not
exercised here. They do not validate the physical Si yields. Production Si
profiles still require your material database and local GPU run.
