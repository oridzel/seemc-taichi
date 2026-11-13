# Taichi emissions to the RFA sampler library

This adapter reuses the supplied plane_sampler_joint_export(5).py unchanged,
and copies the CSV writer functions unchanged from the supplied seemc-imaging
plane_samplers.py. Only numpy is needed for conversion.

Unzip this package in your working directory. Keep its three Python modules together.

1. Generate Taichi emissions (from your working seemc-taichi environment):

```bash
python -m seemc_taichi.sweep_cli ../MaterialDatabase.pkl \
  --material Cu --arch cuda --precision f64 \
  --energies 500 --angles 0 75 --n 10000 --capacity 1000000 \
  --save-emissions --emissions-dir rfa_emissions_test/Cu \
  --output rfa_emissions_test/Cu_yields.csv
```

On Metal use --arch metal --precision f32. Expand the grids after validating the
small run. For a full RFA cascade library include the required low incident
energies and verify the database work function and barrier settings first.

2. Convert the saved emissions:

```bash
python sampler_adapter/taichi_to_sampler_library.py \
  rfa_emissions_test/Cu sampler_library/Cu
```

A Python/notebook entry point is also available:

```python
import sys
sys.path.insert(0, 'sampler_adapter')
from taichi_to_sampler_library import convert_library
rows = convert_library('rfa_emissions_test/Cu', 'sampler_library/Cu')
```

The destination must be new or empty. No existing library is overwritten.
One input folder must contain one material and one cutoff, with at most one file
per angle/energy. Invalid results are rejected before output is written.
All input cases are loaded into host memory; for large libraries process one
angle at a time into separate temporary roots and retain each alpha directory.
The root manifests describe only the cases in that particular conversion.

Each alpha_0deg, alpha_75deg, etc. directory contains:

- SEYFromPlane_SEVaccum_t0nmCuFPA.csv
- BSEYFromPlane_SEVaccum_t0nmCuFPA.csv
- SEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv
- BSEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv
- SEThetaFromPlaneSampler_uncoatedCuFPA.csv
- BSEThetaFromPlaneSampler_uncoatedCuFPA.csv
- SEJointFromPlaneSampler_uncoatedCuFPA.npz
- BSEJointFromPlaneSampler_uncoatedCuFPA.npz
- readme.txt
- raw/E_500eV.npz (and other incident energies)

The six legacy CSV names retain Cu even for other materials, exactly as in the
supplied imaging writer. The two joint NPZ filenames use the actual material.
Use separate material roots. Point the RFA configuration to the new root as
appropriate; this adapter does not edit the RFA configuration.

The joint files preserve each paired emitted energy and beam-relative direction.
They use schema seemc-joint-emission-v1, with Einc_eV, Eout_eV, theta_deg, phi_deg,
dir_beam_back, dir_toward_normal, and dir_side.

Classification follows the supplied imaging exporter: SE Eout < cutoff;
BSE Eout >= cutoff. In the supplied Taichi version, aggregate counters use
Eout <= cutoff for SE. Consequently yields may differ at the exact boundary;
this converter always recomputes from the saved event energies.

The root sampler_manifest.csv and sampler_generation.json are conversion-specific.
Raw files have the v2 fields needed by the joint exporter but are explicitly
labelled seemc-taichi-converted-sampler-case-v1, not imaging's v6 checkpoint schema.
They cannot resume imaging simulations and do not claim its history-based
population classification. Missing database hash/config/seed are recorded as
unknown. Per-electron barrier reflection probabilities are NaN (not exported by
Taichi); mechanism labels remain available. Keep the original database and run
command/settings with the results for production provenance.

Tests: python sampler_adapter/test_adapter.py
Synthetic tests cover exact cutoff, zero-emission cases, paired joint arrays,
beam-relative angles, per-primary counts, CSV creation and invalid-input rejection.
Actual GPU transport and loading into the RFA application were not run here.
