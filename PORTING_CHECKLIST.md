# SEEMC -> Taichi physics port checklist

## Infrastructure
- [x] Separate package; validated `seemc-imaging` remains reference.
- [x] SoA particle state and atomic i32 particle allocation.
- [x] Parent/root-primary/generation bookkeeping.
- [x] CPU f64 / CPU f32 / Metal f32 precision paths.
- [x] Device interpolation, inverse-CDF and 2D ELF primitives.
- [x] Reference material-table extraction.

## v0.2 elastic
- [x] inverse-EMFP interpolation and low-energy model choices.
- [x] stochastic DCS energy-bin selection.
- [x] theta inverse-CDF, azimuth and direction rotation.
- [x] Si 1 keV CPU/f64 one-collision statistical parity.

## v0.3 inelastic + cascade
- [x] total/channel inverse IMFPs and event competition.
- [x] truncated DIIMFP omega sampling.
- [x] relativistic q window + channel ELF q sampling.
- [x] projectile deflection and real secondary construction.
- [x] semiconductor and metal mechanisms.
- [x] secondary atomic enqueue + ancestry.
- [x] Si 1 keV CPU/f32 and Metal/f32 one-inelastic-event parity.
- [ ] repeat dedicated inelastic validator for Cu metal branch.

## v0.4 surface primitives
- [x] outgoing abrupt/classical/expqm transmission.
- [x] internal reflection and refracted outgoing direction.
- [x] incoming reciprocal barrier reflection/refraction.
- [x] CPU/f64 and Metal/f32 standalone barrier validation.

## v0.5 coupled planar transport
- [x] plane intersection truncates the free path before collision.
- [x] no collision after a surface-truncated leg.
- [x] internal reflection draws a fresh free path.
- [x] incoming-barrier reflected primary is an immediate emission.
- [x] event-level emission buffer with energy/direction/ancestry.
- [x] aggregate TEY, conventional SEY/BSEY, cascade/primary yields.
- [x] per-primary emission reconstruction for SEM validation.
- [x] chunk limit is scheduling only, not physical termination.
- [ ] Si normal-incidence TEY/SEY/BSEY parity CPU/f64.
- [ ] Si normal-incidence parity Metal/f32.
- [ ] emission-energy spectrum parity.
- [ ] energy sweep parity.
- [ ] angle sweep parity including grazing incidence.
- [ ] Cu metal-branch planar parity.

## Integration after planar parity
- [ ] accelerated planar sampler-library API.
- [ ] joint E/theta/phi export matching current sampler schema.
- [ ] trapezoid/scene intersection backend.
- [ ] accelerated raster/image batches.
- [ ] production Python-vs-Taichi CPU-vs-GPU benchmark suite.

## v0.6 sweep/validation tooling

- [x] Reuse one Taichi/material/engine initialization across energy/angle sweep points.
- [x] Write one CSV row per energy/angle with yields, diagnostics, and throughput.
- [x] Atomically commit CSV after each completed point.
- [x] Mark overflow/incomplete physics rows invalid and preserve raw diagnostic yields separately.
- [x] Resume valid sweeps and retry invalid points.
- [x] Optional emitted-electron NPZ export for spectra/joint-sampler analysis.
- [ ] Automated Python-reference sweep and comparison plots.
- [ ] Joint sampler export compatible with downstream RFA library layout.

## v0.7 trapezoidal imaging

- [x] analytic top/left-wall/right-wall/substrate intersections in Taichi
- [x] local surface barrier/refraction/reflection on every exposed face
- [x] Gaussian beam-spot sampling at each scan pixel
- [x] persistent engine across the complete line scan
- [x] per-primary yield/SEM accumulation
- [x] TEY and conventional 50 eV SE/BSE channels
- [x] full cascade and full emitted-primary channels
- [x] SE1 = generation 1 + zero own inelastic collisions; SE2 = remaining cascade
- [x] CSV and NPZ line-scan output
- [x] optional quick-look profile plot
- [ ] Taichi JIT compile test on Apple Metal
- [ ] center/top, sidewall, and substrate pixel parity against Python SEEMC
- [ ] full 201-pixel Python-vs-Taichi profile validation
- [ ] trajectory recording/animation backend
- [ ] multi-line array and suspended-line backends
