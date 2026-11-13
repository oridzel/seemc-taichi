"""Convert seemc-taichi --save-emissions NPZs to the imaging/RFA export layout.
Requires only numpy. Does not rerun transport or invent missing run provenance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from plane_sampler_joint_export import case_payload_v2, write_joint_angle_samplers
from legacy_tables import angle_directory_name, cosine_probability_grid, export_angle_tables

class ExportCase(SimpleNamespace):
    @property
    def sey(self): return len(self.se_energy_ev) / self.n_primaries
    @property
    def bsey(self): return len(self.bse_energy_ev) / self.n_primaries
    def validate(self):
        if self.n_primaries < 1: raise ValueError('n_primaries must be positive')
        for prefix in ('se', 'bse'):
            energy = getattr(self, prefix+'_energy_ev')
            ids = getattr(self, prefix+'_primary_id')
            counts = getattr(self, prefix+'_counts_per_primary')
            if len(energy) != len(ids) or counts.sum() != len(energy):
                raise ValueError('inconsistent emission counts')

def convert_case(path):
    with np.load(path, allow_pickle=False) as z:
        if not bool(z['valid'].item()): raise ValueError(f'{path}: INVALID transport result')
        material = str(z['material'].item())
        energy, angle = float(z['energy_eV']), float(z['angle_deg'])
        n, cutoff = int(z['n_primaries']), float(z['bse_cutoff_eV'])
        if n < 1 or not math.isfinite(energy) or energy <= 0 or not 0 <= angle < 90 or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError(f'{path}: invalid case metadata')
        e = np.asarray(z['emission_energy_ev'], dtype=float)
        u = np.column_stack([z['emission_'+axis] for axis in ('ux','uy','uz')])
        ids = np.asarray(z['emission_root_primary_id'])
        mech = np.asarray(z['emission_mechanism'])
        if e.ndim != 1 or u.shape != (len(e),3) or ids.shape != e.shape or mech.shape != e.shape:
            raise ValueError(f'{path}: inconsistent array dimensions')
        if np.any(~np.isfinite(e)) or np.any(e < 0) or np.any(e > energy + max(1e-5, energy*1e-5)):
            raise ValueError(f'{path}: invalid emission energy')
        if np.any(~np.isfinite(ids)) or np.any(ids != ids.astype(np.int64)) or np.any(ids < 0) or np.any(ids >= n):
            raise ValueError(f'{path}: invalid primary IDs')
        if not np.all(np.isin(mech, [1,2])): raise ValueError(f'{path}: unknown emission mechanism')
        events = [SimpleNamespace(energy=ee, uvw=uu, root_primary_id=int(ii),
                  emission_mechanism=('incoming_barrier_reflection' if mm==2 else 'transport_escape'))
                  for ee,uu,ii,mm in zip(e,u,ids,mech)]
        a = math.radians(angle)
        payload = case_payload_v2(emissions=events, incident_energy_ev=energy,
            incidence_angle_deg=angle, n_primaries=n, case_seed=0,
            energy_cutoff_ev=cutoff, config=SimpleNamespace(), database_sha256='',
            material=material, vacuum_incident_direction=(math.sin(a),0,math.cos(a)),
            surface_normal_out=(0,0,-1))
        # The sweep NPZ does not record these values. Never claim a seed of 0,
        # a default configuration, or a database identity as actual provenance.
        del payload['case_seed']
        metadata = json.loads(str(payload['metadata_json']))
        metadata.update(schema='seemc-taichi-converted-sampler-case-v1',
            source_file=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_backend='seemc-taichi', database_sha256=None, config_json=None,
            case_seed=None, missing_provenance=['database_sha256','config','case_seed'],
            classification='SE: Eout < cutoff; BSE: Eout >= cutoff',
            barrier_reflection_probability_available=False)
        payload['metadata_json'] = np.asarray(json.dumps(metadata,sort_keys=True))
    case = ExportCase(**{k:(v.item() if v.ndim==0 else v) for k,v in payload.items()})
    case.validate()
    return material, case, payload, metadata

def convert_library(input_dir, output_dir, probability_count=513):
    source, output = Path(input_dir), Path(output_dir)
    files = sorted(source.glob('*_emissions.npz'))
    if not files: raise ValueError(f'No *_emissions.npz files in {source}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; use a new output directory')
    probabilities = cosine_probability_grid(probability_count)
    # Validate all inputs before writing any output; prohibit mixed materials,
    # cutoffs and duplicate points instead of silently combining incompatible cases.
    records = [convert_case(p) for p in files]
    materials = {r[0] for r in records}
    if len(materials)!=1: raise ValueError('Use a separate input directory per material')
    if len({r[1].energy_cutoff_ev for r in records})!=1: raise ValueError('Mixed cutoffs')
    keys = [(r[1].incidence_angle_deg,r[1].incident_energy_ev) for r in records]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate angle/energy cases')
    material = next(iter(materials))
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for angle in sorted({k[0] for k in keys}):
        directory=output/angle_directory_name(angle)
        raw=directory/'raw'; raw.mkdir(parents=True)
        cases=[]; paths=[]
        for mat,case,payload,metadata in sorted(records,key=lambda r:r[1].incident_energy_ev):
            if case.incidence_angle_deg!=angle: continue
            token=f'{case.incident_energy_ev:g}'.replace('.','p')
            p=raw/f'E_{token}eV.npz'
            np.savez_compressed(p,**payload)
            paths.append(p); cases.append(case)
            rows.append(dict(angle_deg=angle,energy_ev=case.incident_energy_ev,
                n_primaries=case.n_primaries,sey=case.sey,bsey=case.bsey,
                tey=case.sey+case.bsey,raw_file=str(p.relative_to(output))))
        export_angle_tables(directory,cases,material=material,probabilities=probabilities)
        write_joint_angle_samplers(paths,directory,material=material)
    with (output/'sampler_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'sampler_generation.json').write_text(json.dumps(dict(
        schema='seemc-taichi-converted-sampler-library-v1',material=material,
        classification='SE < cutoff; BSE >= cutoff',probability_count=probability_count,
        source_cases=[r[3] for r in records],
        note='RFA CSV and joint NPZ formats match imaging exports. Raw checkpoints and manifest are conversion-specific; not imaging v6 resumable checkpoints.'),indent=2)+'\n')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir'); p.add_argument('output_dir')
    p.add_argument('--probability-count',type=int,default=513)
    a=p.parse_args()
    rows=convert_library(a.input_dir,a.output_dir,a.probability_count)
    print(f'Exported {len(rows)} cases to {a.output_dir}')

if __name__=='__main__': main()
