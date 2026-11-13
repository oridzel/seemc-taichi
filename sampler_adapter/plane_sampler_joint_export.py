"""Joint E-theta-phi export helpers for seemc_imaging.plane_samplers.

Use this in the raw-case writer after one SEEMC case has completed.  It keeps
all legacy v1 arrays while adding the full emitted direction and beam-relative
phi for every emitted electron.  One row remains one physical emitted event.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import numpy as np


def _unit(v):
    a=np.asarray(v,dtype=float)
    n=float(np.linalg.norm(a))
    if not np.isfinite(n) or n<=0: raise ValueError('nonzero finite vector required')
    return a/n


def beam_relative_basis(vacuum_incident_direction, surface_normal_out):
    beam_back=-_unit(vacuum_incident_direction)
    n=_unit(surface_normal_out)
    t=n-float(np.dot(n,beam_back))*beam_back
    nt=float(np.linalg.norm(t))
    if nt < 1e-14:
        ref=np.array([1.,0.,0.])
        if abs(float(np.dot(ref,beam_back)))>0.9: ref=np.array([0.,1.,0.])
        t=ref-float(np.dot(ref,beam_back))*beam_back
        t=_unit(t)
    else:
        t=t/nt
    side=_unit(np.cross(beam_back,t))
    return beam_back,t,side


def direction_angles(directions, vacuum_incident_direction, surface_normal_out):
    u=np.asarray(directions,dtype=float)
    if u.ndim != 2 or u.shape[1] != 3: raise ValueError('directions must have shape (N,3)')
    norms=np.linalg.norm(u,axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms<=0): raise ValueError('bad emitted direction')
    u=u/norms[:,None]
    b,t,s=beam_relative_basis(vacuum_incident_direction,surface_normal_out)
    mb=np.clip(u@b,-1.,1.); mt=u@t; ms=u@s
    theta=np.degrees(np.arccos(mb)); phi=np.degrees(np.arctan2(ms,mt))
    outward=u@_unit(surface_normal_out)
    return u,theta,phi,mb,mt,ms,outward


def case_payload_v2(*, emissions, incident_energy_ev, incidence_angle_deg,
                    n_primaries, case_seed, energy_cutoff_ev, config,
                    database_sha256, material, vacuum_incident_direction,
                    surface_normal_out):
    """Return a raw-case NPZ payload with legacy v1 + joint-direction fields."""
    E=np.asarray([float(e.energy) for e in emissions],dtype=float)
    dirs=np.asarray([e.uvw for e in emissions],dtype=float).reshape((-1,3)) if len(emissions) else np.empty((0,3))
    ids=np.asarray([int(e.root_primary_id) for e in emissions],dtype=np.int64)
    if len(E):
        u,theta,phi,mb,mt,ms,outward=direction_angles(dirs,vacuum_incident_direction,surface_normal_out)
        if np.any(outward <= -1e-10):
            bad=int(np.sum(outward<=-1e-10)); raise RuntimeError(f'{bad} exported emissions point into the solid')
    else:
        u=np.empty((0,3)); theta=phi=mb=mt=ms=outward=np.empty(0)

    # Match transport exactly: < cutoff is SE, >= cutoff is BSE.
    se=E < float(energy_cutoff_ev); bse=~se
    def cnt(mask):
        valid=ids[mask]
        valid=valid[(valid>=0)&(valid<int(n_primaries))]
        return np.bincount(valid,minlength=int(n_primaries)).astype(np.int64)

    cfg_dict=asdict(config) if is_dataclass(config) else dict(getattr(config,'__dict__',{}))
    metadata={
        'schema':'seemc-plane-sampler-case-v2',
        'material':str(material),
        'database_sha256':str(database_sha256),
        'config_json':json.dumps(cfg_dict,sort_keys=True,separators=(',',':')),
        'direction_convention':'beam_back=-vacuum_incident_direction; phi=0 toward outward normal in incidence plane',
        'stores_joint_energy_direction':True,
    }
    mechanism=np.asarray([str(getattr(e,'emission_mechanism','transport_escape')) for e in emissions])
    barrier_R=np.asarray([
        np.nan if getattr(e,'barrier_reflection_probability',None) is None else float(e.barrier_reflection_probability)
        for e in emissions
    ],dtype=float)

    payload={
        'metadata_json':np.asarray(json.dumps(metadata,sort_keys=True)),
        'incidence_angle_deg':np.asarray(float(incidence_angle_deg)),
        'incident_energy_ev':np.asarray(float(incident_energy_ev)),
        'n_primaries':np.asarray(int(n_primaries),dtype=np.int64),
        'case_seed':np.asarray(int(case_seed),dtype=np.uint64),
        'energy_cutoff_ev':np.asarray(float(energy_cutoff_ev)),
        # legacy v1 fields
        'se_energy_ev':E[se], 'bse_energy_ev':E[bse],
        'se_theta_deg':theta[se], 'bse_theta_deg':theta[bse],
        'se_primary_id':ids[se], 'bse_primary_id':ids[bse],
        'se_counts_per_primary':cnt(se), 'bse_counts_per_primary':cnt(bse),
        # v2 fields
        'vacuum_incident_direction':_unit(vacuum_incident_direction),
        'surface_normal_out':_unit(surface_normal_out),
        'se_phi_deg':phi[se], 'bse_phi_deg':phi[bse],
        'se_direction_xyz':u[se], 'bse_direction_xyz':u[bse],
        'se_mu_beam_back':mb[se], 'bse_mu_beam_back':mb[bse],
        'se_mu_toward_normal':mt[se], 'bse_mu_toward_normal':mt[bse],
        'se_mu_side':ms[se], 'bse_mu_side':ms[bse],
        'se_emission_mechanism':mechanism[se], 'bse_emission_mechanism':mechanism[bse],
        'se_barrier_reflection_probability':barrier_R[se],
        'bse_barrier_reflection_probability':barrier_R[bse],
    }
    return payload


def save_case_v2(path, **kwargs):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**case_payload_v2(**kwargs)); return path


def joint_sampler_filenames(material):
    """Return material-aware filenames for the two joint sampler files."""
    token=''.join(ch for ch in str(material) if ch.isalnum() or ch in ('_','-'))
    if not token:
        raise ValueError('material must contain at least one filename-safe character')
    return {
        'SE':f'SEJointFromPlaneSampler_uncoated{token}FPA.npz',
        'BSE':f'BSEJointFromPlaneSampler_uncoated{token}FPA.npz',
    }


def write_joint_angle_samplers(case_paths, output_dir, *, material='Cu'):
    """Aggregate v2 per-energy cases into RFA-ready BSE/SE joint NPZ files."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    accum={k:{'Einc':[],'Eout':[],'theta':[],'phi':[],'mb':[],'mt':[],'ms':[]} for k in ('SE','BSE')}
    for p in sorted(map(Path,case_paths)):
        with np.load(p,allow_pickle=False) as z:
            for key in ('se_phi_deg','bse_phi_deg','se_direction_xyz','bse_direction_xyz'):
                if key not in z.files: raise ValueError(f'{p} is not v2; missing {key}')
            Ei=float(z['incident_energy_ev'])
            for kind,prefix in [('SE','se'),('BSE','bse')]:
                n=len(z[f'{prefix}_energy_ev'])
                a=accum[kind]
                a['Einc'].append(np.full(n,Ei)); a['Eout'].append(z[f'{prefix}_energy_ev'])
                a['theta'].append(z[f'{prefix}_theta_deg']); a['phi'].append(z[f'{prefix}_phi_deg'])
                a['mb'].append(z[f'{prefix}_mu_beam_back']); a['mt'].append(z[f'{prefix}_mu_toward_normal']); a['ms'].append(z[f'{prefix}_mu_side'])
    names=joint_sampler_filenames(material)
    result={}
    for kind,a in accum.items():
        cat=lambda k: np.concatenate(a[k]) if a[k] else np.empty(0)
        path=output_dir/names[kind]
        np.savez_compressed(path,
            Einc_eV=cat('Einc'),Eout_eV=cat('Eout'),theta_deg=cat('theta'),phi_deg=cat('phi'),
            dir_beam_back=cat('mb'),dir_toward_normal=cat('mt'),dir_side=cat('ms'),
            schema=np.asarray('seemc-joint-emission-v1'))
        result[kind]=path
    return result
