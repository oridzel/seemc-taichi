import tempfile
from pathlib import Path
import numpy as np
from taichi_to_sampler_library import convert_library

def fixture(path,energy=100,angle=75,energies=(10.,50.,80.),valid=True,ids=None):
    e=np.array(energies)
    np.savez(path,material='Cu',energy_eV=energy,angle_deg=angle,n_primaries=4,
        bse_cutoff_eV=50.,valid=valid,emission_energy_ev=e,
        emission_ux=np.zeros(len(e)),emission_uy=np.zeros(len(e)),emission_uz=-np.ones(len(e)),
        emission_root_primary_id=np.arange(len(e)) if ids is None else np.array(ids),
        emission_mechanism=np.ones(len(e),dtype=int))

with tempfile.TemporaryDirectory() as t:
    root=Path(t); src=root/'in';src.mkdir()
    fixture(src/'Cu_100_emissions.npz')
    fixture(src/'Cu_200_emissions.npz',energy=200,energies=())
    rows=convert_library(src,root/'out')
    assert rows[0]['sey']==.25 and rows[0]['bsey']==.5
    out=root/'out'/'alpha_75deg'
    assert len(list(out.glob('*.csv')))==6
    with np.load(out/'BSEJointFromPlaneSampler_uncoatedCuFPA.npz') as z:
        np.testing.assert_array_equal(z['Eout_eV'],[50,80])
        np.testing.assert_allclose(z['theta_deg'],75)
        np.testing.assert_allclose(z['phi_deg'],0,atol=1e-12)
        assert str(z['schema'])=='seemc-joint-emission-v1'
    with np.load(out/'raw'/'E_100eV.npz') as z:
        assert 'case_seed' not in z
        np.testing.assert_array_equal(z['bse_counts_per_primary'],[0,1,1,0])
    try: convert_library(src,root/'out')
    except FileExistsError: pass
    else: raise AssertionError('overwrite accepted')
    fixture(src/'Cu_100_emissions.npz',valid=False)
    try: convert_library(src,root/'invalid')
    except ValueError: pass
    else: raise AssertionError('invalid accepted')
    assert not (root/'invalid').exists()
    fixture(src/'Cu_100_emissions.npz',ids=[0,-1,2])
    try: convert_library(src,root/'invalid_ids')
    except ValueError: pass
    else: raise AssertionError('invalid IDs accepted')
print('PASS: exact cutoff, joint pairing/basis, six CSVs, empty class, counts, overwrite and invalid-input rejection')
