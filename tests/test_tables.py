import numpy as np
import pytest
from seemc_taichi.tables import HostMaterialTables


def make_tables(energy=None):
    e = np.array([10.0, 100.0]) if energy is None else np.asarray(energy, float)
    n = e.size
    th = np.array([0.0, 1.0, np.pi])
    cdf = np.column_stack([np.array([0.0, 0.4, 1.0]) for _ in range(n)])
    w = np.column_stack([np.array([1.0, 5.0, 10.0]) for _ in range(n)])
    wcdf = np.column_stack([np.array([0.0, 0.3, 1.0]) for _ in range(n)])
    omega_h = np.array([0.01, 0.1, 1.0])
    qlog = np.array([-4.0, 0.0, 2.0])
    elf = np.ones((omega_h.size, qlog.size))
    return HostMaterialTables(
        energy_ev=e,
        inv_emfp=np.full(n, 0.2),
        inv_imfp=np.full(n, 0.1),
        inv_imfp_se=np.full(n, 0.06),
        inv_imfp_pl=np.full(n, 0.04),
        emfp=np.full(n, 5.0),
        elastic_theta_rad=th, elastic_cdf=cdf,
        diimfp_se_eloss_ev=w, diimfp_se_cdf=wcdf,
        diimfp_pl_eloss_ev=w, diimfp_pl_cdf=wcdf,
        omega_hartree=omega_h, qlog_a0inv=qlog, elf_se=elf, elf_pl=elf,
        is_metal=False, e_fermi_ev=0.0, e_fermi_feg_ev=0.0,
        e_vb_ev=12.0, band_gap_ev=1.1, e_cbm_ev=13.1, inner_potential_ev=17.1,
    )


def test_table_validation_accepts_v03_tables():
    make_tables().validate()


def test_table_validation_rejects_bad_grid():
    with pytest.raises(ValueError, match="strictly increasing"):
        make_tables([100.0, 10.0]).validate()


def test_table_validation_rejects_bad_elf_shape():
    t = make_tables()
    bad = HostMaterialTables(**{**t.__dict__, "elf_se": np.ones((2, 2))})
    with pytest.raises(ValueError, match="elf_se"):
        bad.validate()
