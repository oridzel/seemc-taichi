import numpy as np
from seemc_taichi.tables import HostMaterialTables


def test_elastic_table_validation_accepts_normalized_cdf():
    e = np.array([10.0, 100.0])
    th = np.array([0.0, 1.0, np.pi])
    cdf = np.array([[0.0, 0.0], [0.4, 0.2], [1.0, 1.0]])
    w = np.array([[1.0, 1.0], [5.0, 5.0], [10.0, 10.0]])
    wcdf = np.array([[0.0, 0.0], [0.4, 0.3], [1.0, 1.0]])
    omega_h = np.array([0.01, 0.1, 1.0])
    qlog = np.array([-5.0, 0.0, 5.0])
    elf = np.ones((3, 3))
    t = HostMaterialTables(
        energy_ev=e, inv_emfp=np.array([0.2, 0.1]),
        inv_imfp=np.array([0.2, 0.2]),
        inv_imfp_se=np.array([0.1, 0.1]), inv_imfp_pl=np.array([0.1, 0.1]),
        emfp=np.array([5.0, 10.0]), elastic_theta_rad=th, elastic_cdf=cdf,
        diimfp_se_eloss_ev=w, diimfp_se_cdf=wcdf,
        diimfp_pl_eloss_ev=w, diimfp_pl_cdf=wcdf,
        omega_hartree=omega_h, qlog_a0inv=qlog, elf_se=elf, elf_pl=elf,
        is_metal=True, e_fermi_ev=5.0, e_fermi_feg_ev=5.0,
        e_vb_ev=0.0, band_gap_ev=0.0, e_cbm_ev=0.0, inner_potential_ev=10.0,
    )
    t.validate()
