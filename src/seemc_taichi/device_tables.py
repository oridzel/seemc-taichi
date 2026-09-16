from __future__ import annotations

import numpy as np


class DeviceElasticTables:
    """Elastic-only subset of SEEMC material tables uploaded to Taichi fields."""

    def __init__(self, ti, host, fp):
        host.validate()
        self.fp = fp
        self.nE = int(host.energy_ev.size)
        self.nTheta = int(host.elastic_theta_rad.size)

        self.energy = ti.field(dtype=fp, shape=self.nE)
        self.inv_emfp = ti.field(dtype=fp, shape=self.nE)
        self.elastic_theta = ti.field(dtype=fp, shape=self.nTheta)
        self.elastic_cdf = ti.field(dtype=fp, shape=(self.nTheta, self.nE))

        dtype = np.float32 if fp == ti.f32 else np.float64
        self.energy.from_numpy(np.ascontiguousarray(host.energy_ev, dtype=dtype))
        self.inv_emfp.from_numpy(np.ascontiguousarray(host.inv_emfp, dtype=dtype))
        self.elastic_theta.from_numpy(np.ascontiguousarray(host.elastic_theta_rad, dtype=dtype))
        self.elastic_cdf.from_numpy(np.ascontiguousarray(host.elastic_cdf, dtype=dtype))

        self.e_fermi_ev = float(host.e_fermi_ev)
        self.inner_potential_ev = float(host.inner_potential_ev)


class DeviceTransportTables(DeviceElasticTables):
    """Elastic + electronic-loss lookup tables for v0.3 bulk transport."""

    def __init__(self, ti, host, fp):
        super().__init__(ti, host, fp)
        self.nLossSE = int(host.diimfp_se_eloss_ev.shape[0])
        self.nLossPL = int(host.diimfp_pl_eloss_ev.shape[0])
        self.nOmega = int(host.omega_hartree.size)
        self.nQ = int(host.qlog_a0inv.size)

        self.inv_imfp = ti.field(dtype=fp, shape=self.nE)
        self.inv_imfp_se = ti.field(dtype=fp, shape=self.nE)
        self.inv_imfp_pl = ti.field(dtype=fp, shape=self.nE)

        self.w_se = ti.field(dtype=fp, shape=(self.nLossSE, self.nE))
        self.cdf_se = ti.field(dtype=fp, shape=(self.nLossSE, self.nE))
        self.w_pl = ti.field(dtype=fp, shape=(self.nLossPL, self.nE))
        self.cdf_pl = ti.field(dtype=fp, shape=(self.nLossPL, self.nE))

        self.omega_h = ti.field(dtype=fp, shape=self.nOmega)
        self.qlog = ti.field(dtype=fp, shape=self.nQ)
        self.elf_se = ti.field(dtype=fp, shape=(self.nOmega, self.nQ))
        self.elf_pl = ti.field(dtype=fp, shape=(self.nOmega, self.nQ))

        dtype = np.float32 if fp == ti.f32 else np.float64

        def put(field, a):
            field.from_numpy(np.ascontiguousarray(a, dtype=dtype))

        put(self.inv_imfp, host.inv_imfp)
        put(self.inv_imfp_se, host.inv_imfp_se)
        put(self.inv_imfp_pl, host.inv_imfp_pl)
        put(self.w_se, host.diimfp_se_eloss_ev)
        put(self.cdf_se, host.diimfp_se_cdf)
        put(self.w_pl, host.diimfp_pl_eloss_ev)
        put(self.cdf_pl, host.diimfp_pl_cdf)
        put(self.omega_h, host.omega_hartree)
        put(self.qlog, host.qlog_a0inv)
        put(self.elf_se, host.elf_se)
        put(self.elf_pl, host.elf_pl)

        self.is_metal = bool(host.is_metal)
        self.e_fermi_ev = float(host.e_fermi_ev)
        self.e_fermi_feg_ev = float(host.e_fermi_feg_ev)
        self.e_vb_ev = float(host.e_vb_ev)
        self.band_gap_ev = float(host.band_gap_ev)
        self.e_cbm_ev = float(host.e_cbm_ev)
        self.inner_potential_ev = float(host.inner_potential_ev)
