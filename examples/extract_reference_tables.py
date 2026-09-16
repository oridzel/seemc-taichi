import argparse
from seemc_taichi.reference import require_reference
from seemc_taichi.tables import extract_reference_tables

p = argparse.ArgumentParser()
p.add_argument("database")
p.add_argument("--material", default="Si")
args = p.parse_args()

_, MCConfig, _, Sample = require_reference()
sample = Sample(args.material, db_path=args.database, config=MCConfig())
t = extract_reference_tables(sample)
print("material:", args.material)
print("kind:", "metal" if t.is_metal else "nonconductor")
print("energy bins:", t.energy_ev.size)
print("elastic theta bins:", t.elastic_theta_rad.size)
print("SE loss table:", t.diimfp_se_eloss_ev.shape)
print("plasmon loss table:", t.diimfp_pl_eloss_ev.shape)
print("ELF SE / PL:", t.elf_se.shape, t.elf_pl.shape)
print("omega / log-q bins:", t.omega_hartree.size, t.qlog_a0inv.size)
print("E_F (eV):", t.e_fermi_ev)
print("E_F,FEG (eV):", t.e_fermi_feg_ev)
print("E_vb / gap / CBM (eV):", t.e_vb_ev, t.band_gap_ev, t.e_cbm_ev)
print("U_i (eV):", t.inner_potential_ev)
