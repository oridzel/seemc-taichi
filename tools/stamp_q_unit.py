#!/usr/bin/env python3
"""Record the momentum unit in every entry of a SEEMC material database.

  python3 tools/stamp_q_unit.py MaterialDatabase.pkl 'A^-1'            # report only
  python3 tools/stamp_q_unit.py MaterialDatabase.pkl 'A^-1' --write    # add the key

Entries that already carry a 'q_unit' are left alone; one that disagrees with
the unit given here is reported and nothing is written.  A backup
(<name>.bak) is made before writing.
"""
import argparse, pickle, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seemc_taichi.q_units import normalise_q_unit  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("database"); p.add_argument("q_unit"); p.add_argument("--write", action="store_true")
a = p.parse_args()
unit = normalise_q_unit(a.q_unit)
path = Path(a.database)
with open(path, "rb") as f:
    db = pickle.load(f)
entries = db if isinstance(db, list) else ([db] if "name" in db else list(db.values()))
conflict = False; missing = 0
for e in entries:
    have = e.get("q_unit")
    state = "missing" if have is None else normalise_q_unit(have)
    if have is None:
        missing += 1
    elif state != unit:
        conflict = True; state += "   <-- DIFFERENT"
    q = e.get("q")
    rng = "" if q is None else f"   q grid {float(min(q)):.4g} .. {float(max(q)):.4g}"
    print(f"{str(e.get('name')):12s} q_unit: {state}{rng}")
if conflict:
    sys.exit("some entries record a different unit; nothing written")
if not a.write:
    print(f"\n{missing} entries would be stamped with {unit!r}; rerun with --write"); sys.exit(0)
if missing:
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    for e in entries:
        e.setdefault("q_unit", unit)
    with open(path, "wb") as f:
        pickle.dump(db, f)
print(f"stamped {missing} entries with {unit!r}" + (f"; backup {path.name}.bak" if missing else ""))
