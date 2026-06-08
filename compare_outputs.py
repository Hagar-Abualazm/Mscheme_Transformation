"""Numerically compare Mscheme.csv / Tsingle.csv against the golden reference.

After the Step 2 CG swap, outputs are no longer bit-identical (different float code path),
so we compare numeric columns with a relative tolerance of 1e-10 and require the
non-numeric (quartet) columns to match exactly. Exits non-zero on any mismatch.
"""

import csv
import sys

RTOL = 1e-10
ATOL = 1e-12


def load(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def close(a, b):
    try:
        x = float(a); y = float(b)
    except ValueError:
        return a == b  # non-numeric cell: require exact string match
    return abs(x - y) <= ATOL + RTOL * abs(y)


def compare(new_path, ref_path, numeric_from_col):
    new = load(new_path); ref = load(ref_path)
    assert len(new) == len(ref), f"{new_path}: row count {len(new)} != {len(ref)}"
    max_rel = 0.0
    for ri, (rn, rr) in enumerate(zip(new, ref)):
        assert len(rn) == len(rr), f"{new_path} row {ri}: column count differs"
        for ci, (cn, cr) in enumerate(zip(rn, rr)):
            if ci >= numeric_from_col:
                try:
                    y = float(cr)
                    if y != 0.0:
                        max_rel = max(max_rel, abs(float(cn) - y) / abs(y))
                except ValueError:
                    pass
            if not close(cn, cr):
                print(f"MISMATCH {new_path} row {ri} col {ci}: {cn!r} vs ref {cr!r}")
                return False, max_rel
    return True, max_rel


ok1, rel1 = compare("Mscheme.csv", "refs/reference_Mscheme.csv", numeric_from_col=2)
ok2, rel2 = compare("Tsingle.csv", "refs/reference_Tsingle.csv", numeric_from_col=2)

print(f"Mscheme.csv: {'OK' if ok1 else 'FAIL'}  max relative diff = {rel1:.3e}")
print(f"Tsingle.csv: {'OK' if ok2 else 'FAIL'}  max relative diff = {rel2:.3e}")

if ok1 and ok2:
    print(f"PASS: all values match reference within rtol={RTOL:.0e}")
    sys.exit(0)
sys.exit(1)
