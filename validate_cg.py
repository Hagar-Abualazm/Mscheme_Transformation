"""Standalone validation of the pure-stdlib numeric Clebsch-Gordan against sympy.

Reproduces exactly the CG argument tuples that Mscheme.py's decoupling loop generates
(angular-momentum and isospin couplings, for the target J=1, T=0 channel), then asserts
the math.lgamma implementation in cg_numeric.py agrees with
sympy.physics.wigner.clebsch_gordan to within 1e-12 (absolute) on every encountered tuple.

sympy is imported HERE ONLY (the production path in Mscheme.py uses the numeric version).

Run:  python3 validate_cg.py
"""

import csv
from sympy.physics.wigner import clebsch_gordan as cg_sympy
from cg_numeric import clebsch_gordan as cg_num

state_map_path = "state_map_2B.csv"
MFDN_path = "Helium_2_data_N3LO_EM500_srg1.0_hw16_emax4_e2max8.MFDn"
J_target = 1
T_target = 0

COL_A, COL_B, COL_C, COL_D = 0, 1, 2, 3
COL_J, COL_T = 4, 5

# --- read MFDn, index by quartet, pre-filtered to (J_target, T_target) ---
with open(MFDN_path, "r") as f:
    raw = [ln.strip() for ln in f if ln.strip()]
data = [ln.split() for ln in raw[1:] if len(ln.split()) == 12]

mfdn_by_quartet = {}
for toks in data:
    quart = (int(toks[COL_A]), int(toks[COL_B]), int(toks[COL_C]), int(toks[COL_D]))
    J = float(toks[COL_J]); T = int(toks[COL_T])
    if J == J_target and T == T_target:
        mfdn_by_quartet.setdefault(quart, []).append((J, T))

# --- read orbitals ---
with open(state_map_path, "r", newline="") as fh:
    rows = list(csv.reader(fh))
orbitals = [
    (int(r[0]), int(r[1]), int(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]))
    for r in rows[1:]
]

# --- collect every distinct CG argument tuple the decoupling loop would evaluate ---
cg_args = set()
for oa in orbitals:
    for ob in orbitals:
        for oc in orbitals:
            for od in orbitals:
                a, b, c, d = oa[0], ob[0], oc[0], od[0]
                j_x, j_y, j_z, j_s = oa[4], ob[4], oc[4], od[4]
                m_x, m_y, m_z, m_s = oa[5], ob[5], oc[5], od[5]
                mt_x, mt_y, mt_z, mt_s = oa[6], ob[6], oc[6], od[6]
                M = m_x + m_y
                Mt = mt_x + mt_y
                if not (m_x + m_y == m_z + m_s and mt_x + mt_y == mt_z + mt_s):
                    continue
                # orb-id quartet drives the MFDn lookup (orbital column 1)
                quart = (oa[1], ob[1], oc[1], od[1])
                for (Jl, Tl) in mfdn_by_quartet.get(quart, []):
                    cg_args.add((j_x, j_y, Jl, m_x, m_y, M))
                    cg_args.add((j_z, j_s, Jl, m_z, m_s, M))
                    cg_args.add((0.5, 0.5, Tl, mt_x, mt_y, Mt))
                    cg_args.add((0.5, 0.5, Tl, mt_z, mt_s, Mt))

print(f"Distinct CG argument tuples encountered: {len(cg_args)}")

tol = 1e-12
max_err = 0.0
worst = None
for args in sorted(cg_args):
    ref = float(cg_sympy(*args))
    got = cg_num(*args)
    err = abs(ref - got)
    if err > max_err:
        max_err = err
        worst = (args, ref, got)

print(f"Max |numeric - sympy| over all tuples: {max_err:.3e}")
if worst is not None:
    print(f"Worst tuple: args={worst[0]} sympy={worst[1]!r} numeric={worst[2]!r}")

assert max_err <= tol, f"CG mismatch {max_err:.3e} exceeds tolerance {tol:.0e}"
print(f"PASS: numeric CG matches sympy to within {tol:.0e} on all {len(cg_args)} tuples.")
