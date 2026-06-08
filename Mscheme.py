import csv, math, argparse
# Step 2: production path uses the pure-stdlib, lru_cache-memoized numeric Clebsch-Gordan
# (Condon-Shortley) from cg_numeric.py instead of sympy's symbolic implementation, which
# was ~100-1000x slower per call. Validated against sympy to <1e-12 by validate_cg.py.
from cg_numeric import clebsch_gordan

# Scaling to larger model spaces (emax2=40, emax4=140 single-particle states) drove a series
# of algorithmic fixes. The quartet transform was originally O(n^4) in the orbital count
# (emax4 -> 384M iterations, ~15 min). Profiling showed two facts: (1) of the 15.4M quartets
# that satisfy the (M, MT) conservation rules at emax4, only ~563k (1 in 28) actually have a
# matching MFDn record -- the rest contribute nothing and are filtered out; (2) the symbolic
# overhead was gone, so ~93% of the time was spent *rejecting* those unmatched quartets.
# The current code therefore drives the transform directly from the (few) target-channel MFDn
# quartets (see "Basis Decoupling" below), enumerating only the m-scheme substate quartets that
# can contribute. NumPy vectorization remains unnecessary: the residual cost is the unavoidable
# per-quartet CG accumulation, which is already < 1s, not the enumeration.

#=================================================Inputs and initialization:===========================================================#
#number or particles, protons, neutrons, frequency
A=2 ; Z = 1; Neut = A - Z; hw=16

# Input/output paths are CLI-parametrized (defaults reproduce the original hardcoded run),
# so the same script handles any model space, e.g.:
#   python3 Mscheme.py --state-map emax4_state_map.csv --out-mscheme emax4_Mscheme.csv
_p = argparse.ArgumentParser(
    description="Transform MFDn coupled-JT two-body matrix elements to the M-scheme single-particle basis.")
_p.add_argument("--state-map", default="state_map_2B.csv",
                help="single-particle state map CSV (lists each orbital's id and quantum numbers)")
_p.add_argument("--mfdn", default="Helium_2_data_N3LO_EM500_srg1.0_hw16_emax4_e2max8.MFDn",
                help="MFDn two-body matrix-element file in the coupled JT basis")
_p.add_argument("--out-mscheme", default="Mscheme.csv", help="output M-scheme TBME CSV")
_p.add_argument("--out-tsingle", default="Tsingle.csv", help="output one-body kinetic-energy CSV")
_args = _p.parse_args()

state_map_path = _args.state_map
MFDN_path = _args.mfdn

T_columns = []

# Step 1 optimization: index MFDn records by their orbital quartet, pre-filtered to the
# target (J,T) channel. The original code linearly scanned all ~11,759 records for every
# M-scheme quartet (O(N_quartets * N_mfdn)); this dict gives O(1) lookup. Insertion order
# is preserved so the accumulation/summation order is byte-for-byte unchanged.
mfdn_by_quartet = {}

J_target = 1
T_target = 0
#===============================================MFDN file Reader========================================================================#

# -----------------------------------------------------------------------------
# MFDn column format:
# a  b  c  d  J  T  Trel  Hrel  Vcoul  Vpn  Vpp  Vnn
# 0  1  2  3  4  5   6     7      8      9   10   11
# -----------------------------------------------------------------------------
COL_A, COL_B, COL_C, COL_D = 0, 1, 2, 3
COL_J, COL_T               = 4, 5
COL_TREL = 6
COL_VPN, COL_VPP, COL_VNN  = 9, 10, 11
with open(MFDN_path, "r") as f:
    raw = [ln.strip() for ln in f if ln.strip()]
data = [ln.split() for ln in raw[1:] if len(ln.split()) == 12]

for toks in data:
    J = float(toks[COL_J]); T = int(toks[COL_T])
    # Pre-filter to the target channel only (the decoupling discards everything else), and
    # bucket by orbital quartet. Append preserves original file order, so the per-quartet
    # accumulation order downstream is unchanged.
    if J == J_target and T == T_target:
        mfdn_list = [int(toks[COL_A]), int(toks[COL_B]), int(toks[COL_C]), int(toks[COL_D])]
        Trel = float(toks[COL_TREL]); Vpn = float(toks[COL_VPN]); Vpp = float(toks[COL_VPP]); Vnn = float(toks[COL_VNN])
        record = [mfdn_list, J, T, Trel, Vpn, Vpp, Vnn]
        mfdn_by_quartet.setdefault(tuple(mfdn_list), []).append(record)

#===============================================State Map file Reader========================================================================#
with open(state_map_path, "r", newline="") as file:
    reader = csv.reader(file)
    rows = list(reader)

# Step 3: parse the single-particle orbitals once into typed records, instead of re-casting
# CSV string cells inside the hot loops. Each record: (id, orb, n, l, j, m, mt).
orbitals = [
    (int(row[0]), int(row[1]), int(row[2]),
     float(row[3]), float(row[4]), float(row[5]), float(row[6]))
    for row in rows[1:]
]

# Index each coupled orbital (orb_id) -> its list of m-scheme substates (sp_id, j, m, mt),
# in sp_id order. The decoupling below is driven from the MFDn orbital quartets and looks up
# the substates of each orbital here. Keeping sp_id order lets us sort the generated quartets
# back into the original (a,b,c,d) order for byte-for-byte identical output.
orb_to_sub = {}
for o in orbitals:
    orb_to_sub.setdefault(o[1], []).append((o[0], o[4], o[5], o[6]))  # (sp_id, j, m, mt)

#=======================================================constructing singe body kinetic energy ===================================================================#

# Step 4: pre-parse the orbital rows once into typed records, instead of re-running
# int()/float() on the CSV string cells inside the O(n^2) loop body. The equality tests
# below are kept EXACTLY as in the original: the matching key (l, j, m, mt) and the first
# n-branch compare the raw string cells, while the arithmetic uses the float-parsed n, l.
ke_orbitals = []
for i in range(1, len(rows)):
    ke_orbitals.append((
        int(rows[i][0]),          # 0: sp_id
        rows[i][2],               # 1: n  (raw string, for the string-equality branch)
        float(rows[i][2]),        # 2: n  (float, for arithmetic)
        float(rows[i][3]),        # 3: l  (float, for arithmetic)
        (rows[i][3], rows[i][4], rows[i][5], rows[i][6]),  # 4: (l, j, m, mt) string key
    ))

perf = ((A-1)/A)*(hw/2)  # HO relative kinetic-energy prefactor (A-1)/A * hw/2; loop-invariant

for oi in ke_orbitals:
    for oj in ke_orbitals:
        if oi[4] == oj[4]:                      # same (l, j, m, mt) string-key as original
            if oi[1] == oj[1]:                  # n equal (string compare, as original)
                tij = perf * (2 * oi[2] + oi[3] + 1.5)
                T_columns.append([oi[0], oj[0], tij])

            elif oi[2] == oj[2] - 1:             # n_i == n_j - 1
                tij = perf * math.sqrt(oj[2] * (oj[2] + oj[3] + 0.5))
                T_columns.append([oi[0], oj[0], tij])

            elif oi[2] == oj[2] + 1:             # n_i == n_j + 1
                tij = perf * math.sqrt((oj[2] + 1) * (oj[3] + oj[2] + 1.5))
                T_columns.append([oi[0], oj[0], tij])

#==================================================Channel type helper=========================================================================#

#This ensures that CG transformation is only applied to the correct potential channel:
def pair_channel(mt1, mt2):
    pair = tuple(sorted((float(mt1), float(mt2))))
    if pair == (-0.5, -0.5):
        return "pp"
    elif pair == (0.5, 0.5):
        return "nn"
    elif pair == (-0.5, 0.5):
        return "pn"
    else:
        raise ValueError(f"Unexpected isospin pair: {(mt1, mt2)}")

# =====================================================Basis Decoupling======================================================================#

# Drive the transform directly from the target-channel MFDn quartets. For each coupled orbital
# quartet (x,y,z,s) we enumerate only the m-scheme substate quartets (a in x, b in y, c in z,
# d in s) that conserve (M, MT) -- found by pair-bucketing the ket (c,d) substates on their
# (M, MT) -- and accumulate the CG-weighted matrix elements. This visits only the ~563k
# quartets that can contribute (emax4), never the 15.4M conserved-but-unmatched ones. The
# numerics are identical to the original: same CG calls and multiply order, same loop-invariant
# normalization, and the same per-quartet accumulation over `locations` in MFDn file order.
# Rows are sorted by (a,b,c,d) afterwards to reproduce the original output ordering exactly.
data1 = []
for (x, y, z, s), locations in mfdn_by_quartet.items():
    subA = orb_to_sub.get(x); subB = orb_to_sub.get(y)
    subC = orb_to_sub.get(z); subD = orb_to_sub.get(s)
    # Skip quartets whose orbitals are outside this model space's state map (e.g. an emax2 run
    # against the emax4 MFDn file): those orbitals simply have no m-scheme substates here.
    if not (subA and subB and subC and subD):
        continue

    # normalization is loop-invariant for the orbital quartet (depends only on x==y, z==s)
    norm = math.sqrt(1 + (1 if x == y else 0)) * math.sqrt(1 + (1 if z == s else 0))

    # bucket ket substate pairs (c,d) by (M, MT); precompute each pair's charge channel once
    ket_buckets = {}
    for sc in subC:
        for sd in subD:
            ket_buckets.setdefault((sc[2] + sd[2], sc[3] + sd[3]), []).append(
                (sc, sd, pair_channel(sc[3], sd[3])))

    for sa in subA:
        a = sa[0]; j_x = sa[1]; m_x = sa[2]; mt_x = sa[3]
        for sb in subB:
            b = sb[0]; j_y = sb[1]; m_y = sb[2]; mt_y = sb[3]
            M = m_x + m_y; Mt = mt_x + mt_y
            kets = ket_buckets.get((M, Mt))
            if not kets:
                continue
            bra_channel = pair_channel(mt_x, mt_y)

            for sc, sd, ket_channel in kets:
                # charge-conserving strong interaction: bra and ket must share the channel
                if bra_channel != ket_channel:
                    continue
                channel = bra_channel
                c = sc[0]; j_z = sc[1]; m_z = sc[2]; mt_z = sc[3]
                d = sd[0]; j_s = sd[1]; m_s = sd[2]; mt_s = sd[3]

                sp_Trel = 0.0; sp_Vpn = 0.0; sp_Vpp = 0.0; sp_Vnn = 0.0
                for rec in locations:
                    J_location    = rec[1]
                    T_location    = rec[2]
                    Trel_location = rec[3]
                    Vpn_location  = rec[4]
                    Vpp_location  = rec[5]
                    Vnn_location  = rec[6]

                    cg_momentum = (clebsch_gordan(j_x, j_y, J_location, m_x, m_y, M)
                                * clebsch_gordan(j_z, j_s, J_location, m_z, m_s, M))

                    cg_isospin = (clebsch_gordan(1/2, 1/2, T_location, mt_x, mt_y, Mt)
                               * clebsch_gordan(1/2, 1/2, T_location, mt_z, mt_s, Mt))

                    coeff = float(norm * cg_isospin * cg_momentum)

                    # Trel is isospin-independent, so keep accumulating it
                    sp_Trel += coeff * Trel_location

                    # Only accumulate the matching charge channel
                    if channel == "pn":
                        sp_Vpn += coeff * Vpn_location
                    elif channel == "pp":
                        sp_Vpp += coeff * Vpp_location
                    elif channel == "nn":
                        sp_Vnn += coeff * Vnn_location

                # drop quartets that contribute nothing (same as the original output filter)
                if sp_Trel == 0 and sp_Vpn == 0 and sp_Vpp == 0 and sp_Vnn == 0:
                    continue
                data1.append(((a, b, c, d), [x, y, z, s], sp_Trel, sp_Vpn, sp_Vpp, sp_Vnn))

#===========================================================Writing Output Files=============================================================#

# Sort by the m-scheme (a,b,c,d) sp_id quartet to reproduce the ORIGINAL output row order
# (the original built its dictionary by ascending (a,b,c,d) and wrote nonzero rows in order).
data1.sort(key=lambda row: row[0])

headers1 = ["M-scheme quartet", "J-coupled quartet", "Trel", "Vpn", "Vpp", "Vnn"]

with open(_args.out_mscheme, "w", newline="") as fout1:
    w = csv.writer(fout1)
    w.writerow(headers1)
    w.writerows([list(sp_q), orb_q, tr, vpn, vpp, vnn] for sp_q, orb_q, tr, vpn, vpp, vnn in data1)

with open(_args.out_tsingle, "w", newline="") as fout2:
    w = csv.writer(fout2)
    w.writerows(T_columns)
