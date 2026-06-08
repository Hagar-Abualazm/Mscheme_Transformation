import csv, math
# Step 2: production path uses the pure-stdlib, lru_cache-memoized numeric Clebsch-Gordan
# (Condon-Shortley) from cg_numeric.py instead of sympy's symbolic implementation, which
# was ~100-1000x slower per call. Validated against sympy to <1e-12 by validate_cg.py.
from cg_numeric import clebsch_gordan

# Step 5 (NumPy vectorization) — DEFERRED, intentionally not done.
# At the current model space (emax4: 16 orbitals, ~11,759 MFDn records) the runtime is
# ~0.17s, so vectorizing the accumulation loop with NumPy isn't worth the correctness risk
# (summation-order changes would perturb the last ULPs and break exact reproducibility).
# Revisit ONLY if the model space grows (larger emax) and the runtime climbs back into the
# seconds-to-minutes range; profile first to confirm the accumulation loop is the hot spot.

#=================================================Inputs and initialization:===========================================================#
#number or particles, protons, neutrons, frequency
A=2 ; Z = 1; Neut = A - Z; hw=16

#state_map is a file that lists all single-particle orbitals id and their quantum numbers
state_map_path = "state_map_2B.csv"

#MFDn is the two-body-matrix elements file in the coupled JT basis
MFDN_path = "Helium_2_data_N3LO_EM500_srg1.0_hw16_emax4_e2max8.MFDn"

T_columns = []; sp_tbme = {}; mfdn_tbme = {}

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

for r, toks in enumerate(data):
    mfdn_list = [int(toks[COL_A]), int(toks[COL_B]), int(toks[COL_C]), int(toks[COL_D])]
    J = float(toks[COL_J]); T = int(toks[COL_T])
    Trel = float(toks[COL_TREL]); Vpn = float(toks[COL_VPN]); Vpp = float(toks[COL_VPP]); Vnn = float(toks[COL_VNN])

    #build the MFDn dictionary
    record = [mfdn_list, J, T, Trel, Vpn, Vpp, Vnn]
    mfdn_tbme[str(r+1)] = record

    # Pre-filter to the target channel only (the decoupling loop discards everything else),
    # and bucket by quartet tuple. Append preserves original file/insertion order.
    if J == J_target and T == T_target:
        mfdn_by_quartet.setdefault(tuple(mfdn_list), []).append(record)

#===============================================State Map file Reader========================================================================#
with open(state_map_path, "r", newline="") as file:
    reader = csv.reader(file)
    rows = list(reader)

# Step 3: parse the single-particle orbitals once into typed records, instead of re-casting
# CSV string cells inside the 4-deep (O(n^4)) loop. Each record: (id, orb, n, l, j, m, mt).
orbitals = [
    (int(row[0]), int(row[1]), int(row[2]),
     float(row[3]), float(row[4]), float(row[5]), float(row[6]))
    for row in rows[1:]
]

# Step 3: O(1) set membership for quartet de-duplication, replacing the former
# `[a,b,c,d] not in sp_quartet` linear scan of a growing list (which made the build O(n^8)).
sp_quartet_seen = set()
for oa in orbitals:
    for ob in orbitals:
        for oc in orbitals:
            for od in orbitals:

                #list all the unique single-particle orbital quartets:
                a = oa[0]; b = ob[0]; c = oc[0]; d = od[0]
                quartet_key = (a, b, c, d)
                if quartet_key not in sp_quartet_seen:
                    sp_quartet_seen.add(quartet_key)

                    #get their quantum numbers (orb, n, l, j, m, mt) from the parsed records:
                    a_orb, a_n, a_l, a_j, a_m, a_mt = oa[1], oa[2], oa[3], oa[4], oa[5], oa[6]
                    b_orb, b_n, b_l, b_j, b_m, b_mt = ob[1], ob[2], ob[3], ob[4], ob[5], ob[6]
                    c_orb, c_n, c_l, c_j, c_m, c_mt = oc[1], oc[2], oc[3], oc[4], oc[5], oc[6]
                    d_orb, d_n, d_l, d_j, d_m, d_mt = od[1], od[2], od[3], od[4], od[5], od[6]  # mt (proton) = -1/2  # mt (neutron) = +1/2

                    # Conservation of angular momentum and isospin selection rules (only include valid quartets):
                    if a_m + b_m == c_m + d_m and a_mt + b_mt == c_mt + d_mt:

                        # build the sp-dictionary
                        sp_key = str(a) + "," + str(b) + "," + str(c) + "," + str(d)
                        sp_tbme[sp_key] = [[a, b, c, d], [a_orb, b_orb, c_orb, d_orb], [a_n, b_n, c_n, d_n], [a_l, b_l, c_l, d_l],
                                           [a_j, b_j, c_j, d_j], [a_m, b_m, c_m, d_m], [a_mt, b_mt, c_mt, d_mt], 0, 0, 0, 0]

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

for key in sp_tbme.keys():
    #for each quartet in the single particle dictionary:

    x = sp_tbme[key][1][0];    y = sp_tbme[key][1][1];    z = sp_tbme[key][1][2];    s = sp_tbme[key][1][3]
    j_x = sp_tbme[key][4][0];  j_y = sp_tbme[key][4][1];  j_z = sp_tbme[key][4][2];  j_s = sp_tbme[key][4][3]
    m_x = sp_tbme[key][5][0];  m_y = sp_tbme[key][5][1];  m_z = sp_tbme[key][5][2];  m_s = sp_tbme[key][5][3]
    mt_x = sp_tbme[key][6][0]; mt_y = sp_tbme[key][6][1]; mt_z = sp_tbme[key][6][2]; mt_s = sp_tbme[key][6][3]

    #total M and isospin
    M  = m_x + m_y
    Mt = mt_x + mt_y

    sp_Trel = 0.0; sp_Vpn  = 0.0; sp_Vpp  = 0.0; sp_Vnn  = 0.0

    #checking interaction channel type:
    bra_channel = pair_channel(mt_x, mt_y); ket_channel = pair_channel(mt_z, mt_s)
    # For a charge-conserving strong interaction, bra and ket should be in the same channel.
    if bra_channel != ket_channel:
        sp_tbme[key][7]  = 0.0; sp_tbme[key][8]  = 0.0; sp_tbme[key][9]  = 0.0; sp_tbme[key][10] = 0.0
        continue

    channel = bra_channel

    #find all the location(s)/coupled interactions in the MFDN file that contribute to this
    # single-particle quartet. Step 1: O(1) dict lookup into the pre-filtered (J=J_target,
    # T=T_target) quartet index, replacing the former O(N_mfdn) linear scan. The returned
    # records are already in original file order, so accumulation order is unchanged.
    locations = mfdn_by_quartet.get((x, y, z, s), [])
    if locations:
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

            # Normalization Factors: Keep or remove this depending on convention
            norm_xy = math.sqrt(1 + (1 if x == y else 0))
            norm_zs = math.sqrt(1 + (1 if z == s else 0))
            norm = norm_xy * norm_zs

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

        # Step 4: write the accumulated channel values once, after the location loop,
        # instead of on every iteration (the running totals are unchanged).
        sp_tbme[key][7]=sp_Trel;  sp_tbme[key][8]=sp_Vpn;  sp_tbme[key][9]=sp_Vpp;  sp_tbme[key][10]=sp_Vnn

#===========================================================Writing Output Files=============================================================#

headers1 = ["M-scheme quartet", "J-coupled quartet", "Trel", "Vpn", "Vpp", "Vnn"]
data1 = []
for key in sp_tbme.keys():
    if sp_tbme[key][7] == 0 and sp_tbme[key][8] == 0 and sp_tbme[key][9] == 0 and sp_tbme[key][10] == 0:
        continue
    else:
        data1.append([sp_tbme[key][0], sp_tbme[key][1], sp_tbme[key][7], sp_tbme[key][8], sp_tbme[key][9],sp_tbme[key][10]])

with open("Mscheme.csv", "w", newline="") as fout1:
    w = csv.writer(fout1)
    w.writerow(headers1)
    w.writerows(data1)

with open("Tsingle.csv", "w", newline="") as fout2:
    w = csv.writer(fout2)
    w.writerows(T_columns)
