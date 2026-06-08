import csv, math
from sympy.physics.wigner import clebsch_gordan

#=================================================Inputs and initialization:===========================================================#
#number or particles, protons, neutrons, frequency
A=2 ; Z = 1; Neut = A - Z; hw=16

#state_map is a file that lists all single-particle orbitals id and their quantum numbers
state_map_path = "/Users/habual01/NSCM/state_map_2B.csv"

#MFDn is the two-body-matrix elements file in the coupled JT basis
MFDN_path = "/Users/habual01/NSCM/NuHamil-public/exe/Helium_2_data_N3LO_EM500_srg1.0_hw16_emax4_e2max8.MFDn"

T_columns = []; sp_tbme = {}; mfdn_tbme = {}

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
    mfdn_tbme[str(r+1)] = [mfdn_list, J, T, Trel, Vpn, Vpp, Vnn]

#===============================================State Map file Reader========================================================================#
with open(state_map_path, "r", newline="") as file:
    reader = csv.reader(file)
    rows = list(reader)

sp_quartet = []
for i in range(1, len(rows)):
    for j in range(1, len(rows)):
        for k in range(1, len(rows)):
            for l in range(1, len(rows)):

                #list all the unique single-particle orbital quartets:
                a = int(rows[i][0]);       b = int(rows[j][0]);       c = int(rows[k][0]);        d = int(rows[l][0])
                if [a, b, c, d] not in sp_quartet:
                    sp_quartet.append([a, b, c, d])

                #get their quantum numbers from the state map file:

                    a_orb = int(rows[i][1]);   b_orb = int(rows[j][1]);   c_orb = int(rows[k][1]);    d_orb = int(rows[l][1])

                    a_n = int(rows[i][2]);     b_n = int(rows[j][2]);     c_n = int(rows[k][2]);      d_n = int(rows[l][2])

                    a_l = float(rows[i][3]);   b_l = float(rows[j][3]);   c_l = float(rows[k][3]);    d_l = float(rows[l][3])

                    a_j = float(rows[i][4]);   b_j = float(rows[j][4]);   c_j = float(rows[k][4]);    d_j = float(rows[l][4])

                    a_m = float(rows[i][5]);   b_m = float(rows[j][5]);   c_m = float(rows[k][5]);    d_m = float(rows[l][5])

                    a_mt = float(rows[i][6]); b_mt = float(rows[j][6]); c_mt = float(rows[k][6]); d_mt = float(rows[l][6])  # mt (proton) = -1/2  # mt (neutron) = +1/2

                    # Conservation of angular momentum and isospin selection rules (only include valid quartets):
                    if a_m + b_m == c_m + d_m and a_mt + b_mt == c_mt + d_mt:

                        # build the sp-dictionary
                        sp_key = str(a) + "," + str(b) + "," + str(c) + "," + str(d)
                        sp_tbme[sp_key] = [[a, b, c, d], [a_orb, b_orb, c_orb, d_orb], [a_n, b_n, c_n, d_n], [a_l, b_l, c_l, d_l],
                                           [a_j, b_j, c_j, d_j], [a_m, b_m, c_m, d_m], [a_mt, b_mt, c_mt, d_mt], 0, 0, 0, 0]

#=======================================================constructing singe body kinetic energy ===================================================================#

for i in range(1, len(rows)):
    for j in range(1, len(rows)):
        if rows[i][3] == rows[j][3] and rows[i][4] == rows[j][4] and rows[i][5] == rows[j][5] and rows[i][6] == rows[j][6]:
            perf = ((A-1)/A)*(hw/2)

            if rows[i][2] == rows[j][2]:
                tij = perf * (2 * float(rows[i][2]) + float(rows[i][3]) + 1.5)
                p = int(rows[i][0])
                q = int(rows[j][0])
                T_columns.append([p,q,tij])

            elif float(rows[i][2]) == float(rows[j][2]) - 1:
                tij = perf * math.sqrt(float(rows[j][2]) * (float(rows[j][2]) + float(rows[j][3]) + 0.5))
                p = int(rows[i][0])
                q = int(rows[j][0])
                T_columns.append([p, q, tij])

            elif float(rows[i][2]) == float(rows[j][2]) + 1:
                tij = perf * math.sqrt((float(rows[j][2]) + 1) * (float(rows[j][3]) + float(rows[j][2]) + 1.5))
                p = int(rows[i][0])
                q = int(rows[j][0])
                T_columns.append([p, q, tij])

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

    quartet = [x, y, z, s]
    #find all the location(s)/coupled interactions in the MFDN file that contribute to this single-particle quartet:
    locations = [k for k, v in mfdn_tbme.items() if v[0] == quartet]
    if locations:
        for location in locations:
            J_location    = mfdn_tbme[location][1]
            T_location    = mfdn_tbme[location][2]
            Trel_location = mfdn_tbme[location][3]
            Vpn_location  = mfdn_tbme[location][4]
            Vpp_location  = mfdn_tbme[location][5]
            Vnn_location  = mfdn_tbme[location][6]

            if J_location != J_target or T_location != T_target:
                continue
            else:
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

                #update the dictionary:
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