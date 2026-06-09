import csv, math, argparse, os, sys, tempfile, heapq
import multiprocessing as mp
# Step 2: production path uses the pure-stdlib, lru_cache-memoized numeric Clebsch-Gordan
# (Condon-Shortley) from cg_numeric.py instead of sympy's symbolic implementation, which
# was ~100-1000x slower per call. Validated against sympy to <1e-12 by validate_cg.py.
from cg_numeric import clebsch_gordan

# Scaling history. The quartet transform was originally O(n^4) in the orbital count (emax4 ->
# 384M iterations, ~15 min). It is now driven directly from the (few) target-channel MFDn
# quartets, enumerating only the m-scheme substate quartets that can contribute. Benchmarking
# showed runtime is bound by the *interaction* size (~1 ms per target quartet), not the number
# of single-particle states -- see PERFORMANCE.md. The emax8 interaction (e2max16) is the first
# that is genuinely large: ~41k target quartets -> ~62M output rows. For that regime there is an
# opt-in parallel path (--jobs N): the target quartets are independent, so each worker decouples
# a chunk and streams its rows (sorted) to a temp file, and the parent k-way-merges them. That
# keeps memory bounded (the full row set never lives in one process) and stays byte-for-byte
# identical to the serial path (the final order is a global sort by the (a,b,c,d) sp_id quartet).

# physical constants
A = 2; Z = 1; Neut = A - Z; hw = 16
J_target = 1
T_target = 0


# ==================================================Channel type helper=========================================================================#
# Ensures the CG transformation is applied to the correct potential channel.
def pair_channel(mt1, mt2):
    # mt are +-1/2, so the channel is fixed by their (exactly representable) sum, which avoids
    # the tuple(sorted(...)) allocation in the hot path: -1 -> pp, +1 -> nn, 0 -> pn.
    s = mt1 + mt2
    if s == -1.0:
        return "pp"
    elif s == 1.0:
        return "nn"
    elif s == 0.0:
        return "pn"
    else:
        raise ValueError(f"Unexpected isospin pair: {(mt1, mt2)}")


# =====================================================Basis Decoupling======================================================================#
def decouple_quartet(x, y, z, s, locations, orb_to_sub):
    """All M-scheme rows produced by one coupled orbital quartet (x,y,z,s).

    Shared by the serial and parallel paths so the two cannot diverge. For the quartet we
    enumerate only the substate quartets (a in x, b in y, c in z, d in s) that conserve
    (M, MT) -- found by pair-bucketing the ket (c,d) substates -- and accumulate the
    CG-weighted matrix elements over `locations` in MFDn file order. Returns a list of
    ((a,b,c,d), [x,y,z,s], Trel, Vpn, Vpp, Vnn); rows that contribute nothing are dropped
    (same as the original output filter).
    """
    subA = orb_to_sub.get(x); subB = orb_to_sub.get(y)
    subC = orb_to_sub.get(z); subD = orb_to_sub.get(s)
    out = []
    # Skip quartets whose orbitals are outside this state map (e.g. an emax2 map vs an emax4
    # MFDn): those orbitals simply have no m-scheme substates here.
    if not (subA and subB and subC and subD):
        return out

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

                if sp_Trel == 0 and sp_Vpn == 0 and sp_Vpp == 0 and sp_Vnn == 0:
                    continue
                out.append(((a, b, c, d), [x, y, z, s], sp_Trel, sp_Vpn, sp_Vpp, sp_Vnn))
    return out


# ---- parallel worker plumbing (only used when --jobs > 1) -----------------------------------
_G_ORB = None  # per-worker copy of orb_to_sub, set by the pool initializer


def _init_worker(orb_to_sub):
    global _G_ORB
    _G_ORB = orb_to_sub


def _worker(task):
    """Decouple a chunk of orbital quartets and stream the rows -- sorted by (a,b,c,d) -- to a
    temp file. Only the file path is returned, so the (huge) row set never crosses the IPC
    boundary or accumulates in the parent. repr() preserves exact float bits for the merge."""
    idx, items, tmpdir = task
    rows = []
    for (x, y, z, s), locations in items:
        rows.extend(decouple_quartet(x, y, z, s, locations, _G_ORB))
    rows.sort(key=lambda r: r[0])
    path = os.path.join(tmpdir, f"chunk_{idx:05d}.tsv")
    with open(path, "w") as fh:
        for (a, b, c, d), orb, tr, vpn, vpp, vnn in rows:
            fh.write(f"{a}\t{b}\t{c}\t{d}\t{orb[0]}\t{orb[1]}\t{orb[2]}\t{orb[3]}\t"
                     f"{tr!r}\t{vpn!r}\t{vpp!r}\t{vnn!r}\n")
    return path


def _read_chunk(path):
    """Yield ((a,b,c,d), output_row) from a worker temp file, rebuilding the exact CSV row."""
    with open(path) as fh:
        for line in fh:
            p = line.split("\t")
            a = int(p[0]); b = int(p[1]); c = int(p[2]); d = int(p[3])
            row = [[a, b, c, d], [int(p[4]), int(p[5]), int(p[6]), int(p[7])],
                   float(p[8]), float(p[9]), float(p[10]), float(p[11])]
            yield (a, b, c, d), row


HEADERS = ["M-scheme quartet", "J-coupled quartet", "Trel", "Vpn", "Vpp", "Vnn"]


def main():
    ap = argparse.ArgumentParser(
        description="Transform MFDn coupled-JT two-body matrix elements to the M-scheme single-particle basis.")
    ap.add_argument("--state-map", default="state_map_2B.csv",
                    help="single-particle state map CSV (lists each orbital's id and quantum numbers)")
    ap.add_argument("--mfdn", default="Helium_2_data_N3LO_EM500_srg1.0_hw16_emax4_e2max8.MFDn",
                    help="MFDn two-body matrix-element file in the coupled JT basis")
    ap.add_argument("--out-mscheme", default="Mscheme.csv", help="output M-scheme TBME CSV")
    ap.add_argument("--out-tsingle", default="Tsingle.csv", help="output one-body kinetic-energy CSV")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (default 1 = serial). Use for large interactions "
                         "(e.g. emax8); output is identical to serial.")
    args = ap.parse_args()

    #===============================================MFDN file Reader========================================================================#
    # column format:  a b c d  J T  Trel Hrel Vcoul  Vpn Vpp Vnn
    COL_A, COL_B, COL_C, COL_D = 0, 1, 2, 3
    COL_J, COL_T = 4, 5
    COL_TREL = 6
    COL_VPN, COL_VPP, COL_VNN = 9, 10, 11

    # Index MFDn records by orbital quartet, pre-filtered to the target (J,T) channel (everything
    # else is discarded downstream). Append preserves file order so per-quartet accumulation is
    # unchanged.
    mfdn_by_quartet = {}
    with open(args.mfdn, "r") as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    data = [ln.split() for ln in raw[1:] if len(ln.split()) == 12]
    del raw  # free the ~150 MB line list before the heavy phase (matters for the emax8 file)
    for toks in data:
        J = float(toks[COL_J]); T = int(toks[COL_T])
        if J == J_target and T == T_target:
            mfdn_list = [int(toks[COL_A]), int(toks[COL_B]), int(toks[COL_C]), int(toks[COL_D])]
            Trel = float(toks[COL_TREL]); Vpn = float(toks[COL_VPN]); Vpp = float(toks[COL_VPP]); Vnn = float(toks[COL_VNN])
            record = [mfdn_list, J, T, Trel, Vpn, Vpp, Vnn]
            mfdn_by_quartet.setdefault(tuple(mfdn_list), []).append(record)
    del data

    #===============================================State Map file Reader========================================================================#
    with open(args.state_map, "r", newline="") as file:
        rows = list(csv.reader(file))

    # parsed orbital records: (id, orb, n, l, j, m, mt)
    orbitals = [
        (int(row[0]), int(row[1]), int(row[2]),
         float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        for row in rows[1:]
    ]

    # orb_id -> its m-scheme substates (sp_id, j, m, mt), kept in sp_id order so the generated
    # quartets sort back into the original (a,b,c,d) order.
    orb_to_sub = {}
    for o in orbitals:
        orb_to_sub.setdefault(o[1], []).append((o[0], o[4], o[5], o[6]))

    #=======================================================one-body kinetic energy ===================================================================#
    # T_ij is nonzero only between orbitals sharing (l, j, m, mt) with n differing by 0 or +-1.
    # Group by that key (sp_id order preserved) so the inner loop only visits orbitals that can
    # couple -- O(states * group_size) instead of O(states^2) -- reproducing the original order.
    ke_orbitals = []
    for i in range(1, len(rows)):
        ke_orbitals.append((
            int(rows[i][0]),          # 0: sp_id
            rows[i][2],               # 1: n  (raw string, for the string-equality branch)
            float(rows[i][2]),        # 2: n  (float, for arithmetic)
            float(rows[i][3]),        # 3: l  (float, for arithmetic)
            (rows[i][3], rows[i][4], rows[i][5], rows[i][6]),  # 4: (l, j, m, mt) string key
        ))

    perf = ((A - 1) / A) * (hw / 2)  # HO relative kinetic-energy prefactor; loop-invariant
    ke_groups = {}
    for o in ke_orbitals:
        ke_groups.setdefault(o[4], []).append(o)

    T_columns = []
    for oi in ke_orbitals:
        for oj in ke_groups[oi[4]]:
            if oi[1] == oj[1]:                       # n equal (string compare, as original)
                tij = perf * (2 * oi[2] + oi[3] + 1.5)
                T_columns.append([oi[0], oj[0], tij])
            elif oi[2] == oj[2] - 1:                  # n_i == n_j - 1
                tij = perf * math.sqrt(oj[2] * (oj[2] + oj[3] + 0.5))
                T_columns.append([oi[0], oj[0], tij])
            elif oi[2] == oj[2] + 1:                  # n_i == n_j + 1
                tij = perf * math.sqrt((oj[2] + 1) * (oj[3] + oj[2] + 1.5))
                T_columns.append([oi[0], oj[0], tij])

    with open(args.out_tsingle, "w", newline="") as fout2:
        csv.writer(fout2).writerows(T_columns)

    #===========================================================Basis decoupling + output=============================================================#
    if args.jobs <= 1:
        # --- serial: build all rows, sort by (a,b,c,d), write ---
        data1 = []
        for (x, y, z, s), locations in mfdn_by_quartet.items():
            data1.extend(decouple_quartet(x, y, z, s, locations, orb_to_sub))
        data1.sort(key=lambda row: row[0])
        with open(args.out_mscheme, "w", newline="") as fout1:
            w = csv.writer(fout1)
            w.writerow(HEADERS)
            w.writerows([list(sp_q), orb_q, tr, vpn, vpp, vnn]
                        for sp_q, orb_q, tr, vpn, vpp, vnn in data1)
    else:
        # --- parallel: chunk the independent target quartets, each worker streams its sorted
        #     rows to a temp file, then a k-way merge writes the globally-sorted output ---
        items = list(mfdn_by_quartet.items())
        # more chunks than workers -> better load balance + smaller per-chunk memory; round-robin
        # so high-substate quartets are spread out.
        nchunks = max(args.jobs * 8, args.jobs)
        chunks = [items[i::nchunks] for i in range(nchunks)]
        # fork (when available) avoids re-importing the module in children; falls back to spawn.
        start = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start)
        print(f"[parallel] start={start} jobs={args.jobs} target_quartets={len(items)} chunks={nchunks}",
              file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(prefix="mscheme_") as tmpdir:
            tasks = [(i, ch, tmpdir) for i, ch in enumerate(chunks) if ch]
            with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(orb_to_sub,)) as pool:
                paths = pool.map(_worker, tasks)
            with open(args.out_mscheme, "w", newline="") as fout1:
                w = csv.writer(fout1)
                w.writerow(HEADERS)
                merged = heapq.merge(*[_read_chunk(p) for p in paths], key=lambda kr: kr[0])
                for _, row in merged:
                    w.writerow(row)


if __name__ == "__main__":
    main()
