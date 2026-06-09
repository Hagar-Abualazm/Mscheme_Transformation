# Using `Mscheme.py`

`Mscheme.py` transforms two-body matrix elements (TBMEs) of a nuclear interaction from the
**coupled JT basis** (the MFDn file format) into the **M-scheme single-particle basis**, for
the target channel **J = 1, T = 0**. It also writes the one-body harmonic-oscillator kinetic
energy. It works on the bundled emax1–emax8 examples and on **any `.MFDn` file in the same
format** paired with a matching single-particle state map.

- Outputs: `Mscheme.csv` (decoupled TBMEs) and `Tsingle.csv` (one-body kinetic energy).
- Pure Python 3 standard library — **no packages to install**. (SymPy is only needed for the
  optional correctness cross-check in `validate_cg.py`, never for a normal run.)

---

## Quick start

```bash
# Defaults reproduce the original small (emax1) run:
python3 Mscheme.py

# Any model space — point it at a state map + an MFDn file:
python3 Mscheme.py \
  --state-map emax4_state_map.csv \
  --mfdn      Helium_2_data_N3LO_EM500_srg1.0_hw16_emax4_e2max8.MFDn \
  --out-mscheme Mscheme.csv \
  --out-tsingle Tsingle.csv
```

For a **large** interaction (e.g. emax8), add `--jobs N` to run on `N` cores — see
[Large interactions](#large-interactions-the---jobs-flag).

---

## Command-line options

| Flag | Default | Meaning |
|------|---------|---------|
| `--state-map` | `state_map_2B.csv` | single-particle state map CSV |
| `--mfdn` | `Helium_2_..._emax4_e2max8.MFDn` | MFDn coupled-JT TBME file |
| `--out-mscheme` | `Mscheme.csv` | output M-scheme TBME CSV |
| `--out-tsingle` | `Tsingle.csv` | output one-body kinetic-energy CSV |
| `--jobs` | `1` | worker processes (`1` = serial; `>1` = parallel, identical output) |

Run `python3 Mscheme.py -h` for the same list.

---

## Input file formats

### MFDn file (`--mfdn`)

Plain text. **The first line is a header and is skipped.** Every remaining line is one coupled
TBME with **12 whitespace-separated columns**:

```
 a   b   c   d   J   T   Trel        Hrel        Vcoul       Vpn         Vpp         Vnn
 0   1   2   3   4   5    6           7            8           9          10          11
```

- `a b c d` are **orbital ids** (they must match the `orb_id` column of the state map).
- The transform uses only columns **a,b,c,d, J, T, Trel, Vpn, Vpp, Vnn** (7 and 8 are ignored).
- Only rows with **J = 1 and T = 0** are used; all other channels are skipped.

Example (the bundled emax8 file starts like this):

```
     1180279   8  16 16.00            <- header (record count, emax, e2max, hw): skipped
   1   1   1   1   0   1   0.75000000  1.50000000  0.00000000  -7.29...  -6.28...  -7.07...
   1   1   1   1   1   0   0.75000000  1.50000000  0.00000000 -13.00...   0.00...   0.00...
```

Any file in this layout works — just pass it with `--mfdn`.

### State map (`--state-map`)

CSV with a header row, then one row per single-particle (m-scheme) state. The first **7
columns are required** (an 8th `e` column may be present but is ignored):

```
sp_id,orb_id,n,l,j,m,mt[,e]
0,1,0,0,0.5,-0.5,-0.5,0
1,1,0,0,0.5,0.5,-0.5,0
...
```

| column | meaning |
|--------|---------|
| `sp_id` | m-scheme single-particle index (unique per row) |
| `orb_id` | coupled-orbital index — **must match the `a,b,c,d` ids in the MFDn file** |
| `n, l, j` | radial, orbital, total angular-momentum quantum numbers |
| `m, mt` | angular-momentum and isospin projections (mt = −1/2 proton, +1/2 neutron) |

**The state map must cover every orbital the MFDn references.** Quartets whose orbitals are
absent from the state map are silently skipped (this is how a smaller state map run against a
larger MFDn produces a valid sub-space result). For the bundled files, match emax to emax:
e.g. the emax8 MFDn (orb_ids 1–45) goes with `emax8_state_map.csv`.

---

## Output formats

### `Mscheme.csv`
Header row, then one row per non-zero M-scheme TBME, sorted by the `(a,b,c,d)` sp_id quartet:

```
M-scheme quartet,J-coupled quartet,Trel,Vpn,Vpp,Vnn
"[0, 1, 2, 3]","[1, 1, 1, 1]",0.75,-13.0,0.0,0.0
```

- column 1: the four **sp_id**s `[a,b,c,d]`
- column 2: the four **orb_id**s `[x,y,z,s]` they map to
- columns 3–6: `Trel, Vpn, Vpp, Vnn` (only the channel matching the quartet's isospin is filled)

### `Tsingle.csv`
No header; one row per non-zero one-body kinetic-energy element: `sp_id_i, sp_id_j, T_ij`.

---

## Large interactions: the `--jobs` flag

For big interactions the target quartets are independent, so the work parallelizes. `--jobs N`
splits them across `N` worker processes; each worker streams its (sorted) rows to a temp file
and the parent merges them. **The output is byte-for-byte identical to a serial run.**

```bash
# emax8 example on a 10-core machine (uses 9 workers):
python3 Mscheme.py \
  --state-map emax8_state_map.csv \
  --mfdn      HELIUM_emax8_N3LO_EM500_srg1.0_hw16_emax8_e2max16.MFDn \
  --jobs 9 \
  --out-mscheme emax8_Mscheme.csv \
  --out-tsingle emax8_Tsingle.csv
```

Guidance:
- A good choice is **(number of cores − 1)**. `--jobs 1` (the default) is plain serial.
- Speedup is sub-linear: parsing the MFDn, the final merge, and writing the output are serial,
  so very large files are partly I/O-bound. (emax8: ~64 s serial → ~29 s on 9 cores.)
- The parallel run writes temporary files roughly the size of the final output; make sure
  there is enough free disk (emax8 needs a few GB of scratch + the multi-hundred-MB result).

---

## Runtime & resource expectations

Runtime is set by the **interaction**, not the single-particle space: it scales ~linearly with
the number of target (J=1, T=0) MFDn quartets (order ~1 ms per quartet on a typical machine).
The state map can grow freely with negligible cost. See `PERFORMANCE.md` for the measured
scaling and plots.

- **Memory:** the parent reads the whole MFDn file (a ~136 MB file → ~1.5 GB while parsing).
  Serial mode also holds all output rows before sorting; for very large outputs prefer `--jobs`,
  which keeps each worker's footprint bounded.
- **Output size:** can be large — the emax8 case is ~3.7 M rows / ~274 MB.

---

## Changing the physics parameters

A few quantities are set near the top of `Mscheme.py` (they describe the system, not the file
layout). Edit them there if your case differs:

```python
A = 2 ; Z = 1 ; hw = 16      # mass number, proton number, oscillator frequency (MeV)
J_target = 1 ; T_target = 0  # coupled channel to extract
```

- `hw` feeds the kinetic energy `T_ij` and **must match the interaction's oscillator frequency**
  (the bundled files use ℏω = 16 MeV).
- `A, Z` set the `(A−1)/A` center-of-mass correction in `T_ij`; change them for a different
  nucleus.
- `J_target, T_target` select the coupled channel; change to extract a different one.

---

## Verifying correctness

- `python3 Mscheme.py --jobs 1 ...` and `--jobs N` must produce identical output (they do;
  this is checked against archived golden references in `refs/`).
- `validate_cg.py` cross-checks the numeric Clebsch–Gordan in `cg_numeric.py` against SymPy
  (this is the only script that imports SymPy).
- To compare two outputs: `diff -q a.csv b.csv` (exact), or `compare_outputs.py` for a numeric
  tolerance comparison.

---

## Large files & Git

MFDn files and outputs for big model spaces can exceed **GitHub's 100 MB limit** (the emax8
MFDn is 136 MB; its `Mscheme.csv` is 274 MB). These are kept local and are listed in
`.gitignore`. To share/version them, use **[Git LFS](https://git-lfs.com/)**:

```bash
git lfs install
git lfs track "*.MFDn" "emax8_Mscheme.csv"
git add .gitattributes && git commit -m "track large files with LFS"
```
