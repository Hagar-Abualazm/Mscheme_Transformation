"""Pure-stdlib numeric Clebsch-Gordan coefficient (Condon-Shortley convention).

Computes <j1 m1, j2 m2 | j3 m3> using the Racah closed form with log-gamma factorials
for numerical stability, matching the convention of
``sympy.physics.wigner.clebsch_gordan``. No third-party dependencies.

The arguments are physical angular-momentum quantum numbers (integer or half-integer,
passed as floats). Results are memoized: the decoupling loop hits only a few dozen distinct
argument tuples but calls them many times, so caching collapses the work to those few
evaluations.

Reference: Racah's formula, e.g. Edmonds, "Angular Momentum in Quantum Mechanics", Eq. 3.6.11;
equivalent to the Wigner 3-j relation used by sympy.
"""

import math
from functools import lru_cache


def _logfact(n):
    """log(n!) for non-negative integer n, via lgamma(n+1). Stable for large n."""
    return math.lgamma(n + 1)


@lru_cache(maxsize=None)
def clebsch_gordan(j1, j2, j3, m1, m2, m3):
    """Numeric <j1 m1, j2 m2 | j3 m3> as a float (Condon-Shortley).

    Parameters are floats (integer or half-integer). Returns 0.0 when the standard
    selection rules are violated (m3 != m1+m2, triangle inequality, or non-integer
    j1+j2+j3), exactly as sympy's clebsch_gordan does.

    Complexity: O(number of terms in the Racah sum), with results cached per argument tuple.
    """
    # Selection rule: projections must add up.
    if abs(m1 + m2 - m3) > 1e-9:
        return 0.0
    # Triangle condition on (j1, j2, j3).
    if j3 < abs(j1 - j2) - 1e-9 or j3 > j1 + j2 + 1e-9:
        return 0.0
    # j1+j2+j3 must be a (near-)integer for a non-zero coefficient.
    if abs((j1 + j2 + j3) - round(j1 + j2 + j3)) > 1e-9:
        return 0.0
    # |m| <= j for each.
    if abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(m3) > j3 + 1e-9:
        return 0.0

    # The factorial arguments below must be non-negative integers; round the
    # half-integer combinations that are guaranteed integral by the selection rules.
    def ri(x):
        return int(round(x))

    j1j2mj3 = ri(j1 + j2 - j3)
    j1mj2j3 = ri(j1 - j2 + j3)
    mj1j2j3 = ri(-j1 + j2 + j3)
    j1j2j3p1 = ri(j1 + j2 + j3 + 1)

    j1pm1 = ri(j1 + m1); j1mm1 = ri(j1 - m1)
    j2pm2 = ri(j2 + m2); j2mm2 = ri(j2 - m2)
    j3pm3 = ri(j3 + m3); j3mm3 = ri(j3 - m3)

    # Overall prefactor: sqrt( (2 j3 + 1) * [triangle/factorial product] ).
    # Computed in log space then exponentiated for stability.
    log_pref = 0.5 * (
        math.log(2.0 * j3 + 1.0)
        + _logfact(j1j2mj3) + _logfact(j1mj2j3) + _logfact(mj1j2j3)
        - _logfact(j1j2j3p1)
        + _logfact(j1pm1) + _logfact(j1mm1)
        + _logfact(j2pm2) + _logfact(j2mm2)
        + _logfact(j3pm3) + _logfact(j3mm3)
    )
    prefactor = math.exp(log_pref)

    # Racah summation over k. Each term: (-1)^k / [ k! (j1+j2-j3-k)! (j1-m1-k)!
    #                                              (j2+m2-k)! (j3-j2+m1+k)! (j3-j1-m2+k)! ].
    # k ranges over all integers keeping every factorial argument non-negative.
    kmin = max(0, ri(j2 - j3 - m1), ri(j1 - j3 + m2))
    kmax = min(j1j2mj3, j1mm1, j2pm2)

    total = 0.0
    for k in range(kmin, kmax + 1):
        d1 = j1j2mj3 - k
        d2 = j1mm1 - k
        d3 = j2pm2 - k
        d4 = ri(j3 - j2 + m1) + k
        d5 = ri(j3 - j1 - m2) + k
        if d1 < 0 or d2 < 0 or d3 < 0 or d4 < 0 or d5 < 0:
            continue
        log_term = (
            _logfact(k) + _logfact(d1) + _logfact(d2)
            + _logfact(d3) + _logfact(d4) + _logfact(d5)
        )
        term = math.exp(-log_term)
        if k & 1:
            term = -term
        total += term

    return prefactor * total
