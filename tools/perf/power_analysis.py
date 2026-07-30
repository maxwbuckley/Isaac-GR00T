"""Power analysis for the B4 A/B, using the observed paired variance.

Paired t-test: the test statistic is t = mean(d) / (sd(d)/sqrt(n)) on the n
per-round differences, so the only quantities that matter are mean(d) and sd(d).
Power is computed exactly from the noncentral t distribution, not the normal
approximation, because n is small enough for the difference to matter.
"""

import math

import numpy as np
from scipy import stats


ROUND_SECONDS = 82.0  # measured: 41 s per run x 2 arms

# Quiet-system rounds 7-16 (medians per round)
b4_data = [7.70, 7.76, 7.79, 7.98, 7.46, 7.81, 7.70, 7.49, 7.34, 8.20]
mn_data = [8.78, 8.75, 9.07, 8.27, 8.59, 8.84, 7.94, 8.85, 8.91, 8.57]
b4_e2e = [144.2, 144.0, 140.0, 140.1, 138.0, 145.2, 140.6, 139.1, 143.0, 140.8]
mn_e2e = [147.2, 142.5, 145.5, 138.2, 143.8, 145.4, 138.6, 147.1, 147.1, 140.0]


def power_paired(n, delta, sd, alpha=0.05):
    """Exact power of a two-sided paired t-test via the noncentral t.

    scipy's nct returns nan at moderate-to-large noncentrality (e.g. df=9,
    ncp=15.8). Power is monotonically increasing in ncp, so a nan with ncp
    comfortably above the critical value is saturated power, and a nan with ncp
    far below it is ~0. Guarding this matters: an unguarded nan compares False
    against any threshold, which silently corrupts both the bisection and the
    smallest-n scan.
    """
    if n < 2:
        return 0.0
    df = n - 1
    ncp = delta / sd * math.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, df)
    upper = stats.nct.sf(crit, df, ncp)
    lower = stats.nct.cdf(-crit, df, ncp)
    if not np.isfinite(upper):
        upper = 1.0 if ncp > crit else 0.0
    if not np.isfinite(lower):
        lower = 0.0
    return float(upper + lower)


def n_for_power(delta, sd, target, alpha=0.05, nmax=100000):
    for n in range(2, nmax):
        if power_paired(n, delta, sd, alpha) >= target:
            return n
    return None


def mde(n, sd, target=0.80, alpha=0.05):
    """Minimum detectable effect at given n and power (bisection on delta)."""
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_paired(n, mid, sd) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def hhmm(seconds):
    h, m = divmod(int(round(seconds / 60)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def report(label, a, b, unit="ms"):
    d = np.array(b) - np.array(a)  # main - B4, positive = B4 faster
    n0, mean, sd = len(d), d.mean(), d.std(ddof=1)
    t = mean / (sd / math.sqrt(n0))
    p = 2 * stats.t.sf(abs(t), n0 - 1)
    dz = mean / sd

    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(f"  observed n           : {n0} rounds")
    print(f"  mean difference      : {mean:+.3f} {unit}")
    print(f"  sd of differences    : {sd:.3f} {unit}")
    print(f"  Cohen's dz           : {dz:.3f}")
    print(f"  t                    : {t:.2f} (df={n0 - 1}),  p = {p:.2e}")
    print(f"  achieved power at n={n0}: {power_paired(n0, mean, sd):.3f}")

    print(f"\n  n required to detect the OBSERVED effect ({mean:+.3f} {unit}):")
    for tgt in (0.80, 0.90, 0.95):
        n = n_for_power(mean, sd, tgt)
        print(f"    power {tgt:.0%}: n = {n:>5} rounds   "
              f"({hhmm(n * ROUND_SECONDS)} of machine time)")

    print(f"\n  minimum detectable effect at 80% power:")
    for n in (10, 20, 50, 100, 200):
        print(f"    n = {n:>4} rounds ({hhmm(n * ROUND_SECONDS):>7}): "
              f"MDE = {mde(n, sd):.3f} {unit}  "
              f"({mde(n, sd) / np.mean(b) * 100:.2f}% of baseline)")


report("DATA-PROCESSING STAGE (the stage B4 targets)", b4_data, mn_data)
report("END-TO-END", b4_e2e, mn_e2e)

# --- what variance reduction would buy instead of more n -------------------
d_e2e = np.array(mn_e2e) - np.array(b4_e2e)
mean_e, sd_e = d_e2e.mean(), d_e2e.std(ddof=1)
print(f"\n{'=' * 72}\nE2E: buying power by reducing sd instead of raising n\n{'=' * 72}")
print(f"  current sd = {sd_e:.2f} ms; n for 80% power = {n_for_power(mean_e, sd_e, 0.80)} rounds")
for factor in (0.75, 0.5, 0.25):
    n = n_for_power(mean_e, sd_e * factor, 0.80)
    print(f"    sd x {factor:<4} ({sd_e * factor:4.2f} ms): n = {n:>4} rounds  "
          f"({hhmm(n * ROUND_SECONDS)})")
