"""Interval estimates used by every analysis.

Wilson score intervals for proportions, and the percentile bootstrap: resample item indices with replacement
(random.Random(seed), seed 0), compute the statistic on each resample, sort, and read the 2.5 % and 97.5 %
order statistics. random.choice(seq) and seq[random.randrange(len(seq))] draw the same index from the same
generator state, so both spellings give identical resamples.
"""
import math
import random

Z95 = 1.96


def wilson(k, n, z=Z95, clamp=False):
    """95 % Wilson interval (lo, hi) as fractions; (nan, nan) for n = 0. clamp limits it to [0, 1]."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h)) if clamp else (c - h, c + h)


def wilson_unscaled_center(k, n, z=Z95):
    """Same interval, written as ((p + z²/2n) ± h) / d with the division last; used by the depth analyses (it can
    differ from `wilson` in the last floating-point bit)."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def resample(rng, n):
    """One bootstrap resample of item indices 0..n-1."""
    return [rng.randrange(n) for _ in range(n)]


def resamples(n, B=10000, seed=0):
    """B resamples drawn up front (when several statistics must share the same resamples)."""
    rng = random.Random(seed)
    return [resample(rng, n) for _ in range(B)]


def bootstrap(stat, n, B=10000, seed=0):
    """Sorted bootstrap distribution of stat(indices) over B resamples of n items."""
    rng = random.Random(seed)
    return sorted(stat(resample(rng, n)) for _ in range(B))


def bootstrap_mean(values, B=10000, seed=0, scale=1):
    """Sorted bootstrap distribution of scale * mean(values)."""
    m = len(values)
    if scale == 1:
        return bootstrap(lambda idx: sum(values[i] for i in idx) / m, m, B, seed)
    return bootstrap(lambda idx: scale * sum(values[i] for i in idx) / m, m, B, seed)


def pct_ci(k, n):
    """'xx.x% [lo,hi] n=N' with the interval in whole percent (the format of most summaries)."""
    lo, hi = wilson(k, n) if n else (0, 0)
    return f"{100 * k / max(n, 1):5.1f}% [{100 * lo:.0f},{100 * hi:.0f}] n={n}"


def frac_ci(k, n):
    """Same, formatted from fractions ('xx.x% [lo%,hi%] n=N'); '-' for n = 0."""
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:6.1%} [{lo:.0%},{hi:.0%}] n={n}"


def percentile_ci(sorted_bs, lo=0.025, hi=0.975):
    """(bs[int(lo B)], bs[int(hi B) - 1]): the order statistics the analyses report as the 95 % interval."""
    B = len(sorted_bs)
    return sorted_bs[int(lo * B)], sorted_bs[int(hi * B) - 1]
