"""Friedman ranks with ties. Means that differ by at most 1e-8 relative (absolute below 1) count as tied, the same
tolerance as the CEC rule that sets errors below 1e-8 to zero; this keeps the ranks independent of the order in
which runs are summed."""
import numpy as np
from scipy import stats

RTOL = 1e-8


def trank(row, rtol=RTOL):
    row = np.asarray(row, float); o = np.argsort(row, kind='stable'); r = np.empty(len(row)); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and row[o[j + 1]] - row[o[i]] <= rtol * max(1.0, abs(row[o[i]])):
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def ranks(M):
    """Rank matrix, one row per block (function or problem), columns = algorithms."""
    return np.array([trank(x) for x in M])


def friedman(M):
    """Friedman chi-square with the tie correction of scipy.stats.friedmanchisquare, on tolerance-based ranks."""
    R = ranks(M); n, k = R.shape
    ties = 0.0
    for r in R:
        _, t = np.unique(r, return_counts=True); ties += float(np.sum(t ** 3 - t))
    c = 1 - ties / (n * k * (k * k - 1))
    chi2 = (12.0 / (n * k * (k + 1)) * np.sum(R.sum(0) ** 2) - 3 * n * (k + 1)) / c
    return chi2, float(stats.chi2.sf(chi2, k - 1))
