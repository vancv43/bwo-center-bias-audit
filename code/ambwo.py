"""AMBWO: You et al. (2024), "AMBWO: An augmented multi-strategy beluga whale optimization for numerical
optimization problems", Biomimetics 9(12):727, doi 10.3390/biomimetics9120727 (PMC11674027, full text and
MathML version of the article). No author code was found, so this is a port of the printed
equations and Algorithm 1; every choice the paper leaves open is tagged 'assumption' in config.AMBWO.

Per iteration (Algorithm 1), with tau = spent FE / B in place of t/T (budget rule):
  Bf_i = B0_i (1 - tau/2)  (BWO Eq. 8);  Wf = 0.1 - 0.05 tau  (Eq. 12)
  E_factor = (E_max - E_min) e^tau - E_min,  E_min = 1.1, E_max = 2.5  (Eq. 13, as printed)
  CF = (1 - tau)^(2 - tau)  (Eq. 22, as printed)
  S_i = a1 (Dis_i - Dis_min)/Dis_max + (1 - a1)(Fit_i - Fit_min)/Fit_max, a1 = 0.5  (Eq. 19)
  for each agent i, sequentially:
    Bf_i > 0.5, rand > 0.5: X_i = N(X_mean, Cov) + r9 (X_mean - X_i)          (APLS, Eqs. 15-18)
    Bf_i > 0.5, else:      X_ij = X_i,pj + E_factor (r8 X_r,p1 - X_i,pj)(1 + r1) sin|cos(2 pi r2)   (Eq. 14)
    Bf_i <= 0.5:           X_i = r3 X_ref - r4 X_i + C1 LF (X_r - X_i), X_ref by roulette on S   (RESS, Eq. 20)
    Bf_i < Wf:             whale fall, BWO Eq. 9
    always:                AAS, Eq. 21 (rand <= 0.34: X_i + CF (LB + r10 (UB - LB)) U; else
                           X_i + (0.34 (1 - r11) + r11)(X_r1 - X_r2))
Every candidate is clipped, evaluated through the budget wrapper and accepted greedily.
"""
import math

import numpy as np

from budget import BudgetExhausted
from config import AMBWO, flat

P = flat(AMBWO)


def _mantegna(rng, size, beta):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
           / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0.0, 1.0, size) * sig / np.abs(rng.normal(0.0, 1.0, size)) ** (1 / beta)


# Calibration readings. Default = the reading used in R1 and all comparisons (all 'printed'/'prop').
READING0 = {'ef': 'printed', 'cf': 'printed', 'roulette': 'prop'}


def e_factor(tau, mode='printed'):
    """Eq. 13. 'printed': (E_max - E_min) e^tau - E_min, rising from 0.3 to 2.706 although the text calls it a
    'nonlinear decrease'. 'decreasing' (alternative reading): E_min + (E_max - E_min) e^-tau, falling from
    2.5 to 1.615 (sign of the exponent and of E_min flipped so the factor decreases and stays above E_min)."""
    if mode == 'printed':
        return (P['e_max'] - P['e_min']) * math.exp(tau) - P['e_min']
    assert mode == 'decreasing', mode
    return P['e_min'] + (P['e_max'] - P['e_min']) * math.exp(-tau)


def cf(tau, mode='printed'):
    """Eq. 22. 'printed': (1 - t/T)^(2 - t/T). 'gazelle' (alternative reading): (1 - t/T)^(2 t/T), the source rule of
    the Gazelle optimisation algorithm from which AAS is taken."""
    if mode == 'printed':
        return (1.0 - tau) ** (2.0 - tau)
    assert mode == 'gazelle', mode
    return (1.0 - tau) ** (2.0 * tau)


def scores(X, fit, best):
    """Eq. 19. Distances to the best agent and fitness, each shifted by its minimum and divided by its maximum."""
    dis = np.linalg.norm(X - best, axis=1)
    dmax = dis.max(); fmax = fit.max()
    a = P['alpha1']
    s_d = (dis - dis.min()) / dmax if dmax > 0 else np.zeros_like(dis)
    s_f = (fit - fit.min()) / fmax if fmax != 0 else np.zeros_like(fit)
    return a * s_d + (1 - a) * s_f


def roulette(rng, S, mode='prop'):
    """'prop': probability proportional to S (paper: 'roulette wheel principle'); uniform if all scores are 0.
    Negative scores (possible when Fit_max < 0) are shifted to start at 0 (assumption).
    'inverse' (alternative reading): probability proportional to max(S) - S, favouring agents near the best with low
    fitness (S is small for those)."""
    if mode == 'inverse':
        S = float(S.max()) - S
    else:
        assert mode == 'prop', mode
    w = S - min(0.0, float(S.min()))
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return int(rng.integers(len(S)))
    return int(rng.choice(len(S), p=w / tot))


def mean_cov(X, fit):
    """Eqs. 15-17: log-weighted mean of the better half, and its sample covariance (divisor 0.5N - 1)."""
    h = len(X) // 2
    idx = np.argsort(fit, kind='stable')[:h]
    i = np.arange(1, h + 1)
    w = np.log(0.5 * len(X) + 1) - np.log(i)
    w = w / w.sum()
    top = X[idx]
    xm = w @ top
    Z = top - xm
    cov = Z.T @ Z / (h - 1)
    return xm, cov


def explore_eq14(rng, X, i, r, ef):
    d = X.shape[1]
    perm = rng.permutation(d)
    r1, r2, r8 = rng.random(), rng.random(), rng.random()
    j1 = np.arange(1, d + 1)
    trig = np.where(j1 % 2 == 0, math.sin(2 * math.pi * r2), math.cos(2 * math.pi * r2))
    src = X[i, perm]
    return src + ef * (r8 * X[r, perm[0]] - src) * (1 + r1) * trig


def gauss_eq18(rng, X, i, xm, cov):
    r9 = rng.random()
    g = rng.multivariate_normal(xm, cov, method='eigh', check_valid='ignore')
    return g + r9 * (xm - X[i])


def exploit_eq20(rng, X, i, r, ref, tau):
    d = X.shape[1]
    r3, r4 = rng.random(), rng.random()
    C1 = 2 * r4 * (1 - tau)
    LF = P['levy_scale'] * _mantegna(rng, d, P['levy_beta'])
    return r3 * ref - r4 * X[i] + C1 * LF * (X[r] - X[i])


def fall_eq9(rng, X, i, r, step):
    r5, r6, r7 = rng.random(), rng.random(), rng.random()
    return r5 * X[i] - r6 * X[r] + r7 * step


def avoid_eq21(rng, X, i, lo, hi, c):
    n, d = X.shape
    if rng.random() <= P['psr']:
        U = (rng.random(d) < P['psr']).astype(float)
        return X[i] + c * (lo + rng.random(d) * (hi - lo)) * U
    r11 = rng.random()
    a, b = rng.choice(n, 2, replace=False)
    return X[i] + (P['psr'] * (1 - r11) + r11) * (X[a] - X[b])


def _peer(rng, n, i):
    r = int(rng.integers(n - 1))
    return r + (r >= i)


def _ambwo(obj, lb, ub, dim, seed, iters=None, reading=None):
    R = {**READING0, **(reading or {})}
    assert set(R) == set(READING0), R
    rng = np.random.default_rng(seed)
    lo = np.broadcast_to(np.asarray(lb, float), (dim,)).copy()
    hi = np.broadcast_to(np.asarray(ub, float), (dim,)).copy()
    n = P['n']; B = obj.max_fe
    ops = {'gauss': 0, 'explore': 0, 'exploit': 0, 'fall': 0, 'avoid': 0, 'iters': 0}
    obj.ops = ops
    obj.reading = R
    X = lo + rng.random((n, dim)) * (hi - lo)
    fit = np.array([obj(x) for x in X])
    g = int(np.argmin(fit)); best = X[g].copy(); fbest = float(fit[g])

    def accept(i, c):
        nonlocal best, fbest
        c = np.clip(c, lo, hi); fc = obj(c)
        if fc < fit[i]:
            X[i] = c; fit[i] = fc
            if fc < fbest:
                best = c.copy(); fbest = fc

    t = 0
    while iters is None or t < iters:
        tau = obj.n / B if iters is None else t / iters     # iters: paper mode t/T (replication check only)
        Bf = rng.random(n) * (1 - tau / 2)
        Wf = P['wf_start'] - (P['wf_start'] - P['wf_end']) * tau
        ef = e_factor(tau, R['ef']); c = cf(tau, R['cf'])
        S = scores(X, fit, best)
        xm, cov = mean_cov(X, fit)
        step = (hi - lo) * math.exp(-2 * Wf * n * tau)
        for i in range(n):
            r = _peer(rng, n, i)
            if Bf[i] > P['bf_threshold']:
                if rng.random() > 0.5:
                    cand = gauss_eq18(rng, X, i, xm, cov); ops['gauss'] += 1
                else:
                    cand = explore_eq14(rng, X, i, r, ef); ops['explore'] += 1
            else:
                ref = X[roulette(rng, S, R['roulette'])].copy()
                cand = exploit_eq20(rng, X, i, r, ref, tau); ops['exploit'] += 1
            accept(i, cand)
            if Bf[i] < Wf:
                accept(i, fall_eq9(rng, X, i, _peer(rng, n, i), step)); ops['fall'] += 1
            accept(i, avoid_eq21(rng, X, i, lo, hi, c)); ops['avoid'] += 1
        ops['iters'] += 1
        t += 1


def run_ambwo(obj, lb, ub, dim, seed, params=None):
    try:
        _ambwo(obj, lb, ub, dim, seed, **(params or {}))
    except BudgetExhausted:
        pass
    return obj.best
