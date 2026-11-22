"""Beluga whale optimization arms.

BWO-P   literal Eqs. 3-10 of Zhong, Li and Meng (2022), KBS 251:109215.
BWO-M   port of mealpy master swarm_based/BWO.py, sequential mode.
+E/+X/+W  one invariant fix each on BWO-P; BWO-TI = all three.

Every runner: (obj, lb, ub, dim, seed, params=None) -> obj.best, evaluations only through
budget.Objective, which raises BudgetExhausted at evaluation B+1. Operator counts are written to
obj.ops (dict) so the result rows can show that every phase was reached.
"""
import math

import numpy as np

from budget import BudgetExhausted
from config import BWO, flat

P = flat(BWO)


def _mantegna(rng, size, beta):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
           / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0.0, 1.0, size) * sig / np.abs(rng.normal(0.0, 1.0, size)) ** (1 / beta)


def _box(lb, ub, dim):
    return (np.broadcast_to(np.asarray(lb, float), (dim,)).copy(),
            np.broadcast_to(np.asarray(ub, float), (dim,)).copy())


def _peer(rng, n, i):
    r = int(rng.integers(n - 1))
    return r + (r >= i)


# ------------------------------------------------------------------ phase operators
def explore_paper(rng, X, i, r):
    """Eq. 4, literal: coordinate j receives a copy of coordinate perm_j, pulled toward the single
    scalar X[r, perm_1]; one r1 and one r2 per agent."""
    d = X.shape[1]
    perm = rng.permutation(d)
    r1, r2 = rng.random(), rng.random()
    j1 = np.arange(1, d + 1)
    trig = np.where(j1 % 2 == 0, math.sin(2 * math.pi * r2), math.cos(2 * math.pi * r2))
    src = X[i, perm]
    return src + (X[r, perm[0]] - src) * (1 + r1) * trig


def explore_fixed(rng, X, i, r):
    """+E: coordinates paired along a random permutation; each pair has its own r1, r2; each
    coordinate moves relative to the same coordinate of the peer (no copying, no shared scalar)."""
    d = X.shape[1]
    perm = rng.permutation(d)
    npair = (d + 1) // 2
    r1 = rng.random(npair); r2 = rng.random(npair)
    k = np.arange(d) // 2                       # pair index of position in perm
    second = (np.arange(d) % 2 == 1)
    trig = np.where(second, np.cos(2 * np.pi * r2[k]), np.sin(2 * np.pi * r2[k]))
    coef = np.empty(d)
    coef[perm] = (1 + r1[k]) * trig
    return X[i] + (X[r] - X[i]) * coef


def exploit_paper(rng, X, i, r, best, tau):
    """Eq. 5: r3 X_best - r4 X_i + C1 LF (X_r - X_i), C1 = 2 r4 (1 - tau)."""
    d = X.shape[1]
    r3, r4 = rng.random(), rng.random()
    C1 = 2 * r4 * (1 - tau)
    LF = P['levy_scale'] * _mantegna(rng, d, P['levy_beta'])
    return r3 * best - r4 * X[i] + C1 * LF * (X[r] - X[i])


def exploit_fixed(rng, X, i, r, best, tau):
    """+X: X_i + r3 (X_best - X_i) + C1 LF (X_r - X_i), r3 per coordinate."""
    d = X.shape[1]
    r3 = rng.random(d); r4 = rng.random()
    C1 = 2 * r4 * (1 - tau)
    LF = P['levy_scale'] * _mantegna(rng, d, P['levy_beta'])
    return X[i] + r3 * (best - X[i]) + C1 * LF * (X[r] - X[i])


def fall_paper(rng, X, i, r, step):
    """Eq. 8: r5 X_i - r6 X_r + r7 X_step (scalars)."""
    r5, r6, r7 = rng.random(), rng.random(), rng.random()
    return r5 * X[i] - r6 * X[r] + r7 * step


def fall_fixed(rng, X, i, r, step):
    """+W: X_i + r6 (X_i - X_r) + (2 r7 - 1) X_step, per coordinate."""
    d = X.shape[1]
    r6 = rng.random(d); r7 = rng.random(d)
    return X[i] + r6 * (X[i] - X[r]) + (2 * r7 - 1) * step


# ------------------------------------------------------------------ BWO-P family
def _bwo_family(obj, lb, ub, dim, seed, fix_e=False, fix_x=False, fix_w=False, clip=True):
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim)
    n = P['n']; B = obj.max_fe
    ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    obj.ops = ops
    explore = explore_fixed if fix_e else explore_paper
    exploit = exploit_fixed if fix_x else exploit_paper
    fall = fall_fixed if fix_w else fall_paper
    bound = (lambda x: np.clip(x, lo, hi)) if clip else (lambda x: x)

    X = lo + rng.random((n, dim)) * (hi - lo)
    fit = np.array([obj(x) for x in X])
    best = X[int(np.argmin(fit))].copy(); fbest = float(fit.min())
    while True:
        tau = obj.n / B
        Bf = rng.random(n) * (1 - tau / 2)                      # Eq. 3
        cand = np.empty_like(X)
        for i in range(n):                                       # batch from the population at iteration start
            r = _peer(rng, n, i)
            if Bf[i] > P['bf_threshold']:
                cand[i] = explore(rng, X, i, r); ops['explore'] += 1
            else:
                cand[i] = exploit(rng, X, i, r, best, tau); ops['exploit'] += 1
        for i in range(n):
            c = bound(cand[i]); fc = obj(c)
            if fc < fit[i]:
                X[i] = c; fit[i] = fc
                if fc < fbest:
                    best = c.copy(); fbest = fc
        tau = obj.n / B
        Wf = P['wf_start'] - (P['wf_start'] - P['wf_end']) * tau  # Eq. 10
        C2 = 2 * Wf * n
        step = (hi - lo) * math.exp(-C2 * tau)                   # Eq. 9
        for i in range(n):
            if Bf[i] < Wf:                                       # Fig. 2
                r = _peer(rng, n, i)
                c = bound(fall(rng, X, i, r, step)); fc = obj(c); ops['fall'] += 1
                if fc < fit[i]:
                    X[i] = c; fit[i] = fc
                    if fc < fbest:
                        best = c.copy(); fbest = fc
        ops['iters'] += 1


def _guard(fn):
    def wrapped(obj, lb, ub, dim, seed, params=None):
        try:
            fn(obj, lb, ub, dim, seed, **(params or {}))
        except BudgetExhausted:
            pass
        return obj.best
    wrapped.__name__ = getattr(fn, '__name__', 'bwo')
    return wrapped


def _arm(**kw):
    def run(obj, lb, ub, dim, seed, **extra):
        return _bwo_family(obj, lb, ub, dim, seed, **{**kw, **extra})
    return _guard(run)


# ------------------------------------------------------------------ BWO-M (mealpy master port)
@_guard
def run_bwo_mealpy(obj, lb, ub, dim, seed):
    """Sequential-mode logic of mealpy master BWO.py: exploration uses the agent's OWN coordinates
    (pos_ii[p1], not X_r), pj and p1 drawn with replacement, trig by index j = 1..d (even -> sin);
    exploitation as Eq. 5; immediate greedy update; whale fall triggered by rand < Wf for every agent,
    evaluated and accepted immediately. Schedule uses the spent FE fraction (budget rule)."""
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim)
    n = P['n']; B = obj.max_fe
    ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    obj.ops = ops
    X = lo + rng.random((n, dim)) * (hi - lo)
    fit = np.array([obj(x) for x in X])
    g = int(np.argmin(fit)); best = X[g].copy(); fbest = float(fit[g])
    j1 = np.arange(1, dim + 1)
    while True:
        tau = obj.n / B
        Bf = rng.random(n) * (1 - tau / 2)
        for i in range(n):
            r = _peer(rng, n, i)
            if Bf[i] > P['bf_threshold']:
                r1, r2 = rng.random(2)
                pj = rng.integers(0, dim, size=dim); p1 = rng.integers(0, dim, size=dim)
                base = X[i, pj]; diff = X[i, p1] - base
                trig = np.where(j1 % 2 == 0, math.sin(2 * math.pi * r2), math.cos(2 * math.pi * r2))
                c = base + diff * (1 + r1) * trig; ops['explore'] += 1
            else:
                r3, r4 = rng.random(2)
                C1 = 2 * r4 * (1 - tau)
                LF = P['levy_scale'] * _mantegna(rng, dim, P['levy_beta'])
                c = r3 * best - r4 * X[i] + C1 * LF * (X[r] - X[i]); ops['exploit'] += 1
            c = np.clip(c, lo, hi); fc = obj(c)
            if fc < fit[i]:
                X[i] = c; fit[i] = fc
                if fc < fbest:
                    best = c.copy(); fbest = fc
        tau = obj.n / B
        Wf = P['wf_start'] - (P['wf_start'] - P['wf_end']) * tau
        step = (hi - lo) * math.exp(-2 * Wf * n * tau)
        for i in range(n):
            if rng.random() < Wf:
                r = _peer(rng, n, i)
                r5, r6, r7 = rng.random(3)
                c = np.clip(r5 * X[i] - r6 * X[r] + r7 * step, lo, hi); fc = obj(c); ops['fall'] += 1
                if fc < fit[i]:
                    X[i] = c; fit[i] = fc
                    if fc < fbest:
                        best = c.copy(); fbest = fc
        ops['iters'] += 1


# ------------------------------------------------------------------ BWO-ML (authors' MATLAB, literal port)
@_guard
def run_bwo_matlab(obj, lb, ub, dim, seed):
    """Authors' BWO.m through the line-by-line port bwo_matlab.BWO (n = 50 as main.m). The only change is
    the schedule: T/Max_it is the spent FE fraction at the start of each iteration (budget rule); the
    budget wrapper ends the run."""
    from bwo_matlab import BWO as _BWO
    ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    obj.ops = ops
    B = obj.max_fe
    _BWO(P['n'], 10**9, lb, ub, dim, obj, rng=np.random.default_rng(seed), frac=lambda: obj.n / B, ops=ops)


RUNNERS = {
    'BWO-P': _arm(),
    'BWO-ML': run_bwo_matlab,
    'BWO-M': run_bwo_mealpy,
    'BWO-P+E': _arm(fix_e=True),
    'BWO-P+X': _arm(fix_x=True),
    'BWO-P+W': _arm(fix_w=True),
    'BWO-TI': _arm(fix_e=True, fix_x=True, fix_w=True),
}
