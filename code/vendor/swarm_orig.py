"""Swarm baselines of the paper (PSO, GWO, WOA, HHO, MPA), written from the original publications.

The published equations are primary; where an article leaves a detail open, the structure of the authors'
public code is followed. Every runner has the signature (obj, lb, ub, dim, seed, params=None), evaluates only
through `obj` (budget.Objective, which raises BudgetExhausted at evaluation B+1) and returns obj.best.
Iteration-scheduled parameters use the horizon T implied by the budget. Out-of-range coordinates are
clipped to the box unless the original prescribes otherwise.
"""
import io
import math
import os
import random
import sys
import contextlib

import numpy as np

from budget import BudgetExhausted

HERE = os.path.dirname(os.path.abspath(__file__))
POP = 30


def _guard(fn):
    def wrapped(obj, lb, ub, dim, seed, params=None):
        try:
            with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
                fn(obj, lb, ub, dim, seed, params)
        except BudgetExhausted:
            pass
        return obj.best
    wrapped.__name__ = fn.__name__
    return wrapped


def _box(lb, ub, dim):
    return (np.broadcast_to(np.asarray(lb, float), (dim,)).copy(),
            np.broadcast_to(np.asarray(ub, float), (dim,)).copy())


def _mantegna(rng, size, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
           / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, sig, size) / np.abs(rng.normal(0, 1, size)) ** (1 / beta)


def _iters(B, n_init, per_iter):
    return max(2, int(math.ceil((B - n_init) / per_iter)))


# =============================================================================================
# 1  PSO: tuned global-best inertia PSO (Shi and Eberhart 1998), verbatim
# =============================================================================================
def _reflect(x, lb, ub):
    span = ub - lb; y = (x - lb) % (2 * span); y = np.where(y <= span, y, 2 * span - y); return lb + y


@_guard
def run_pso(obj, lb, ub, dim, seed, params=None):
    import json
    p = {'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'N': 30,
         **json.load(open(os.path.join(HERE, 'tuned_configs.json')))['PSO']}
    rng = np.random.default_rng(seed); n = int(p['N']); lo, hi = _box(lb, ub, dim)
    vmax = 0.2 * (hi - lo)
    X = rng.uniform(lo, hi, (n, dim)); V = rng.uniform(-vmax, vmax, (n, dim))
    fit = np.array([obj(x) for x in X]); P = X.copy(); Pf = fit.copy(); g = int(np.argmin(Pf))
    while True:
        for i in range(n):
            r1 = rng.random(dim); r2 = rng.random(dim)
            V[i] = np.clip(p['w'] * V[i] + p['c1'] * r1 * (P[i] - X[i]) + p['c2'] * r2 * (P[g] - X[i]), -vmax, vmax)
            X[i] = _reflect(X[i] + V[i], lo, hi); fx = obj(X[i])
            if fx < Pf[i]:
                P[i] = X[i].copy(); Pf[i] = fx
                if fx < Pf[g]:
                    g = i


# =============================================================================================
# 4, 5  WOA (Mirjalili and Lewis 2016): EvoloPy WOA.py, verified line by line against the authors' WOA.m,
#        used verbatim (third_party/evolopy, Apache-2.0). GWO (Mirjalili et al. 2014): literal port of GWO.m.
# =============================================================================================
def _evolopy(name):
    @_guard
    def run(obj, lb, ub, dim, seed, params=None):
        sys.path.insert(0, os.path.join(HERE, 'third_party'))
        import importlib
        fn = getattr(importlib.import_module(f'evolopy.{name}'), name)
        random.seed(int(seed)); np.random.seed(int(seed) % (2 ** 32))
        lo, hi = _box(lb, ub, dim)

        def objf(x):
            return obj(x)
        objf.__name__ = 'objective'
        fn(objf, lo.tolist(), hi.tolist(), dim, POP, _iters(obj.max_fe, 0, POP))
    run.__name__ = f'run_{name.lower()}'
    return run


run_woa = _evolopy('WOA')


@_guard
def run_gwo(obj, lb, ub, dim, seed, params=None):
    """Literal port of the authors' GWO.m (seyedalimirjalili.com, GWO.zip): leaders replaced without
    the alpha -> beta -> delta cascade that EvoloPy adds; per-dimension r1, r2."""
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, N)
    al_p = np.zeros(dim); be_p = np.zeros(dim); de_p = np.zeros(dim); al_s = be_s = de_s = np.inf
    X = rng.uniform(lo, hi, (N, dim))
    for l in range(T):
        for i in range(N):
            X[i] = np.clip(X[i], lo, hi); f = obj(X[i])
            if f < al_s:
                al_s, al_p = f, X[i].copy()
            if al_s < f < be_s:
                be_s, be_p = f, X[i].copy()
            if f > al_s and f > be_s and f < de_s:
                de_s, de_p = f, X[i].copy()
        a = 2 - l * (2 / T)
        for i in range(N):
            for j in range(dim):
                A1 = 2 * a * rng.random() - a; C1 = 2 * rng.random()
                X1 = al_p[j] - A1 * abs(C1 * al_p[j] - X[i, j])
                A2 = 2 * a * rng.random() - a; C2 = 2 * rng.random()
                X2 = be_p[j] - A2 * abs(C2 * be_p[j] - X[i, j])
                A3 = 2 * a * rng.random() - a; C3 = 2 * rng.random()
                X3 = de_p[j] - A3 * abs(C3 * de_p[j] - X[i, j])
                X[i, j] = (X1 + X2 + X3) / 3


# =============================================================================================
# 6  HHO (Heidari et al. 2019, FGCS 97:849), Eqs. 1-13; acceptance in Eqs. 10-11 compares with
#    F(X_i(t)); S ~ U(0,1)^D and the 0.01-scaled Levy flight of Eq. 9 as in HHO.m.
#    Evaluations per iteration vary, so t/T is the spent budget fraction.
# =============================================================================================
@_guard
def run_hho(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP; B = obj.max_fe
    X = rng.uniform(lo, hi, (N, dim)); rab = None; frab = np.inf
    while True:
        X = np.clip(X, lo, hi)
        fx = np.empty(N)
        for i in range(N):
            fx[i] = obj(X[i])
            if fx[i] < frab:
                frab = fx[i]; rab = X[i].copy()
        E1 = 2 * (1 - obj.n / B)
        Xm = X.mean(0)                                                   # Eq. 2
        Xold = X.copy()
        for i in range(N):
            E = E1 * (2 * rng.random() - 1)                              # Eq. 3
            if abs(E) >= 1:                                              # exploration, Eq. 1
                if rng.random() >= 0.5:
                    Xr = Xold[rng.integers(N)]
                    X[i] = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * Xold[i])
                else:
                    X[i] = (rab - Xm) - rng.random() * (lo + rng.random() * (hi - lo))
                continue
            r = rng.random(); J = 2 * (1 - rng.random())
            if r >= 0.5 and abs(E) >= 0.5:                               # soft besiege, Eq. 4
                X[i] = (rab - Xold[i]) - E * np.abs(J * rab - Xold[i])
            elif r >= 0.5:                                               # hard besiege, Eq. 6
                X[i] = rab - E * np.abs(rab - Xold[i])
            else:                                                        # progressive dives, Eqs. 7-13
                ref = Xold[i] if abs(E) >= 0.5 else Xm
                Y = np.clip(rab - E * np.abs(J * rab - ref), lo, hi)
                if obj(Y) < fx[i]:
                    X[i] = Y; continue
                Z = np.clip(Y + rng.random(dim) * 0.01 * _mantegna(rng, dim), lo, hi)
                if obj(Z) < fx[i]:
                    X[i] = Z


# =============================================================================================
# 17 MPA (Faramarzi et al. 2020, ESWA 152:113377), Eqs. 10-16 as in MPA.m: three phases by
#    iteration, Brownian/Levy steps (RL = 0.05 Levy), P = 0.5, FADs = 0.2, marine memory after each
#    evaluation, two evaluations per iteration
# =============================================================================================
@_guard
def run_mpa(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, 2 * N); P, FADs = 0.5, 0.2
    Prey = rng.uniform(lo, hi, (N, dim)); top, ftop = None, np.inf
    fold = Pold = None

    def evaluate(Prey, it):
        nonlocal top, ftop, fold, Pold
        Prey = np.clip(Prey, lo, hi); fit = np.empty(N)
        for i in range(N):
            fit[i] = obj(Prey[i])
            if fit[i] < ftop:
                ftop, top = fit[i], Prey[i].copy()
        if it == 0:                                                      # as MPA.m, both blocks
            fold, Pold = fit.copy(), Prey.copy()
        keep = fold < fit                                                # marine memory
        Prey = np.where(keep[:, None], Pold, Prey); fit = np.where(keep, fold, fit)
        fold, Pold = fit.copy(), Prey.copy()
        return Prey
    for it in range(T):
        Prey = evaluate(Prey, it)
        Elite = np.tile(top, (N, 1))
        CF = (1 - it / T) ** (2 * it / T)
        RL = 0.05 * _mantegna(rng, (N, dim)); RB = rng.standard_normal((N, dim))
        R = rng.random((N, dim))
        if it < T / 3:                                                   # Eq. 12
            Prey = Prey + P * R * (RB * (Elite - RB * Prey))
        elif T / 3 < it < 2 * T / 3:                                     # Eqs. 13-14
            half = np.arange(N) + 1 > N / 2
            levy = Prey + P * R * (RL * (Elite - RL * Prey))
            brown = Elite + P * CF * (RB * (RB * Elite - Prey))
            Prey = np.where(half[:, None], brown, levy)
        else:                                                            # Eq. 15
            Prey = Elite + P * CF * (RL * (RL * Elite - Prey))
        Prey = evaluate(Prey, it)
        if rng.random() < FADs:                                          # Eq. 16
            U = rng.random((N, dim)) < FADs
            Prey = Prey + CF * ((lo + rng.random((N, dim)) * (hi - lo)) * U)
        else:
            r = rng.random()
            Prey = Prey + (FADs * (1 - r) + r) * (Prey[rng.permutation(N)] - Prey[rng.permutation(N)])


RUNNERS = {'PSO': run_pso, 'GWO': run_gwo, 'WOA': run_woa, 'HHO': run_hho, 'MPA': run_mpa}
