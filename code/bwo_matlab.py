"""Line-by-line Python port of the authors' BWO.m (FileExchange 112830, v1.0.2, zip sha256
bed2efb3416d665c0b18cfd0484e6c9c9ba8d8707694188f4c3c61b065183c21).

MATLAB -> Python conventions: 1-based -> 0-based indices; rand -> rng.random; randn -> rng.standard_normal;
randperm(nD) -> rng.permutation(nD); randperm(nD, 2) -> rng.choice(nD, 2, replace=False);
ceil(Npop*rand) -> int(ceil(Npop * rng.random())) - 1. Every statement keeps its MATLAB order, so the
sequence of random draws per iteration follows BWO.m.

Usage as in main.m:
    xposbest, fvalbest, Curve = BWO(Npop, Max_it, lb, ub, nD, fobj)
Optional `frac`: a callable returning T/Max_it; the runner passes the spent-FE fraction and a huge Max_it, the budget wrapper then ends the run.
"""
import math

import numpy as np


def _rj(rng, Npop, i=None):
    """RJ = ceil(Npop*rand); while RJ == i, redraw (i is 0-based here)."""
    RJ = int(math.ceil(Npop * rng.random())) - 1
    while i is not None and RJ == i:
        RJ = int(math.ceil(Npop * rng.random())) - 1
    return max(RJ, 0)          # rand() == 0 exactly would give index -1; MATLAB's rand never returns 0


def BWO(Npop, Max_it, lb, ub, nD, fobj, rng=None, frac=None, ops=None):
    rng = np.random.default_rng() if rng is None else rng
    fit = np.full(Npop, np.inf)
    newfit = fit.copy()
    Curve = np.full(Max_it if Max_it < 10**7 else 0, np.inf)
    Counts_run = 0
    lb = np.broadcast_to(np.asarray(lb, float), (nD,)).copy()
    ub = np.broadcast_to(np.asarray(ub, float), (nD,)).copy()
    pos = rng.random((Npop, nD)) * (ub - lb) + lb
    for i in range(Npop):
        fit[i] = fobj(pos[i]); Counts_run += 1
    index = int(np.argmin(fit)); fvalbest = float(fit[index]); xposbest = pos[index].copy()
    alpha = 3 / 2
    sigma = (math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2)
             / (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)

    T = 1
    while T <= Max_it:
        t = (T / Max_it) if frac is None else frac()
        newpos = pos.copy()
        WF = 0.1 - 0.05 * t                                   # probability of whale fall
        kk = (1 - 0.5 * t) * rng.random(Npop)                 # exploration / exploitation
        for i in range(Npop):
            if kk[i] > 0.5:                                   # exploration phase
                r1 = rng.random(); r2 = rng.random()
                RJ = _rj(rng, Npop, i)
                if nD <= Npop / 5:
                    params = rng.choice(nD, 2, replace=False)
                    newpos[i, params[0]] = pos[i, params[0]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (r1 + 1) * math.sin(r2 * 360)
                    newpos[i, params[1]] = pos[i, params[1]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (r1 + 1) * math.cos(r2 * 360)
                else:
                    params = rng.permutation(nD)
                    for j in range(1, nD // 2 + 1):
                        a, b = params[2 * j - 2], params[2 * j - 1]
                        newpos[i, 2 * j - 2] = pos[i, a] + (pos[RJ, params[0]] - pos[i, a]) * (r1 + 1) * math.sin(r2 * 360)
                        newpos[i, 2 * j - 1] = pos[i, b] + (pos[RJ, params[0]] - pos[i, b]) * (r1 + 1) * math.cos(r2 * 360)
                if ops is not None:
                    ops['explore'] += 1
            else:                                             # exploitation phase
                r3 = rng.random(); r4 = rng.random(); C1 = 2 * r4 * (1 - t)
                RJ = _rj(rng, Npop, i)
                u = rng.standard_normal(nD) * sigma
                v = rng.standard_normal(nD)
                S = u / np.abs(v) ** (1 / alpha)
                KD = 0.05
                LevyFlight = KD * S
                newpos[i] = r3 * xposbest - r4 * pos[i] + C1 * LevyFlight * (pos[RJ] - pos[i])
                if ops is not None:
                    ops['exploit'] += 1
            # boundary
            newpos[i] = np.where(newpos[i] > ub, ub, np.where(newpos[i] < lb, lb, newpos[i]))
            newfit[i] = fobj(newpos[i]); Counts_run += 1
            if newfit[i] < fit[i]:
                pos[i] = newpos[i]; fit[i] = newfit[i]

        for i in range(Npop):                                 # whale falls
            if kk[i] <= WF:
                RJ = _rj(rng, Npop); r5 = rng.random(); r6 = rng.random(); r7 = rng.random()
                C2 = 2 * Npop * WF
                stepsize2 = r7 * (ub - lb) * math.exp(-C2 * t)
                newpos[i] = (r5 * pos[i] - r6 * pos[RJ]) + stepsize2
                newpos[i] = np.where(newpos[i] > ub, ub, np.where(newpos[i] < lb, lb, newpos[i]))
                newfit[i] = fobj(newpos[i]); Counts_run += 1
                if newfit[i] < fit[i]:
                    pos[i] = newpos[i]; fit[i] = newfit[i]
                if ops is not None:
                    ops['fall'] += 1
        index = int(np.argmin(fit)); fval = float(fit[index])
        if fval < fvalbest:
            fvalbest = fval; xposbest = pos[index].copy()
        if T - 1 < len(Curve):
            Curve[T - 1] = fvalbest
        if ops is not None:
            ops['iters'] += 1
        T += 1
    return xposbest, fvalbest, Curve
