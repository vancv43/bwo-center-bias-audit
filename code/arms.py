"""All arms of behind one signature (obj, lb, ub, dim, seed, params=None) -> best."""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))

from budget import BudgetExhausted          # noqa: E402
import baselines as _bl                     # noqa: E402  (audited: DE, L-SHADE, jSO)
import swarm_orig as _sw                    # noqa: E402  (audited vs authors' code: PSO, GWO, WOA, HHO, MPA)
from bwo import RUNNERS as _BWO             # noqa: E402
from ambwo import run_ambwo as _ambwo        # noqa: E402  (You et al. 2024)


def _guard(fn):
    def wrapped(obj, lb, ub, dim, seed, params=None):
        try:
            fn(obj, lb, ub, dim, seed)
        except BudgetExhausted:
            pass
        return obj.best
    wrapped.__name__ = fn.__name__
    return wrapped


@_guard
def run_rs(obj, lb, ub, dim, seed):
    rng = np.random.default_rng(seed)
    lo = np.broadcast_to(np.asarray(lb, float), (dim,)); hi = np.broadcast_to(np.asarray(ub, float), (dim,))
    while True:
        for _ in range(1000):
            obj(lo + rng.random(dim) * (hi - lo))


@_guard
def run_bipop(obj, lb, ub, dim, seed):
    """BIPOP-CMA-ES as in an earlier benchmark study: pycma 4.4.4, sigma0 = 0.3 of the range, 9 restarts, bipop=True."""
    import cma
    rng = np.random.default_rng(seed)
    lo = np.broadcast_to(np.asarray(lb, float), (dim,)); hi = np.broadcast_to(np.asarray(ub, float), (dim,))
    x0 = lo + rng.random(dim) * (hi - lo)
    opts = {'bounds': [lo.tolist(), hi.tolist()], 'seed': int(rng.integers(1, 2**31 - 1)), 'verbose': -9,
            'maxfevals': obj.max_fe, 'tolflatfitness': 10, 'CMA_stds': None}
    with contextlib.redirect_stdout(io.StringIO()):
        cma.fmin2(obj, x0, 0.3 * float(np.mean(hi - lo)), opts, restarts=9, bipop=True)


import json as _json  # noqa: E402
_TUNED = _json.load(open(os.path.join(HERE, 'vendor', 'tuned_configs.json')))


def _run_de(obj, lb, ub, dim, seed, params=None):
    """DE/rand/1/bin with tuned values from an earlier benchmark study (F, CR, N)."""
    return _bl.run_de(obj, lb, ub, dim, seed, _TUNED['DE'])


ARMS = {
    **_BWO,
    'RS': run_rs,
    'DE': _run_de, 'L-SHADE': _bl.run_lshade, 'jSO': _bl.run_jso,
    'PSO': _sw.run_pso, 'GWO': _sw.run_gwo, 'WOA': _sw.run_woa, 'HHO': _sw.run_hho, 'MPA': _sw.run_mpa,
    'BIPOP-CMA-ES': run_bipop,
    'AMBWO': _ambwo,
}
BWO_ARMS = list(_BWO)
E1_ARMS = list(ARMS)                              # AMBWO was run as a separate pass
E2_ARMS = [a for a in ARMS if a != 'RS']
# E3 list + AMBWO; order fixed before any E3 run
E3_ARMS = ['BWO-P', 'BWO-M', 'BWO-TI', 'AMBWO', 'L-SHADE', 'jSO', 'BIPOP-CMA-ES', 'DE', 'PSO', 'GWO', 'WOA', 'HHO',
           'MPA']


# ---------------------------------------------------------------- R1 replication check: paper settings
def _ambwo_paper(obj, lb, ub, dim, seed):
    """AMBWO exactly as You et al. 2024 ran it: N = 30, T = 500 iterations, t/T schedule (no FE matching)."""
    return _ambwo(obj, lb, ub, dim, seed, params={'iters': 500})


def _bwo_ml_paper(obj, lb, ub, dim, seed):
    """Authors' BWO.m with N = 30, Max_it = 500, T/Max_it schedule: the BWO column of You et al. 2024."""
    from bwo_matlab import BWO as _B
    obj.ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    try:
        _B(30, 500, lb, ub, dim, obj, rng=np.random.default_rng(seed), ops=obj.ops)
    except BudgetExhausted:
        pass
    return obj.best


REP_ARMS = {'AMBWO@paper': _ambwo_paper, 'BWO-ML@paper': _bwo_ml_paper}


# ---------------------------------------------------------------- C1 AMBWO calibration: paper settings
# 2^3 readings of the three open points with the largest effect on the search: Eq. 13 direction (ef), Eq. 22
# exponent (cf), RESS roulette direction (roulette). R0 = the reading of R1 and all comparisons. Order fixed before any C1 run.
CAL_READINGS = {f'AMBWO@R{k}': {'ef': ef, 'cf': cf, 'roulette': ro}
                for k, (ef, cf, ro) in enumerate((ef, cf, ro) for ef in ('printed', 'decreasing')
                                                 for cf in ('printed', 'gazelle') for ro in ('prop', 'inverse'))}


def _cal(reading):
    def f(obj, lb, ub, dim, seed):
        return _ambwo(obj, lb, ub, dim, seed, params={'iters': 500, 'reading': reading})
    return f


CAL_ARMS = {k: _cal(v) for k, v in CAL_READINGS.items()}
