"""Unit tests: run with  python -m pytest -q tests/test_g0.py  from code/."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'vendor'))

from budget import Objective                     # noqa: E402
import bwo                                       # noqa: E402
from arms import ARMS, BWO_ARMS                  # noqa: E402
import problems as PR                            # noqa: E402

SHIFT10 = np.random.default_rng(0).uniform(-80, 80, 10)


def sph(x):
    return float(np.sum((np.asarray(x) - SHIFT10[:len(x)]) ** 2))


class Recorder:
    def __init__(self, f, lb, ub):
        self.f, self.lb, self.ub, self.outside = f, lb, ub, 0

    def __call__(self, x):
        x = np.asarray(x, float)
        if np.any(x < self.lb - 1e-12) or np.any(x > self.ub + 1e-12):
            self.outside += 1
        return self.f(x)


# ---------------------------------------------------------------- budget, determinism, box
@pytest.mark.parametrize('arm', list(ARMS))
def test_budget_determinism_box(arm):
    B = 3000
    rec = Recorder(sph, -100., 100.)
    o1 = Objective(rec, 10, B); b1 = ARMS[arm](o1, -100., 100., 10, 7)
    o2 = Objective(sph, 10, B); b2 = ARMS[arm](o2, -100., 100., 10, 7)
    if arm == 'BIPOP-CMA-ES':
        assert 0 < o1.n <= B           # pycma may stop before maxfevals; unused tail is counted as used
    else:
        assert o1.n == B, (arm, o1.n)
    assert b1 == b2, (arm, b1, b2)
    assert rec.outside == 0, (arm, rec.outside)
    assert np.isfinite(b1)


# ---------------------------------------------------------------- operator reach at the real E1 budget
@pytest.mark.parametrize('arm', BWO_ARMS)
def test_operator_reach(arm):
    f, lb, ub, dim, fopt, _ = PR.classical('F1', 'Sr', 0)
    o = Objective(f, dim, 50_000, fopt)
    ARMS[arm](o, lb, ub, dim, 0)
    ops = o.ops
    assert o.n == 50_000
    assert ops['explore'] > 0 and ops['exploit'] > 0 and ops['fall'] > 0, ops
    # Bf ~ U(0, 1 - tau/2): exploration share must fall over the run and stay below ~0.5
    assert ops['explore'] < ops['exploit'], ops


# ---------------------------------------------------------------- equations on hand-set inputs
class FixedRng:
    """Replays a scripted sequence of draws so each operator can be checked by hand."""
    def __init__(self, perm=None, rand=None, vec=None, normal=None):
        self._perm, self._rand, self._vec, self._normal = perm, list(rand or []), list(vec or []), list(normal or [])

    def permutation(self, d):
        return np.array(self._perm)

    def random(self, size=None):
        if size is None:
            return self._rand.pop(0)
        return np.array(self._vec.pop(0), float)

    def normal(self, mu, sd, size):
        return np.array(self._normal.pop(0), float) * sd + mu


X = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [-1.0, -2.0, -3.0, -4.0]])


def test_explore_paper_literal():
    rng = FixedRng(perm=[2, 0, 3, 1], rand=[0.25, 0.125])
    out = bwo.explore_paper(rng, X, 0, 1)
    r1, r2 = 0.25, 0.125
    perm = [2, 0, 3, 1]
    exp = []
    for j in range(4):                                    # j 1-based even -> sin
        trig = math.sin(2 * math.pi * r2) if (j + 1) % 2 == 0 else math.cos(2 * math.pi * r2)
        src = X[0, perm[j]]
        exp.append(src + (X[1, perm[0]] - src) * (1 + r1) * trig)   # one shared scalar X[r, perm_1]
    assert np.allclose(out, exp)


def test_explore_fixed():
    rng = FixedRng(perm=[2, 0, 3, 1], vec=[[0.5, 0.25], [0.1, 0.3]])
    out = bwo.explore_fixed(rng, X, 0, 1)
    coef = np.empty(4)
    coef[2] = 1.5 * math.sin(2 * math.pi * 0.1); coef[0] = 1.5 * math.cos(2 * math.pi * 0.1)
    coef[3] = 1.25 * math.sin(2 * math.pi * 0.3); coef[1] = 1.25 * math.cos(2 * math.pi * 0.3)
    assert np.allclose(out, X[0] + (X[1] - X[0]) * coef)


def test_exploit_paper_and_fixed():
    best = np.array([5.0, 5.0, 5.0, 5.0]); tau = 0.3
    u = [[1.0, -1.0, 0.5, 2.0]]; v = [[1.0, 1.0, 1.0, 1.0]]   # Mantegna numerator, denominator
    sig = (math.gamma(2.5) * math.sin(math.pi * 0.75) / (math.gamma(1.25) * 1.5 * 2 ** 0.25)) ** (1 / 1.5)
    LF = 0.05 * np.array(u[0]) * sig
    rng = FixedRng(rand=[0.6, 0.4], normal=[u[0], v[0]])
    out = bwo.exploit_paper(rng, X, 0, 2, best, tau)
    C1 = 2 * 0.4 * (1 - tau)
    assert np.allclose(out, 0.6 * best - 0.4 * X[0] + C1 * LF * (X[2] - X[0]))
    rng = FixedRng(rand=[0.4], vec=[[0.1, 0.2, 0.3, 0.4]], normal=[u[0], v[0]])
    out = bwo.exploit_fixed(rng, X, 0, 2, best, tau)
    assert np.allclose(out, X[0] + np.array([0.1, 0.2, 0.3, 0.4]) * (best - X[0]) + C1 * LF * (X[2] - X[0]))


def test_fall_paper_and_fixed():
    step = np.array([0.5, 0.5, 0.5, 0.5])
    rng = FixedRng(rand=[0.2, 0.3, 0.4])
    assert np.allclose(bwo.fall_paper(rng, X, 0, 1, step), 0.2 * X[0] - 0.3 * X[1] + 0.4 * step)
    r6 = [0.1, 0.2, 0.3, 0.4]; r7 = [0.0, 0.5, 1.0, 0.25]
    rng = FixedRng(vec=[r6, r7])
    out = bwo.fall_fixed(rng, X, 0, 1, step)
    assert np.allclose(out, X[0] + np.array(r6) * (X[0] - X[1]) + (2 * np.array(r7) - 1) * step)


def test_fall_trigger_schedule():
    """Wf goes 0.1 -> 0.05 (Eq. 10); C2 = 2 Wf n; whale fall only for Bf < Wf (Fig. 2)."""
    P = bwo.P
    assert P['wf_start'] == 0.1 and P['wf_end'] == 0.05 and P['n'] == 50


def _bwo_ml_candidates(nD, Npop=50, Max_it=3, seed=0):
    """Run the literal port and record every evaluated point with the population state before it."""
    import bwo_matlab as BM
    seen = []
    f = lambda x: (seen.append(np.array(x, float).copy()), float(np.sum(np.asarray(x) ** 2)))[1]   # noqa: E731
    ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    BM.BWO(Npop, Max_it, -100, 100, nD, f, rng=np.random.default_rng(seed), ops=ops)
    return seen, ops


def test_bwo_ml_budget_count():
    """Counts_run logic: Npop initial + Npop per iteration + one per whale fall."""
    seen, ops = _bwo_ml_candidates(30)
    assert len(seen) == 50 + 3 * 50 + ops['fall']
    assert ops['explore'] + ops['exploit'] == 3 * 50 and ops['fall'] > 0


def test_bwo_ml_low_dim_changes_two_coords():
    """nD <= Npop/5 (BWO.m branch): every exploration candidate differs from its parent in exactly the two
    coordinates params(1:2); exploitation candidates move every coordinate. frac = 0 so kk ~ U(0,1)."""
    import bwo_matlab as BM
    calls = []
    def f(x):
        calls.append(np.array(x, float)); return float(np.sum(np.asarray(x) ** 2))
    ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    BM.BWO(50, 1, -100, 100, 4, f, rng=np.random.default_rng(7), frac=lambda: 0.0, ops=ops)
    init, cand = np.array(calls[:50]), np.array(calls[50:100])   # parent of agent i = its initial position
    nchanged = (cand != init).sum(1)
    assert ops['explore'] > 5
    assert (nchanged == 2).sum() == ops['explore'], (nchanged, ops)
    assert (nchanged == 4).sum() == ops['exploit']


def test_bwo_ml_odd_high_dim_last_coord_kept():
    """nD odd and > Npop/5: the last coordinate of an exploration candidate is the parent's value."""
    import bwo_matlab as BM
    calls = []
    def f(x):
        calls.append(np.array(x, float)); return float(np.sum(np.asarray(x) ** 2))
    ops = {'explore': 0, 'exploit': 0, 'fall': 0, 'iters': 0}
    BM.BWO(10, 1, -100, 100, 5, f, rng=np.random.default_rng(3), frac=lambda: 0.0, ops=ops)
    init, cand = np.array(calls[:10]), np.array(calls[10:20])
    same_last = (cand[:, 4] == init[:, 4]).sum()
    assert ops['explore'] > 0 and same_last == ops['explore']


def test_bwo_ml_harness_equals_direct_call():
    """The BWO-ML arm and a direct call of the port give the same best for the same seed and budget."""
    import bwo_matlab as BM
    from budget import BudgetExhausted
    B = 4000
    o1 = Objective(sph, 10, B); ARMS['BWO-ML'](o1, -100., 100., 10, 9)
    o2 = Objective(sph, 10, B)
    try:
        BM.BWO(50, 10**9, -100., 100., 10, o2, rng=np.random.default_rng(9), frac=lambda: o2.n / B)
    except BudgetExhausted:
        pass
    assert o1.best == o2.best and o1.n == o2.n == B


# ---------------------------------------------------------------- translation equivariance
def _traj(fix, s, B=3000):
    f0 = lambda x: float(np.sum(np.asarray(x) ** 2) + 3 * np.sum(np.cos(np.asarray(x))))  # noqa: E731
    f = lambda x: f0(np.asarray(x) - s)                                                     # noqa: E731
    o = Objective(f, 10, B)
    try:
        bwo._bwo_family(o, -50 + s, 50 + s, 10, 3, fix_e=fix, fix_x=fix, fix_w=fix, clip=False)
    except Exception as e:  # noqa: BLE001  BudgetExhausted
        if type(e).__name__ != 'BudgetExhausted':
            raise
    return o.best, o.best_x - s


def test_translation_equivariance_ti():
    s = np.full(10, 0.0) + np.linspace(-30, 30, 10)
    b0, x0 = _traj(True, np.zeros(10)); b1, x1 = _traj(True, s)
    assert abs(b0 - b1) <= 1e-6 * max(1.0, abs(b0)), (b0, b1)
    assert np.allclose(x0, x1, atol=1e-6)


def test_translation_equivariance_paper_fails():
    """Recorded, not a requirement: BWO-P is expected to change behaviour under translation."""
    s = np.linspace(-30, 30, 10)
    b0, _ = _traj(False, np.zeros(10)); b1, _ = _traj(False, s)
    print(f'BWO-P unshifted best {b0:.4g}, translated best {b1:.4g}')
    assert abs(b0 - b1) > 1e-6 * max(1.0, abs(b0))


# ---------------------------------------------------------------- problems
def test_shift_conditions():
    _, lb, ub, d, _, s_u = PR.classical('F1', 'U', 3)
    _, _, _, _, _, s_r = PR.classical('F1', 'Sr', 3)
    _, _, _, _, _, s_d = PR.classical('F1', 'Sd', 3)
    assert np.all(s_u == 0)
    assert np.all(np.abs(s_r) <= 0.4 * 100) and np.std(s_r) > 1
    assert np.allclose(s_d, s_d[0]) and abs(s_d[0]) <= 40
    f, *_ = PR.classical('F1', 'Sr', 3)
    assert f(s_r) == 0.0
    s2 = PR.classical('F1', 'Sr', 3)[5]
    assert np.array_equal(s_r, s2)                         # same stream for every arm


def test_cec2017_optimum():
    from cec2017_official import CEC2017Function
    for fn in (1, 3, 5, 11, 22, 30):
        F = CEC2017Function(fn, 10)
        assert abs(F(F.shift()) - 100 * fn) < 1e-6 * 100 * fn, fn


def test_cec2022_loader():
    f, lb, ub, d, fopt = PR.cec2022(1, 10)
    v = f(np.zeros(10))
    assert np.isfinite(v) and v >= fopt
    assert isinstance(PR.cec2022_seed(10, 1, 1), int)


# ---------------------------------------------------------------- shifted-sphere sanity
@pytest.mark.parametrize('arm', list(ARMS))
def test_shifted_sphere_sanity(arm):
    o = Objective(sph, 10, 30_000)
    ARMS[arm](o, -100., 100., 10, 11)
    rnd = np.random.default_rng(1).uniform(-100, 100, (1000, 10))
    assert o.best < np.median([sph(x) for x in rnd]), (arm, o.best)
    print(f'{arm:14s} shifted-sphere D10 30k FE: {o.best:.3e}')
