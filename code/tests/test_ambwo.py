"""Tests for the AMBWO arm. Budget, determinism, box and shifted-sphere
sanity for AMBWO are covered by the parametrised tests in test_g0.py (AMBWO is in ARMS)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'vendor'))

from budget import Objective  # noqa: E402
import ambwo as A             # noqa: E402
import problems as PR         # noqa: E402


def test_schedules_as_printed():
    assert math.isclose(A.e_factor(0.0), 1.4 - 1.1)
    assert math.isclose(A.e_factor(1.0), 1.4 * math.e - 1.1)
    assert math.isclose(A.cf(0.0), 1.0) and A.cf(1.0) == 0.0
    assert math.isclose(A.cf(0.5), 0.5 ** 1.5)


def test_scores_eq19():
    X = np.array([[0., 0.], [3., 4.], [6., 8.]]); fit = np.array([1., 2., 5.])
    S = A.scores(X, fit, X[0])
    assert np.allclose(S, 0.5 * np.array([0, 5, 10]) / 10 + 0.5 * np.array([0, 1, 4]) / 5)


def test_mean_cov_eq15_17():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3)); fit = rng.random(10)
    xm, cov = A.mean_cov(X, fit)
    top = X[np.argsort(fit)[:5]]
    w = np.log(6) - np.log(np.arange(1, 6)); w /= w.sum()
    assert np.allclose(xm, w @ top)
    assert np.allclose(cov, (top - xm).T @ (top - xm) / 4)


class R:
    def __init__(self, vals): self.v = list(vals)
    def random(self, size=None):
        return self.v.pop(0) if size is None else np.array(self.v.pop(0), float)
    def permutation(self, d): return np.arange(d)[::-1]
    def choice(self, n, k, replace=False): return np.array([1, 2])


def test_explore_eq14_hand():
    X = np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]])
    r1, r2, r8, ef = 0.5, 0.125, 0.25, 2.0
    got = A.explore_eq14(R([r1, r2, r8]), X, 0, 1, ef)
    src = X[0, [3, 2, 1, 0]]
    trig = np.array([math.cos(2 * math.pi * r2), math.sin(2 * math.pi * r2)] * 2)
    assert np.allclose(got, src + ef * (r8 * X[1, 3] - src) * 1.5 * trig)


def test_avoid_eq21_both_branches():
    X = np.array([[1., 1.], [2., 4.], [0., 1.]]); lo = np.array([-10., -10.]); hi = np.array([10., 10.])
    got = A.avoid_eq21(R([0.2, [0.1, 0.9], [0.5, 0.5]]), X, 0, lo, hi, 0.5)   # U = [1, 0], LB + r10 (UB - LB) = 0
    assert np.allclose(got, X[0])
    got = A.avoid_eq21(R([0.9, 0.5]), X, 0, lo, hi, 0.5)
    assert np.allclose(got, X[0] + (0.34 * 0.5 + 0.5) * (X[1] - X[2]))


def test_operator_reach_and_budget_e2_scale():
    f, lb, ub, dim, fopt = PR.cec2017(5, 10)
    o = Objective(f, dim, 100_000, fopt)
    A.run_ambwo(o, lb, ub, dim, 3)
    assert o.n == 100_000
    assert all(o.ops[k] > 0 for k in ('gauss', 'explore', 'exploit', 'fall', 'avoid')), o.ops
    assert o.ops['avoid'] == 30 * o.ops['iters'] or o.ops['avoid'] - 30 * o.ops['iters'] < 30


# ---------------------------------------------------------------- C1 calibration readings
def test_reading_switches():
    assert math.isclose(A.e_factor(0.0, 'decreasing'), 2.5)
    assert math.isclose(A.e_factor(1.0, 'decreasing'), 1.1 + 1.4 / math.e)
    assert A.e_factor(0.3, 'decreasing') > A.e_factor(0.6, 'decreasing')
    assert math.isclose(A.cf(0.5, 'gazelle'), 0.5 ** 1.0) and math.isclose(A.cf(0.0, 'gazelle'), 1.0)
    rng = np.random.default_rng(1)
    S = np.array([0.0, 0.0, 1.0])                      # inverse: the agent with the top score is never drawn
    assert all(A.roulette(rng, S, 'inverse') != 2 for _ in range(200))
    assert all(A.roulette(rng, S, 'prop') == 2 for _ in range(50))


def test_eight_readings_distinct():
    from arms import CAL_READINGS
    assert len(CAL_READINGS) == 8 and len({tuple(v.values()) for v in CAL_READINGS.values()}) == 8
    assert CAL_READINGS['AMBWO@R0'] == A.READING0


def test_r0_equals_pre_d3_code():
    """AMBWO@R0 must equal the original ambwo.py run by the same numpy. References were computed with
    the original file (numpy 1.26.4). numpy 2.1.3 draws multivariate_normal differently, so the
    bit-for-bit check against the R1 rows is done on the released C1 results."""
    import numpy
    import run
    if not numpy.__version__.startswith('1.26'):
        import pytest
        pytest.skip('reference values are for numpy 1.26')
    ref = {('F1', 1, 1001): (1528.4464672423671, 31604), ('F5', 2, 5002): (15.17531747993985, 31522)}
    for (fn, r, seed), (err, fe) in ref.items():
        out = run.one(dict(exp='C1', arm='AMBWO@R0', suite='cec2017', fn=fn, dim=10, cond='paper', seed=seed,
                           run=r, budget=1_000_000))
        assert float(out['error']) == err and out['fe_used'] == fe, (fn, out['error'], out['fe_used'])


def test_readings_change_the_search():
    import run
    errs = {k: float(run.one(dict(exp='C1', arm=k, suite='cec2017', fn='F5', dim=10, cond='paper', seed=5002,
                                  run=2, budget=1_000_000))['error']) for k in ('AMBWO@R0', 'AMBWO@R1', 'AMBWO@R2', 'AMBWO@R4')}
    assert len(set(errs.values())) == 4, errs
