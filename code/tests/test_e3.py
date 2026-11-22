"""E3 (CEC2020-RW RC15-RC33) checks: loader, penalty, budget rule, box, exact budget, determinism."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), 'vendor')]

import problems as PR  # noqa: E402
PR.build_rw()
from run import jobs_e3, one  # noqa: E402
from rw_official import D  # noqa: E402


def test_jobs_design():
    J = jobs_e3()
    assert len(J) == 13 * 19 * 25
    assert {j['fn'] for j in J} == {f'RC{k}' for k in range(15, 34)}
    for j in J:
        assert j['budget'] == (100_000 if j['dim'] <= 10 else 200_000)
        assert j['dim'] == D[int(j['fn'][2:]) - 1]


def test_penalty_and_bounds():
    rng = np.random.default_rng(0)
    for k in range(15, 34):
        f, lb, ub, dim, parts = PR.rw2020(f'RC{k}')
        assert lb.shape == ub.shape == (dim,) and np.all(lb < ub)
        for _ in range(5):
            x = lb + rng.random(dim) * (ub - lb)
            fx, v = parts(x)
            y = f(x)
            assert v >= 0 and (y == 1e30 or np.isclose(y, fx + 1e6 * v, rtol=1e-12, atol=0))


def test_exact_budget_box_determinism():
    for arm in ('BWO-TI', 'BWO-P', 'AMBWO', 'L-SHADE', 'BIPOP-CMA-ES'):
        j = dict(exp='E3', arm=arm, suite='rw2020', fn='RC27', dim=10, cond='official', seed=27001, run=1,
                 budget=3000)
        a, b = one(dict(j)), one(dict(j))
        assert a['best'] == b['best'] and a['fe_used'] == b['fe_used']
        assert a['fe_used'] <= 3000
