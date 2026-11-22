"""Problems for E1 (classical, three shift conditions) and E2 (official CEC2017 / CEC2022)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))
sys.path.insert(0, os.path.join(HERE, 'vendor', 'cec2017_official'))
sys.path.insert(0, os.path.join(HERE, 'vendor', 'cec2022_data'))

from config import E1, flat   # noqa: E402

E1P = flat(E1)
CLASSICAL = [f'F{k}' for k in range(1, 14)]
RATIO_FUNCS = [f for f in CLASSICAL if f not in E1P['ratio_excluded']]


def shift_vector(fn, dim, seed, cond, lb, ub):
    """U: zeros; Sr: s_j ~ U(-0.4 h, 0.4 h) independently; Sd: one c ~ U(-0.4 h, 0.4 h) for every j.
    Same stream for every arm (paired), one stream per (function, seed)."""
    if cond == 'U':
        return np.zeros(dim)
    k = int(fn[1:])
    rng = np.random.default_rng([E1P['shift_seed_offset'] + seed, k])
    h = (np.broadcast_to(np.asarray(ub, float), (dim,)) - np.broadcast_to(np.asarray(lb, float), (dim,))) / 2
    a = E1P['shift_frac']
    sr = rng.uniform(-a, a, dim) * h
    c = rng.uniform(-a, a)
    if cond == 'Sr':
        return sr
    if cond == 'Sd':
        return c * h
    raise KeyError(cond)


def classical(fn, cond, seed, dim=None):
    """-> f(x) (scalar), lb, ub, dim, f_opt, shift"""
    from benchmarks import BENCHMARKS
    func, lb, ub, d0, fopt = BENCHMARKS[fn]
    dim = dim or d0
    s = shift_vector(fn, dim, seed, cond, lb, ub)

    def f(x):
        return float(func((np.asarray(x, float) - s)[None, :])[0])
    return f, float(lb), float(ub), dim, float(fopt), s


def cec2017(fn, dim):
    from cec2017_official import CEC2017Function
    F = CEC2017Function(fn, dim)
    return F, -100.0, 100.0, dim, F.optimum


def cec2022(fn, dim):
    import cec2022_fast as C
    F = C.CEC2022Function(fn, dim)
    return F, -100.0, 100.0, dim, F.optimum


def cec2022_seed(dim, fn, run, runs=30):
    """Official CEC2022 rule, the official seeding rule."""
    seeds = np.loadtxt(os.path.join(HERE, 'vendor', 'cec2022_data', 'input_data', 'Rand_Seeds.txt'))
    idx = (dim // 10 * fn * runs + run) - runs
    return int(seeds[idx % 1000])


_RW_SO = None


def build_rw(out_dir=None):
    """Compile the official CEC2020-RW C++ code once. Returns the .so path."""
    import subprocess
    src = os.path.join(HERE, 'vendor', 'cec2020rwc_src')
    so = os.path.join(out_dir or os.path.join(HERE, 'vendor'), 'cec2020rwc.so')
    if not os.path.exists(so):
        subprocess.run(['g++', '-O2', '-std=c++14', '-shared', '-fPIC', '-I', os.path.join(src, 'eigen-3.4.0'),
                        os.path.join(src, 'cec2020rwc.cpp'), '-o', so], check=True)
    os.environ['CEC2020RWC_SO'] = so
    return so


def rw2020(code):
    """-> penalised f(x), lb (array), ub (array), dim, parts(x) -> (f, v). Integer variables rounded inside the
    official code. Violation v = sum max(0, g) + sum max(0, |h| - 1e-4)."""
    from rw_official import rw_official
    func, lb, ub, dim, _ = rw_official(code)

    def f(x):
        return float(func(np.asarray(x, float)[None, :])[0])
    return f, lb, ub, dim, func.parts


def rw_budget(dim):
    from config import E3
    return E3['budget_small'][0] if dim <= 10 else E3['budget_large'][0]
