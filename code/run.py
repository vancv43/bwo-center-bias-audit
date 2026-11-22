"""Experiment runner. One CSV per (experiment, shard); finished rows are never
rewritten; a restart skips rows already present (resume replays nothing).

  python run.py --exp E1 --shard 0 --nshards 4 --out results/E1 --workers 4
  python run.py --exp E2 --suite cec2017 --dim 30 --shard 0 --nshards 8 --out results/E2
Smoke runs must pass --smoke (budget / 100, results flagged smoke=1, written to a separate folder).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'vendor'))

import config  # noqa: E402

FIELDS = ['exp', 'arm', 'suite', 'fn', 'dim', 'cond', 'seed', 'run', 'budget', 'fe_used', 'best', 'error',
          'trace', 'ops', 'sec', 'smoke', 'code_sha', 'config']


def code_sha():
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(HERE, '*.py')) + glob.glob(os.path.join(HERE, 'vendor', '*.py'))):
        h.update(os.path.basename(p).encode()); h.update(open(p, 'rb').read())
    return h.hexdigest()[:16]


def jobs_e1():
    from arms import E1_ARMS
    from problems import CLASSICAL
    P = config.flat(config.E1)
    return [dict(exp='E1', arm=a, suite='classical', fn=f, dim=P['dim'], cond=c, seed=s, run=s, budget=P['budget'])
            for a in E1_ARMS for f in CLASSICAL for c in P['conds'] for s in P['seeds']]


def jobs_e2(suite, dim):
    from arms import E2_ARMS
    from problems import cec2022_seed
    P = config.flat(config.E2)
    out = []
    if suite == 'cec2017':
        for a in E2_ARMS:
            for fn in [1] + list(range(3, 31)):
                for r in range(1, P['cec2017_runs'] + 1):
                    out.append(dict(exp='E2', arm=a, suite=suite, fn=f'F{fn}', dim=dim, cond='official',
                                    seed=1000 * fn + r, run=r, budget=P['cec2017_budget_per_dim'] * dim))
    elif suite == 'cec2022':
        for a in E2_ARMS:
            for fn in range(1, 13):
                for r in range(1, P['cec2022_runs'] + 1):
                    out.append(dict(exp='E2', arm=a, suite=suite, fn=f'F{fn}', dim=dim, cond='official',
                                    seed=cec2022_seed(dim, fn, r), run=r, budget=P['cec2022_budget'][dim]))
    return out


def jobs_r1(suite, dim, cal=False):
    """R1: AMBWO and BWO-ML at the settings of You et al. 2024 (N = 30, T = 500), 30 runs, for comparison with
    their Tables A1 and A5. The FE cap is never reached; fe_used records the real count."""
    from arms import REP_ARMS, CAL_ARMS
    from problems import cec2022_seed
    arms = CAL_ARMS if cal else REP_ARMS
    exp = 'C1' if cal else 'R1'
    fns = [1] + list(range(3, 31)) if suite == 'cec2017' else list(range(1, 13))
    return [dict(exp=exp, arm=a, suite=suite, fn=f'F{fn}', dim=dim, cond='paper',
                 seed=(1000 * fn + r) if suite == 'cec2017' else cec2022_seed(dim, fn, r), run=r, budget=1_000_000)
            for a in arms for fn in fns for r in range(1, 31)]


def jobs_e3():
    from arms import E3_ARMS
    from problems import rw2020, rw_budget
    from rw_official import D
    P = config.flat(config.E3)
    out = []
    for a in E3_ARMS:
        for code in P['problems']:
            k = int(code[2:]); dim = D[k - 1]
            for r in range(1, P['runs'] + 1):
                out.append(dict(exp='E3', arm=a, suite='rw2020', fn=code, dim=dim, cond='official',
                                seed=1000 * k + r, run=r, budget=rw_budget(dim)))
    return out


def key(j):
    return (j['arm'], j['suite'], j['fn'], str(j['dim']), j['cond'], str(j['run']))


def one(job):
    from arms import ARMS, REP_ARMS, CAL_ARMS
    ARMS = {**ARMS, **REP_ARMS, **CAL_ARMS}
    from budget import Objective
    import problems as PR
    np.random.seed(int(job['seed']) % (2**32))         # F7 noise and any library using the global stream
    if job['suite'] == 'classical':
        f, lb, ub, dim, fopt, _ = PR.classical(job['fn'], job['cond'], job['seed'], job['dim'])
    elif job['suite'] == 'cec2017':
        f, lb, ub, dim, fopt = PR.cec2017(int(job['fn'][1:]), job['dim'])
    elif job['suite'] == 'cec2022':
        f, lb, ub, dim, fopt = PR.cec2022(int(job['fn'][1:]), job['dim'])
    else:                                              # rw2020: raw penalised values (some optima are negative)
        f, lb, ub, dim, parts = PR.rw2020(job['fn']); fopt = 0.0
    obj = Objective(f, dim, job['budget'], fopt)
    t = time.perf_counter()
    ARMS[job['arm']](obj, lb, ub, dim, job['seed'])
    sec = time.perf_counter() - t
    assert obj.n <= job['budget'], (job, obj.n)
    assert obj.best_x is None or np.all((obj.best_x >= lb - 1e-12) & (obj.best_x <= ub + 1e-12)), job
    if job['suite'] == 'rw2020':
        fb, vb = parts(obj.best_x)
        assert np.isclose(fb + config.E3['penalty'][0] * vb, obj.best, rtol=1e-12, atol=0) or obj.best >= 1e30, job
        tr = list(obj.trace) + [obj.best] * (len(obj.ck) - len(obj.trace))
        ops = {**getattr(obj, 'ops', {}), 'f_best': fb, 'v_best': vb}
        return {**job, 'fe_used': obj.n, 'best': repr(float(obj.best)), 'error': repr(float(obj.best)),
                'trace': json.dumps([float(f'{v:.6g}') for v in tr]), 'ops': json.dumps(ops), 'sec': round(sec, 3)}
    return {**job, 'fe_used': obj.n, 'best': repr(float(obj.best)), 'error': repr(obj.error()),
            'trace': json.dumps([float(f'{v:.6g}') for v in obj.error_trace()]),
            'ops': json.dumps(getattr(obj, 'ops', {})), 'sec': round(sec, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=['E1', 'E2', 'R1', 'C1', 'E3'])
    ap.add_argument('--suite', default='classical'); ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--arms', default='')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.exp == 'E3':
        import problems as PR
        PR.build_rw()                                  # compile once before the worker pool forks
        a.suite, a.dim = 'rw2020', 0
    jobs = (jobs_e1() if a.exp == 'E1' else jobs_e2(a.suite, a.dim) if a.exp == 'E2'
            else jobs_e3() if a.exp == 'E3' else jobs_r1(a.suite, a.dim, cal=a.exp == 'C1'))
    if a.arms:
        keep = set(a.arms.split(',')); jobs = [j for j in jobs if j['arm'] in keep]
    jobs = jobs[a.shard::a.nshards]
    if a.limit:
        jobs = jobs[:a.limit]
    if a.smoke:
        assert 'smoke' in a.out.lower(), 'smoke results must go to a folder named smoke'
        for j in jobs:
            j['budget'] = max(1000, j['budget'] // 100)
    sha = code_sha(); cfg = json.dumps({'E3': config.flat(config.E3), 'BWO': config.flat(config.BWO), 'tags': config.tags(config.BWO), 'AMBWO': config.flat(config.AMBWO), 'AMBWO_tags': config.tags(config.AMBWO)})
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'{a.exp}_{a.suite}_D{a.dim}_shard{a.shard:02d}of{a.nshards:02d}.csv')
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row['code_sha'] != sha:
                    sys.exit(f'{path} was written by code {row["code_sha"]}, now {sha}: use a new folder')
                done.add(key(row))
    todo = [j for j in jobs if key({k: str(v) if k in ('dim', 'run') else v for k, v in j.items()}) not in done]
    print(f'{len(jobs)} jobs, {len(done)} done, {len(todo)} to run, code {sha}', flush=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for k, row in enumerate(pool.imap_unordered(one, todo, chunksize=1)):
            row.update(smoke=int(a.smoke), code_sha=sha, config=cfg)
            w.writerow({f: row.get(f, '') for f in FIELDS}); fh.flush()
            if k % 50 == 0:
                print(f'{k + 1}/{len(todo)} {row["arm"]} {row["fn"]} {row["cond"]} err={row["error"]} {row["sec"]}s', flush=True)
    open(path + '.DONE', 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
