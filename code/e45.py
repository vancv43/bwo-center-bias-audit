"""E4 parameter sensitivity and E5 population diversity of BWO-TI.

Design fixed before any run (no outcome-driven choice; every configuration is reported):
E4  BWO-TI on CEC2017 D=10 (29 functions), 25 runs, budget 10000 D, one factor at a time around the published
    defaults: n in {20, 30, 50, 80, 100} with levy_scale 0.05; levy_scale in {0.01, 0.02, 0.05, 0.1, 0.2} with n = 50.
    9 distinct configurations. Seeds 1000 fn + r as in E2, so the default row repeats E2 (consistency check).
E5  BWO, BWO+E, BWO+X, BWO+W, BWO-TI on CEC2017 D=30, functions F1, F5, F10, F22, 10 runs, budget 300000.
    Per iteration: dimension-wise diversity Div = mean_j mean_i |median_j - x_ij| (Hussain et al. 2019),
    exploration % = 100 Div / max Div, exploitation % = 100 |Div - max Div| / max Div, distance of the population
    centroid to the origin and to the optimum o, recorded at 100 points of the budget.

  python e45.py --exp E4 --shard k --nshards K --workers 4 --out DIR
  python e45.py --exp E5 --shard k --nshards K --workers 4 --out DIR
"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, 'vendor')]
import bwo
from budget import Objective, BudgetExhausted
import problems as PR

E4_N = [20, 30, 50, 80, 100]
E4_L = [0.01, 0.02, 0.05, 0.1, 0.2]
E4_CONF = [(n, 0.05) for n in E4_N] + [(50, l) for l in E4_L if l != 0.05]
E5_ARMS = {'BWO-P': {}, 'BWO-P+E': dict(fix_e=True), 'BWO-P+X': dict(fix_x=True), 'BWO-P+W': dict(fix_w=True),
           'BWO-TI': dict(fix_e=True, fix_x=True, fix_w=True)}
E5_FNS = [1, 5, 10, 22]
NREC = 100


def jobs(exp):
    out = []
    if exp == 'E4':
        for n, l in E4_CONF:
            for fn in [1] + list(range(3, 31)):
                for r in range(1, 26):
                    out.append(dict(exp='E4', n=n, levy=l, fn=fn, dim=10, run=r, seed=1000 * fn + r, budget=100_000))
    else:
        for a in E5_ARMS:
            for fn in E5_FNS:
                for r in range(1, 11):
                    out.append(dict(exp='E5', arm=a, fn=fn, dim=30, run=r, seed=1000 * fn + r, budget=300_000))
    return out


class Rec(Objective):
    """Objective that also records population statistics at NREC evenly spaced points of the budget."""
    def __init__(self, *a, opt=None, **k):
        super().__init__(*a, **k); self.opt = opt; self.pop = None; self.rec = []; self.next = 0

    def snap(self, X):
        med = np.median(X, 0); div = float(np.mean(np.abs(X - med)))
        c = X.mean(0)
        self.rec.append((self.n, div, float(np.linalg.norm(c)), float(np.linalg.norm(c - self.opt)), self.best))


def diversity_family(obj, lb, ub, dim, seed, fix_e=False, fix_x=False, fix_w=False):
    """bwo._bwo_family with a snapshot of the population at the start of every iteration; the random stream and
    every decision are identical (test_e45 checks that the best value matches bwo._bwo_family)."""
    P = bwo.P
    rng = np.random.default_rng(seed)
    lo, hi = bwo._box(lb, ub, dim)
    n = P['n']; B = obj.max_fe
    explore = bwo.explore_fixed if fix_e else bwo.explore_paper
    exploit = bwo.exploit_fixed if fix_x else bwo.exploit_paper
    fall = bwo.fall_fixed if fix_w else bwo.fall_paper
    bound = lambda x: np.clip(x, lo, hi)
    X = lo + rng.random((n, dim)) * (hi - lo)
    fit = np.array([obj(x) for x in X])
    best = X[int(np.argmin(fit))].copy(); fbest = float(fit.min())
    while True:
        if obj.n >= obj.next:
            obj.snap(X); obj.next = obj.n + B / NREC
        tau = obj.n / B
        Bf = rng.random(n) * (1 - tau / 2)
        cand = np.empty_like(X)
        for i in range(n):
            r = bwo._peer(rng, n, i)
            cand[i] = explore(rng, X, i, r) if Bf[i] > P['bf_threshold'] else exploit(rng, X, i, r, best, tau)
        for i in range(n):
            c = bound(cand[i]); fc = obj(c)
            if fc < fit[i]:
                X[i] = c; fit[i] = fc
                if fc < fbest:
                    best = c.copy(); fbest = fc
        tau = obj.n / B
        Wf = P['wf_start'] - (P['wf_start'] - P['wf_end']) * tau
        step = (hi - lo) * math.exp(-2 * Wf * n * tau)
        for i in range(n):
            if Bf[i] < Wf:
                r = bwo._peer(rng, n, i)
                c = bound(fall(rng, X, i, r, step)); fc = obj(c)
                if fc < fit[i]:
                    X[i] = c; fit[i] = fc
                    if fc < fbest:
                        best = c.copy(); fbest = fc


def one(job):
    np.random.seed(job['seed'] % 2**32)
    f, lb, ub, dim, fopt = PR.cec2017(job['fn'], job['dim'])
    t = time.perf_counter()
    if job['exp'] == 'E4':
        bwo.P['n'] = job['n']; bwo.P['levy_scale'] = job['levy']
        obj = Objective(f, dim, job['budget'], fopt)
        try:
            bwo._bwo_family(obj, lb, ub, dim, job['seed'], fix_e=True, fix_x=True, fix_w=True)
        except BudgetExhausted:
            pass
        extra = {}
    else:
        o = np.loadtxt(os.path.join(HERE, 'vendor', 'cec2017_official', 'input_data', f'shift_data_{job["fn"]}.txt')).ravel()[:dim]
        obj = Rec(f, dim, job['budget'], fopt, opt=o)
        try:
            diversity_family(obj, lb, ub, dim, job['seed'], **E5_ARMS[job['arm']])
        except BudgetExhausted:
            pass
        extra = {'rec': json.dumps([[int(a), float(f'{b:.6g}'), float(f'{c:.6g}'), float(f'{d:.6g}'),
                                     float(f'{e:.6g}')] for a, b, c, d, e in obj.rec])}
    assert obj.n == job['budget'], obj.n
    return {**job, 'error': repr(obj.error()), 'sec': round(time.perf_counter() - t, 3), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    J = jobs(a.exp)[a.shard::a.nshards]
    if a.limit:
        J = J[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'{a.exp}_s{a.shard:02d}.csv')
    done = set()
    if os.path.exists(path):
        done = {(r['seed'], r.get('n', ''), r.get('levy', ''), r.get('arm', '')) for r in csv.DictReader(open(path))}
    k = lambda j: (str(j['seed']), str(j.get('n', '')), str(j.get('levy', '')), j.get('arm', ''))
    J = [j for j in J if k(j) not in done]
    print(a.exp, 'shard', a.shard, 'jobs', len(J), flush=True)
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(one, J)):
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
            if i % 50 == 0:
                print(i, row.get('arm', (row.get('n'), row.get('levy'))), row['fn'], row['error'], row['sec'], flush=True)


if __name__ == '__main__':
    main()
