"""E2 analysis. Rules fixed before the data; nothing tuned on results.

  python analysis/e2_analysis.py results   -> analysis/E2_REPORT.md, analysis/e2_wtl.csv, analysis/e2_ranks.csv

5.1 Per function: Wilcoxon rank-sum (= Mann-Whitney U), two-sided, alpha 0.05, Holm over the functions of one
    suite x dimension, BWO-TI vs each other arm. W/T/L from BWO-TI's side; direction = which arm has the lower
    mean rank (U statistic). Errors below 1e-8 set to 0 (CEC rule).
5.2 Suite level: Friedman over functions (blocks) on mean errors, average ranks, post-hoc z-test vs BWO-TI
    (Demsar 2006), Holm over the comparisons.
H4: BWO-TI vs BWO-P on CEC2017 D30: wins > losses.
All rows of all arms are used; row counts are checked against the design before anything is computed.
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ranking import ranks as _ranks, friedman as _friedman
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REF = 'BWO-TI'
CELLS = [('cec2017', 10), ('cec2017', 30), ('cec2017', 50), ('cec2022', 10), ('cec2022', 20)]
NRUNS = {'cec2017': 51, 'cec2022': 30}
NFN = {'cec2017': 29, 'cec2022': 12}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[k])); adj[k] = run
    return adj


def load(d):
    data = defaultdict(list); seen = set()
    for f in glob.glob(os.path.join(d, 'bwo-e2*', '**', 'E2_*.csv'), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r['smoke'] != '0':
                    continue
                key = (r['suite'], int(r['dim']), r['arm'], r['fn'], int(r['run']))
                assert key not in seen, key; seen.add(key)
                assert int(r['fe_used']) <= int(r['budget']), key
                e = float(r['error']); data[key[:4]].append(e if e >= 1e-8 else 0.0)
    return data


def main(d):
    data = load(d)
    arms = sorted({k[2] for k in data}); others = [a for a in arms if a != REF]
    L = ['# E2 results, analysed by analysis/e2_analysis.py', '',
         f'Arms ({len(arms)}): {", ".join(arms)}. Reference arm: {REF}.', '']
    wtl_rows, rank_rows = [], []
    for suite, dim in CELLS:
        fns = sorted({k[3] for k in data if k[0] == suite and k[1] == dim}, key=lambda s: int(s[1:]))
        assert len(fns) == NFN[suite], (suite, dim, fns)
        for a in arms:
            for fn in fns:
                assert len(data[(suite, dim, a, fn)]) == NRUNS[suite], (suite, dim, a, fn, len(data[(suite, dim, a, fn)]))
        L += [f'## {suite.upper()} D{dim} ({len(fns)} functions x {NRUNS[suite]} runs)', '',
              '| BWO-TI vs | W | T | L |', '|---|---|---|---|']
        for a in others:
            ps, sign = [], []
            for fn in fns:
                x = np.array(data[(suite, dim, REF, fn)]); y = np.array(data[(suite, dim, a, fn)])
                if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
                    ps.append(1.0); sign.append(0); continue
                u = stats.mannwhitneyu(x, y, alternative='two-sided')
                ps.append(float(u.pvalue)); sign.append(-1 if u.statistic < len(x) * len(y) / 2 else 1)
            adj = holm(ps)
            w = sum(1 for p, s in zip(adj, sign) if p < 0.05 and s < 0)
            l = sum(1 for p, s in zip(adj, sign) if p < 0.05 and s > 0)
            t = len(fns) - w - l
            L.append(f'| {a} | {w} | {t} | {l} |')
            wtl_rows.append(dict(suite=suite, dim=dim, vs=a, W=w, T=t, L=l))
        # Friedman on mean errors
        M = np.array([[np.mean(data[(suite, dim, a, fn)]) for a in arms] for fn in fns])
        fr_s, fr_p = _friedman(M)
        R = _ranks(M).mean(0)
        k, n = len(arms), len(fns); se = math.sqrt(k * (k + 1) / (6 * n)); i0 = arms.index(REF)
        pz = [2 * stats.norm.sf(abs(R[arms.index(a)] - R[i0]) / se) for a in others]
        padj = dict(zip(others, holm(pz)))
        L += ['', f'Friedman chi2 = {fr_s:.1f}, p = {fr_p:.2e}. Average ranks (1 = best); '
              f'Holm post-hoc vs {REF}:', '', '| rank | arm | avg rank | p_Holm vs BWO-TI |', '|---|---|---|---|']
        for j in np.argsort(R):
            a = arms[j]; p = '—' if a == REF else f'{padj[a]:.3g}'
            L.append(f'| {list(np.argsort(R)).index(j) + 1} | {a} | {R[j]:.2f} | {p} |')
            rank_rows.append(dict(suite=suite, dim=dim, arm=a, avg_rank=R[j], p_holm_vs_ref=p))
        L.append('')
    h4 = next(r for r in wtl_rows if r['suite'] == 'cec2017' and r['dim'] == 30 and r['vs'] == 'BWO-P')
    L += ['## H4', f"BWO-TI vs BWO-P on CEC2017 D30: W/T/L = {h4['W']}/{h4['T']}/{h4['L']} -> "
          f"H4 {'supported' if h4['W'] > h4['L'] else 'not supported'}."]
    for name, rows in (('e2_wtl.csv', wtl_rows), ('e2_ranks.csv', rank_rows)):
        with open(os.path.join(HERE, 'out', name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    open(os.path.join(HERE, 'out', 'E2_REPORT.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1])
