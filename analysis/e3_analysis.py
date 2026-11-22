"""E3 analysis, written before any E3 result existed.

  python analysis/e3_analysis.py results   -> analysis/E3_REPORT.md, analysis/e3_wtl.csv, analysis/e3_feas.csv

(1) Per problem: Wilcoxon rank-sum (Mann-Whitney U) BWO-TI vs each arm on the penalised value, two-sided,
    Holm over the 19 problems, W/T/L from BWO-TI's side.
(2) Friedman on the mean penalised value per problem, average ranks, Holm post-hoc z-test vs BWO-TI.
(3) Feasibility rate (v = 0) of the final best point per arm and problem.
(4) Secondary ranking by the CEC2020 feasibility rule: within a problem, all final points of all arms are ranked
    (feasible beats infeasible; feasible by f; infeasible by v); each arm gets its mean rank; Friedman ranks over
    problems. Problems where the primary per-problem order of arms differs from the secondary one are named
    (Kendall tau < 1 between the two per-problem arm orders).
"""
import csv
import glob
import json
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
PROBS = [f'RC{k}' for k in range(15, 34)]
NRUNS = 25


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[k])); adj[k] = run
    return adj


def load(d):
    data = defaultdict(list); seen = set()
    for f in glob.glob(os.path.join(d, 'bwo-e3-*', '**', 'E3_*.csv'), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r['smoke'] != '0':
                    continue
                key = (r['arm'], r['fn'], int(r['run']))
                assert key not in seen, key; seen.add(key)
                assert int(r['fe_used']) <= int(r['budget']), key
                o = json.loads(r['ops'])
                data[key[:2]].append((float(r['best']), float(o['f_best']), float(o['v_best'])))
    return data


def friedman_block(M, arms, L, title):
    fr_s, fr_p = _friedman(M)
    R = _ranks(M).mean(0)
    k, n = len(arms), M.shape[0]; se = math.sqrt(k * (k + 1) / (6 * n)); i0 = arms.index(REF)
    others = [a for a in arms if a != REF]
    padj = dict(zip(others, holm([2 * stats.norm.sf(abs(R[arms.index(a)] - R[i0]) / se) for a in others])))
    L += ['', f'{title}: Friedman chi2 = {fr_s:.1f}, p = {fr_p:.2e}.', '',
          '| rank | arm | avg rank | p_Holm vs BWO-TI |', '|---|---|---|---|']
    for pos, j in enumerate(np.argsort(R)):
        a = arms[j]
        L.append(f"| {pos + 1} | {a} | {R[j]:.2f} | {'—' if a == REF else f'{padj[a]:.3g}'} |")
    return R


def main(d):
    data = load(d)
    arms = sorted({k[0] for k in data}); others = [a for a in arms if a != REF]
    for a in arms:
        for p in PROBS:
            assert len(data[(a, p)]) == NRUNS, (a, p, len(data[(a, p)]))
    L = ['# E3 results, CEC2020-RW RC15-RC33, analysed by analysis/e3_analysis.py', '',
         f'Arms ({len(arms)}): {", ".join(arms)}. {len(PROBS)} problems x {NRUNS} runs.', '',
         '## (1) BWO-TI vs each arm, rank-sum, Holm over problems', '', '| BWO-TI vs | W | T | L |', '|---|---|---|---|']
    wtl = []
    for a in others:
        ps, sg = [], []
        for p in PROBS:
            x = np.array([t[0] for t in data[(REF, p)]]); y = np.array([t[0] for t in data[(a, p)]])
            if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
                ps.append(1.0); sg.append(0); continue
            u = stats.mannwhitneyu(x, y, alternative='two-sided')
            ps.append(float(u.pvalue)); sg.append(-1 if u.statistic < len(x) * len(y) / 2 else 1)
        adj = holm(ps)
        w = sum(1 for q, s in zip(adj, sg) if q < 0.05 and s < 0); l = sum(1 for q, s in zip(adj, sg) if q < 0.05 and s > 0)
        L.append(f'| {a} | {w} | {len(PROBS) - w - l} | {l} |'); wtl.append(dict(vs=a, W=w, T=len(PROBS) - w - l, L=l))
    M = np.array([[np.mean([t[0] for t in data[(a, p)]]) for a in arms] for p in PROBS])
    L.append('\n## (2) Primary ranking (mean penalised value)')
    friedman_block(M, arms, L, 'Primary')
    # (3) feasibility
    L += ['', '## (3) Feasibility rate of final points (v = 0), per arm over all problems', '',
          '| arm | feasible runs | problems with all runs feasible |', '|---|---|---|']
    feas = []
    for a in arms:
        fr = {p: np.mean([t[2] == 0 for t in data[(a, p)]]) for p in PROBS}
        n = sum(sum(t[2] == 0 for t in data[(a, p)]) for p in PROBS)
        L.append(f'| {a} | {n}/{len(PROBS) * NRUNS} | {sum(v == 1 for v in fr.values())}/{len(PROBS)} |')
        feas += [dict(arm=a, problem=p, feasible_rate=fr[p]) for p in PROBS]
    # (4) feasibility-rule ranking
    S = np.zeros((len(PROBS), len(arms))); disagree = []
    for i, p in enumerate(PROBS):
        pts = [(a, t) for a in arms for t in data[(a, p)]]
        keyv = [(0, t[1]) if t[2] == 0 else (1, t[2]) for _, t in pts]
        order = sorted(range(len(pts)), key=lambda j: keyv[j]); rk = np.empty(len(pts))
        j = 0
        while j < len(order):                         # average ranks on ties
            e = j
            while e + 1 < len(order) and keyv[order[e + 1]] == keyv[order[j]]:
                e += 1
            for q in order[j:e + 1]:
                rk[q] = (j + e) / 2 + 1
            j = e + 1
        for c, a in enumerate(arms):
            S[i, c] = np.mean([rk[j] for j, (b, _) in enumerate(pts) if b == a])
        tau = stats.kendalltau(_ranks(M[i:i+1])[0], stats.rankdata(S[i])).statistic
        if not (tau > 1 - 1e-12):
            disagree.append(f'{p} (tau {tau:.2f})')
    L.append('\n## (4) Secondary ranking (CEC2020 feasibility rule)')
    friedman_block(S, arms, L, 'Secondary')
    L += ['', f'Problems where the per-problem arm order differs between (2) and (4): '
          f'{", ".join(disagree) if disagree else "none"}.']
    for name, rows in (('e3_wtl.csv', wtl), ('e3_feas.csv', feas)):
        with open(os.path.join(HERE, 'out', name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    open(os.path.join(HERE, 'out', 'E3_REPORT.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1])
