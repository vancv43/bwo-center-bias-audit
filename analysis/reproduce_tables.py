"""Rebuild every table of the paper and of its supplementary material from the per-run results in results/.

  python analysis/reproduce_tables.py      -> analysis/out/tables.json

Each table is written as a list of rows, each row a list of cells formatted exactly as printed in the paper.
verify_results.py compares these cells with the printed values in reported/reported_values.json.
The script first runs e1_bias.py, e2_analysis.py and e3_analysis.py when their outputs are missing.
"""
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ranking import ranks, friedman  # noqa: E402

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); OUT = os.path.join(HERE, 'out')
REF = 'BWO-TI'
NAME = {'BWO-P': 'BWO', 'BWO-ML': 'BWO-MATLAB', 'BWO-M': 'BWO-mealpy', 'BWO-P+E': 'BWO+E', 'BWO-P+X': 'BWO+X',
        'BWO-P+W': 'BWO+W', 'RS': 'Random search'}
nm = lambda a: NAME.get(a, a)
CELLS = [('cec2017', 10), ('cec2017', 30), ('cec2017', 50), ('cec2022', 10), ('cec2022', 20)]


def read(path):
    with open(os.path.join(RES, path)) as fh:
        return list(csv.DictReader(fh))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[k])); adj[k] = run
    return adj


def fmt(x):                       # supplementary tables: E notation, errors below 1e-8 printed as 0
    return '0' if x == 0 or abs(x) < 1e-8 else '%.2E' % x


def sci(x):
    if x == 0:
        return '0'
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if -2 <= e <= 3:
        return '%.*f' % (max(0, 2 - e), x) if abs(x) >= 1 else f'{x:.2g}'
    return f'{m:.2f}E{e:+03d}'.replace('E+0', 'E+').replace('E-0', 'E-')


def pfmt(p):
    return '1' if p >= 0.9995 else (f'{p:.3g}' if p >= 1e-3 else sci(p))


def wtl(x_by, y_by, keys):
    ps, sg = [], []
    for k in keys:
        x, y = np.array(x_by[k]), np.array(y_by[k])
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            ps.append(1.0); sg.append(0); continue
        u = stats.mannwhitneyu(x, y, alternative='two-sided')
        ps.append(float(u.pvalue)); sg.append(-1 if u.statistic < len(x) * len(y) / 2 else 1)
    adj = holm(ps)
    w = sum(1 for q, s in zip(adj, sg) if q < 0.05 and s < 0); l = sum(1 for q, s in zip(adj, sg) if q < 0.05 and s > 0)
    return w, len(keys) - w - l, l


def ensure_stats():
    need = [('e1_summary.json', ['analysis/e1_bias.py']), ('e2_ranks.csv', ['analysis/e2_analysis.py', 'results']),
            ('e3_wtl.csv', ['analysis/e3_analysis.py', 'results'])]
    for f, cmd in need:
        if not os.path.exists(os.path.join(OUT, f)):
            subprocess.run([sys.executable] + cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------- Table 4, AMBWO readings
def tab_c1():
    pub = json.load(open(os.path.join(ROOT, 'reported', 'ambwo_published_tables.json')))
    fopt22 = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]

    def ref(suite, fn):
        m = int(fn[1:])
        if suite == 'cec2017':          # the AMBWO article numbers CEC2017 without F2: its Fk is official F(k+1), k >= 2
            return pub['Table A1.']['F1' if m == 1 else f'F{m - 1}']['AMBWO']
        return pub['Table A5.'][fn]['AMBWO']
    V = defaultdict(list)
    for r in read('bwo-c1/C1_ambwo_readings.csv'):
        V[(r['arm'], r['suite'], r['fn'])].append(float(r['best']))
    RD = [('printed', 'printed', 'proportional'), ('printed', 'printed', 'inverse'), ('printed', 'gazelle', 'proportional'),
          ('printed', 'gazelle', 'inverse'), ('decreasing', 'printed', 'proportional'), ('decreasing', 'printed', 'inverse'),
          ('decreasing', 'gazelle', 'proportional'), ('decreasing', 'gazelle', 'inverse')]
    rows = []
    for k in range(8):
        a = f'AMBWO@R{k}'; cells = [f'R{k}', *RD[k]]
        for suite in ('cec2017', 'cec2022'):
            fns = sorted({f for (b, s, f) in V if b == a and s == suite}, key=lambda s: int(s[1:]))
            ps, sign = [], []
            for fn in fns:
                v = np.array(V[(a, suite, fn)]); pm, psd = ref(suite, fn)
                t = stats.ttest_ind_from_stats(v.mean(), v.std(ddof=1), 30, pm, psd, 30, equal_var=False)
                ps.append(1.0 if not np.isfinite(t.pvalue) else float(t.pvalue)); sign.append(v.mean() > pm)
            adj = holm(ps)
            same = int(sum(q >= 0.05 for q in adj)); worse = int(sum(q < 0.05 and s for q, s in zip(adj, sign)))
            cells.append(f'{same}/{worse}/{len(fns) - same - worse}')
        rows.append(cells)
    return rows


# ---------------------------------------------------------------- Table 5, bias test
def tab_e1():
    S = json.load(open(os.path.join(OUT, 'e1_summary.json')))
    err = defaultdict(list)
    for r in read('bwo-e1/E1_classical_D30.csv'):
        if r['cond'] == 'Sr':
            err[(r['arm'], r['fn'])].append(float(r['error']))
    fns = [f'F{k}' for k in range(1, 14)]
    order = ['RS', 'PSO', 'DE', 'L-SHADE', 'jSO', 'BIPOP-CMA-ES', 'GWO', 'WOA', 'HHO', 'MPA', 'BWO-P', 'BWO-ML', 'BWO-M',
             'AMBWO', 'BWO-P+E', 'BWO-P+X', 'BWO-P+W', 'BWO-TI']
    rows = []
    for a in order:
        s = S[a]
        w = '--' if a == REF else '/'.join(map(str, wtl({f: err[(REF, f)] for f in fns}, {f: err[(a, f)] for f in fns}, fns)))
        rows.append([nm(a), str(s['center_n']), f"{s['medlog10R']:.2f}", str(s['diag_n']), f"{s['medlog10D']:.2f}", w])
    return rows


# ---------------------------------------------------------------- Tables 6-8, CEC suites and single repairs
def tab_e2():
    R = list(csv.DictReader(open(os.path.join(OUT, 'e2_ranks.csv'))))
    W = list(csv.DictReader(open(os.path.join(OUT, 'e2_wtl.csv'))))
    rank = {(r['suite'], int(r['dim']), r['arm']): float(r['avg_rank']) for r in R}
    w = {(r['suite'], int(r['dim']), r['vs']): (int(r['W']), int(r['T']), int(r['L'])) for r in W}
    arms = sorted({r['arm'] for r in R})
    mean_rank = {a: np.mean([rank[(s, d, a)] for s, d in CELLS]) for a in arms}
    order = sorted([a for a in arms if a != REF], key=lambda a: mean_rank[a]) + [REF]
    t6, t7 = [], []
    for a in order:
        t6.append([nm(a)] + [f'{rank[(s, d, a)]:.2f}' for s, d in CELLS] + [f'{mean_rank[a]:.2f}'])
        if a == REF:
            t7.append([nm(a)] + ['--'] * (len(CELLS) + 1))
        else:
            tot = np.sum([w[(s, d, a)] for s, d in CELLS], axis=0)
            t7.append([nm(a)] + ['/'.join(map(str, w[(s, d, a)])) for s, d in CELLS] + ['/'.join(map(str, tot))])
    S = json.load(open(os.path.join(OUT, 'e1_summary.json')))
    t8 = []
    for a, fix in (('BWO-P', 'none'), ('BWO-P+E', 'exploration'), ('BWO-P+X', 'exploitation'), ('BWO-P+W', 'whale fall'),
                   ('BWO-TI', 'all three')):
        tot = '--' if a == REF else '/'.join(map(str, np.sum([w[(c, d, a)] for c, d in CELLS], axis=0)))
        t8.append([nm(a), fix, str(S[a]['center_n']), str(S[a]['diag_n']), f'{mean_rank[a]:.2f}', tot])
    return t6, t7, t8


# ---------------------------------------------------------------- Table 9, design problems
def e3_data():
    data = defaultdict(list)
    for r in read('bwo-e3-all/E3_rw2020.csv'):
        o = json.loads(r['ops'])
        data[(r['arm'], r['fn'])].append((float(r['best']), float(o['f_best']), float(o['v_best'])))
    return data


def tab_e3(data):
    arms = sorted({k[0] for k in data}); probs = [f'RC{k}' for k in range(15, 34)]
    M = np.array([[np.mean([t[0] for t in data[(a, p)]]) for a in arms] for p in probs])
    Rk = ranks(M).mean(0)
    k, n = len(arms), len(probs); se = math.sqrt(k * (k + 1) / (6 * n)); i0 = arms.index(REF)
    oth = [a for a in arms if a != REF]
    padj = dict(zip(oth, holm([2 * stats.norm.sf(abs(Rk[arms.index(a)] - Rk[i0]) / se) for a in oth])))
    S = np.zeros((n, k))
    for i, p in enumerate(probs):               # feasibility rule over all final points of the problem
        pts = [(a, t) for a in arms for t in data[(a, p)]]
        kv = [(0, t[1]) if t[2] == 0 else (1, t[2]) for _, t in pts]
        order = sorted(range(len(pts)), key=lambda j: kv[j]); rk = np.empty(len(pts)); j = 0
        while j < len(order):
            e = j
            while e + 1 < len(order) and kv[order[e + 1]] == kv[order[j]]:
                e += 1
            for q in order[j:e + 1]:
                rk[q] = (j + e) / 2 + 1
            j = e + 1
        for c, a in enumerate(arms):
            S[i, c] = np.mean([rk[j] for j, (b, _) in enumerate(pts) if b == a])
    Fk = np.array([stats.rankdata(row) for row in S]).mean(0)
    W = {r['vs']: (r['W'], r['T'], r['L']) for r in csv.DictReader(open(os.path.join(OUT, 'e3_wtl.csv')))}
    order = sorted(oth, key=lambda a: Rk[arms.index(a)]) + [REF]
    rows = []
    for a in order:
        j = arms.index(a)
        feas = sum(sum(t[2] == 0 for t in data[(a, p)]) for p in probs)
        allf = sum(all(t[2] == 0 for t in data[(a, p)]) for p in probs)
        pos = list(np.argsort(Rk)).index(j) + 1
        rows.append([nm(a), f'{Rk[j]:.2f}', str(pos), '--' if a == REF else pfmt(padj[a]),
                     '--' if a == REF else '/'.join(W[a]), str(feas), str(allf), f'{Fk[j]:.2f}'])
    return rows


# ---------------------------------------------------------------- supplementary tables
def friedman_order(D, keys):
    arms = sorted({a for a, _ in D})
    Mx = np.array([[np.mean(D[(a, f)]) for a in arms] for f in keys])
    r = ranks(Mx).mean(0)
    return [arms[i] for i in np.argsort(r) if arms[i] != REF] + [REF]


def chunks(x, k):
    n = int(np.ceil(len(x) / k)); s = int(np.ceil(len(x) / n))
    return [x[i:i + s] for i in range(0, len(x), s)]


def supp(data3):
    out = {}; k = 0
    E = defaultdict(list)
    for s, d in CELLS:
        for r in read(f'bwo-e2/E2_{s}_D{d}.csv'):
            e = float(r['error']); E[(s, d, r['arm'], r['fn'])].append(e if e >= 1e-8 else 0.0)
    for s, d in CELLS:                                          # S1-S16, CEC suites
        fns = sorted({x[3] for x in E if x[0] == s and x[1] == d}, key=lambda f: int(f[1:]))
        D = {(a, f): E[(s2, d2, a, f)] for (s2, d2, a, f) in E if (s2, d2) == (s, d)}
        arms = friedman_order(D, fns)
        for cols in chunks(fns, 8):
            k += 1; rows = []
            for a in arms:
                rows.append([nm(a), 'Mean'] + [fmt(np.mean(D[(a, f)])) for f in cols])
                rows.append(['', 'SD'] + [fmt(np.std(D[(a, f)], ddof=1)) for f in cols])
            out[f'S{k}'] = rows
    probs = [f'RC{i}' for i in range(15, 34)]                 # S17-S23, design problems
    arms = friedman_order({kk: [t[0] for t in v] for kk, v in data3.items()}, probs); runs = 25
    for cols in chunks(probs, 5):
        k += 1; rows = []
        for a in arms:
            cells = []
            for c in cols:
                v = [t[2] for t in data3[(a, c)]]; n = sum(x == 0 for x in v)
                cells.append(str(n) if n == runs else f'{n}, {max(v):.2E}' if n > 0 else f'0, {min(v):.2E}')
            rows.append([nm(a), ''] + cells)
        out[f'S{k}'] = rows
    for cols in chunks(probs, 7):
        k += 1; rows = []
        for a in arms:
            rows.append([nm(a), 'Mean'] + ['%.4E' % np.mean([t[1] for t in data3[(a, c)]]) for c in cols])
            rows.append(['', 'SD'] + [fmt(np.std([t[1] for t in data3[(a, c)]], ddof=1)) for c in cols])
        out[f'S{k}'] = rows
    E4 = defaultdict(list)                                    # S24-S27, parameter sensitivity
    for r in read('e4/E4.csv'):
        e = float(r['error']); E4[(int(r['n']), float(r['levy']), int(r['fn']))].append(e if e >= 1e-8 else 0.0)
    confs = [(n, 0.05) for n in (20, 30, 50, 80, 100)] + [(50, l) for l in (0.01, 0.02, 0.1, 0.2)]
    for cols in chunks([f'F{f}' for f in [1] + list(range(3, 31))], 8):
        k += 1; rows = []
        for c in confs:
            v = [E4[c + (int(f[1:]),)] for f in cols]
            rows.append([f'n={c[0]}, Lévy {c[1]}', 'Mean'] + [fmt(np.mean(x)) for x in v])
            rows.append(['', 'SD'] + [fmt(np.std(x, ddof=1)) for x in v])
        out[f'S{k}'] = rows
    C = defaultdict(list)                                     # S28, population statistics
    for r in read('e5/E5.csv'):
        C[(r['arm'], int(r['fn']))].append(np.array(json.loads(r['rec'])))
    grid = np.linspace(0, 300_000, 101)[1:]
    ip = lambda rec, col: np.interp(grid, rec[:, 0], rec[:, col])
    k += 1; rows = []
    for a in ['BWO-P', 'BWO-P+E', 'BWO-P+X', 'BWO-P+W', 'BWO-TI']:
        st = {f: dict(x=float(np.mean(np.median([100 * ip(r, 1) / r[:, 1].max() for r in C[(a, f)]], 0))),
                      dv=float(np.median([ip(r, 1) for r in C[(a, f)]], 0)[-1]),
                      do=float(np.median([ip(r, 2) for r in C[(a, f)]], 0)[-1]),
                      dp=float(np.median([ip(r, 3) for r in C[(a, f)]], 0)[-1])) for f in (1, 5, 10, 22)}
        fs = (1, 5, 10, 22)
        rows.append([nm(a), 'Mean exploration, %'] + ['%.2f' % st[f]['x'] for f in fs])
        rows.append(['', 'Final diversity'] + [fmt(st[f]['dv']) if st[f]['dv'] >= 1e-8 else '%.2E' % st[f]['dv'] for f in fs])
        rows.append(['', 'Centroid to origin'] + ['%.1f' % st[f]['do'] for f in fs])
        rows.append(['', 'Centroid to o'] + ['%.1f' % st[f]['dp'] for f in fs])
    out[f'S{k}'] = rows
    return out


# ---------------------------------------------------------------- Figure 12 values, parameter sensitivity
def fig12():
    E = defaultdict(list)
    for r in read('e4/E4.csv'):
        e = float(r['error']); E[(int(r['n']), float(r['levy']), int(r['fn']))].append(e if e >= 1e-8 else 0.0)
    confs = sorted({k[:2] for k in E}); fns = sorted({k[2] for k in E})
    M = np.array([[np.mean(E[(n, l, f)]) for (n, l) in confs] for f in fns])
    rk = ranks(M).mean(0); base = (50, 0.05); rows = []
    for c, r in zip(confs, rk):
        if c == base:
            w = 'default'
        else:
            ps = [stats.ranksums(E[c + (f,)], E[base + (f,)]).pvalue for f in fns]
            rej = multipletests(ps, method='holm')[0]
            ww = sum(rj and np.median(E[base + (f,)]) < np.median(E[c + (f,)]) for rj, f in zip(rej, fns))
            ll = sum(rj and np.median(E[base + (f,)]) > np.median(E[c + (f,)]) for rj, f in zip(rej, fns))
            w = f'{ww}/{len(fns) - ww - ll}/{ll}'
        rows.append([str(c[0]), str(c[1]), f'{r:.2f}', w])
    p = friedman(M)[1]
    return rows, f'{p:.1e}'


def main():
    os.makedirs(OUT, exist_ok=True); ensure_stats()
    t6, t7, t8 = tab_e2(); d3 = e3_data(); f12, f12p = fig12()
    T = {'Table 4': tab_c1(), 'Table 5': tab_e1(), 'Table 6': t6, 'Table 7': t7, 'Table 8': t8, 'Table 9': tab_e3(d3),
         'Figure 12': f12, 'Figure 12 Friedman p': [[f12p]], **supp(d3)}
    json.dump(T, open(os.path.join(OUT, 'tables.json'), 'w'), indent=1, ensure_ascii=False)
    print('tables rebuilt:', ', '.join(T))


if __name__ == '__main__':
    main()
