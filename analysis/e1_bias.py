"""E1 bias analysis, rules fixed before the data. Reads every E1 csv; writes
analysis/out/e1_bias.csv and e1_summary.json. No row, seed or arm dropped."""
import glob, json, sys
import numpy as np, pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
fs = sorted(glob.glob('results/bwo-e1/*.csv'))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
assert (df.smoke == 0).all() and (df.fe_used <= df.budget).all()
cnt = df.groupby(['arm', 'fn', 'cond']).size()
print('files', len(fs), 'rows', len(df), 'arms', df.arm.nunique(), 'runs/cell min-max', cnt.min(), cnt.max())
E = 1e-8
RF = [f'F{i}' for i in range(1, 14) if i != 8]
rows = []
for arm, g in df.groupby('arm'):
    for fn in sorted(g.fn.unique(), key=lambda s: int(s[1:])):
        h = g[g.fn == fn]
        e = {c: h[h.cond == c].sort_values('seed').error.to_numpy() for c in ('U', 'Sr', 'Sd')}
        m = {c: float(np.median(v)) for c, v in e.items()}
        rows.append(dict(arm=arm, fn=fn, med_U=m['U'], med_Sr=m['Sr'], med_Sd=m['Sd'],
                         R=(m['Sr'] + E) / (m['U'] + E), D=(m['Sr'] + E) / (m['Sd'] + E),
                         p_R=ranksums(e['U'], e['Sr']).pvalue, p_D=ranksums(e['Sr'], e['Sd']).pvalue))
T = pd.DataFrame(rows)
summ = {}
for arm, g in T[T.fn.isin(RF)].groupby('arm'):
    for k in ('R', 'D'):
        T.loc[g.index, 'h' + k] = multipletests(g['p_' + k].fillna(1), method='holm')[1]
for arm, g in T[T.fn.isin(RF)].groupby('arm'):
    nR = int(((g.R > 10) & (g.hR < .05)).sum()); nD = int(((g.D > 10) & (g.hD < .05)).sum())
    summ[arm] = dict(center_n=nR, center_biased=nR >= 3, diag_n=nD, diag_biased=nD >= 3,
                     medlog10R=float(np.median(np.log10(g.R))), medlog10D=float(np.median(np.log10(g.D))))
T.to_csv('analysis/out/e1_bias.csv', index=False)
json.dump(summ, open('analysis/out/e1_summary.json', 'w'), indent=1)
S = pd.DataFrame(summ).T.sort_values('center_n')
print(S.to_string())
