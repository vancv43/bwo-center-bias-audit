"""Check every printed table value of the paper and its supplementary material against values rebuilt from results/.

  python verify_results.py        -> prints a summary, writes reported/verification_report.md, exits 1 on any mismatch

The printed values are in reported/reported_values.json, copied from the submitted tables. The rebuilt values come
from analysis/reproduce_tables.py, which reads only the per-run results. Both sides are compared as printed strings.
"""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(ROOT, 'analysis', 'reproduce_tables.py')], cwd=ROOT, check=True)
rep = json.load(open(os.path.join(ROOT, 'reported', 'reported_values.json')))
new = json.load(open(os.path.join(ROOT, 'analysis', 'out', 'tables.json')))
norm = lambda c: c.replace('–', '--').strip()
lines = ['# Verification report', '', '| Table | Values checked | Matching | Mismatches |', '|---|---|---|---|']
total = ok = 0; bad = []
for t, rows in rep.items():
    got = new.get(t, []); n = m = 0
    for i, r in enumerate(rows):
        g = got[i] if i < len(got) else []
        for j, c in enumerate(r):
            n += 1; v = g[j] if j < len(g) else None
            if v is not None and norm(v) == norm(c):
                m += 1
            else:
                bad.append(f'{t} row {i + 1} col {j + 1}: printed {c!r}, rebuilt {v!r}')
    total += n; ok += m
    lines.append(f'| {t} | {n} | {m} | {n - m} |')
lines += ['', f'**{ok} of {total} printed values match.**', ''] + [f'- {b}' for b in bad[:200]]
open(os.path.join(ROOT, 'reported', 'verification_report.md'), 'w').write('\n'.join(lines) + '\n')
print(f'{ok} of {total} printed values match ({len(rep)} tables).')
for b in bad[:20]:
    print('MISMATCH', b)
sys.exit(1 if bad else 0)
