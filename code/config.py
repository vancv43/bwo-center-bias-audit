"""Every parameter of the study in one place, each tagged with its source.
Tags: 'paper' = Zhong et al. 2022 KBS 251:109215; 'assumption' = paper silent, our choice;
'library' = mealpy master BWO.py; 'design' = fixed before the experiments; 'cec' = competition rules."""

BWO = {
    'n':            (50,   'paper', 'population size, Sec. 4.1'),
    'levy_scale':   (0.05, 'paper', 'Eq. 6'),
    'levy_beta':    (1.5,  'paper', 'Eq. 7'),
    'wf_start':     (0.1,  'paper', 'Eq. 10'),
    'wf_end':       (0.05, 'paper', 'Eq. 10'),
    'bf_threshold': (0.5,  'paper', 'Sec. 3.3 step 2'),
    'tau':          ('spent_fe_fraction', 'assumption', 'T/Tmax replaced by FE spent / B (FE per iteration varies)'),
    'bounds':       ('clip', 'assumption', 'paper silent'),
    'selection':    ('greedy', 'assumption', 'paper silent; mealpy and Dik (MQL5) greedy'),
    'p1':           ('first entry of the random dimension sequence', 'assumption', 'Eq. 4 gives no other definition'),
    'fall_trigger': ('Bf_i < Wf', 'paper', 'Fig. 2'),
}

AMBWO = {
    'n':            (30,   'paper', 'You et al. 2024 Sec. 4.1.2: population 30'),
    'e_min':        (1.1,  'paper', 'Eq. 13'),
    'e_max':        (2.5,  'paper', 'Eq. 13 (printed formula rises with t; text says decrease; printed formula used)'),
    'alpha1':       (0.5,  'paper', 'Eq. 19'),
    'psr':          (0.34, 'paper', 'Eq. 21 threshold 0.34'),
    'u_rule':       ('U_j = 1 if rand < 0.34', 'assumption', 'Eq. 21 binary U threshold unstated; Gazelle/MPA code rule'),
    'cf':           ('(1 - tau)^(2 - tau)', 'paper', 'Eq. 22 as printed'),
    'levy_scale':   (0.05, 'paper', 'BWO Eq. 6, inherited'),
    'levy_beta':    (1.5,  'paper', 'BWO Eq. 7, inherited'),
    'wf_start':     (0.1,  'paper', 'Eq. 12'),
    'wf_end':       (0.05, 'paper', 'Eq. 12'),
    'bf_threshold': (0.5,  'paper', 'Algorithm 1'),
    'fall_trigger': ('Bf_i < Wf', 'assumption', 'Algorithm 1 line truncated in PMC rendering; BWO Fig. 2 rule'),
    'ref_select':   ('roulette, probability proportional to S_i (Eq. 19)', 'assumption', 'direction of roulette unstated'),
    'mean_cov':     ('better half, computed at iteration start', 'assumption', 'timing unstated'),
    'tau':          ('spent_fe_fraction', 'assumption', 'budget rule'),
    'bounds':       ('clip', 'assumption', 'paper silent'),
    'selection':    ('greedy after every candidate', 'assumption', 'paper silent; BWO code greedy'),
}

E1 = {
    'suite':   ('classical F1-F13, Yao et al. 1999 domains', 'design', ''),
    'dim':     (30, 'design', ''),
    'budget':  (50_000, 'design', 'about paper 50 x 1000'),
    'seeds':   (list(range(30)), 'design', ''),
    'shift_frac': (0.4, 'design', 's_j ~ U(-0.4 h, 0.4 h), h = (ub - lb)/2'),
    'shift_seed_offset': (10_000, 'design', ''),
    'ratio_excluded': (['F8'], 'design', 'optimum not at the centre'),
    'conds': (['U', 'Sr', 'Sd'], 'design', 'unshifted, random shift, diagonal shift'),
}

E2 = {
    'cec2017_dims': ([10, 30, 50], 'design', 'D=100 conditional on timing'),
    'cec2017_budget_per_dim': (10_000, 'cec', 'B = 10000 D'),
    'cec2017_runs': (51, 'cec', ''),
    'cec2022_budget': ({10: 200_000, 20: 1_000_000}, 'cec', ''),
    'cec2022_runs': (30, 'cec', ''),
}

E3 = {   # mechanical group of CEC2020-RW, as in an earlier benchmark study
    'problems': ([f'RC{k}' for k in range(15, 34)], 'design', 'RC15-RC33, official C++ code (Stanovov port)'),
    'runs': (25, 'cec', 'CEC2020-RW rule'),
    'budget_small': (100_000, 'cec', 'D <= 10'),
    'budget_large': (200_000, 'cec', '10 < D <= 30'),
    'penalty': (1e6, 'assumption', 'static penalty f + 1e6 v, as in an earlier benchmark study; feasibility-rule ranking secondary'),
    'eq_tol': (1e-4, 'cec', 'equality tolerance'),
}


def flat(d):
    return {k: v[0] for k, v in d.items()}


def tags(d):
    return {k: v[1] for k, v in d.items()}
