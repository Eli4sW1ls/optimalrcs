"""Summary figures of the 2D Brownian Z plateau investigation (notes section 9.2, results overview II.3):
1. summary_offsets.png: plateau (fit − exact) of every fit, halves joined by a line;
2. summary_maps.png: fit − exact on the potential for the main cases (from cases/<case>/map.npz);
3. summary_tests.png: martingale test at first band entry, and how many plateau pieces reach A or B.
Plateau values come from cases/*.log (case_plots.py) where available, else from the run logs (fixed below).
"""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
C = HERE + 'cases/'

val = {}
for log in glob.glob(C + '*.log'):
    for m in re.finditer(r'^(\S+)\s+RMS ([\d.]+)\s+plateau ([+-][\d.]+)', open(log).read(), re.M):
        val[m.group(1)] = float(m.group(3))
FALLBACK = {'synth_full': -0.007, 'synth_cut_all': -0.015, 'synth_cut_even': -0.040, 'synth_cut_odd': 0.009,
            'synth_cut_even_lag10': -0.031, 'synth_cut_odd_lag10': 0.031, 'staple_both_all': 0.031,
            'staple_both_even': 0.049, 'staple_both_odd': 0.014, 'staple_both_even_lag50': 0.034,
            'staple_both_odd_lag50': 0.016, 'staple_both_2250_lag50': 0.022, 'staple_forward': 0.061,
            'md_npq_all': 0.060, 'md_npne_s1-8_lag1': 0.029, 'md_npne_s1-8_lag10': 0.031, 'md_npne_s9-16_lag10': 0.079,
            'md_npne_s1-8_lag50': 0.027}
v = lambda k: val.get(k, FALLBACK[k])

# --- 1. plateau offsets ---------------------------------------------------------------------------
rows = [  # (group, label, keys: one value or a pair of halves)
    ('synthetic walkers\n(exact answer known)', 'run until absorbed', ['synth_full']),
    ('synthetic walkers\n(exact answer known)', 'cut at interfaces, all', ['synth_cut_all']),
    ('synthetic walkers\n(exact answer known)', 'cut, halves, lag 1', ['synth_cut_even', 'synth_cut_odd']),
    ('synthetic walkers\n(exact answer known)', 'cut, halves, lag 10', ['synth_cut_even_lag10', 'synth_cut_odd_lag10']),
    ('staple TIS', 'forward pieces only, 3000/ens.', ['staple_forward']),
    ('staple TIS', 'both pieces, 1500/ens., lag 1', ['staple_both_all']),
    ('staple TIS', 'both, halves, lag 1', ['staple_both_even', 'staple_both_odd']),
    ('staple TIS', 'both, halves, lag 50', ['staple_both_even_lag50', 'staple_both_odd_lag50']),
    ('staple TIS', 'both pieces, 2250/ens., lag 50', ['staple_both_2250_lag50']),
    ('long MD', 'NPq, 16 seeds', ['md_npq_all']),
    ('long MD', 'NPNEq seeds 1-8, lag 1', ['md_npne_s1-8_lag1']),
    ('long MD', 'NPNEq seeds 1-8, lag 50', ['md_npne_s1-8_lag50']),
    ('long MD', 'NPNEq lag 10, seeds 1-8 / 9-16', ['md_npne_s1-8_lag10', 'md_npne_s9-16_lag10']),
]
col = {'synthetic walkers\n(exact answer known)': 'tab:green', 'staple TIS': 'tab:blue', 'long MD': 'tab:purple'}
fig, ax = plt.subplots(figsize=(9, 7.6))
ypos = np.arange(len(rows))[::-1].astype(float)
for yy, (g, lab, keys) in zip(ypos, rows):
    xs = [v(k) for k in keys]
    if len(xs) == 2:
        ax.plot(xs, [yy, yy], '-', color=col[g], lw=2, alpha=0.5)
        ax.plot(xs, [yy, yy], 'o', mfc='white', color=col[g], ms=8)
        ax.plot(np.mean(xs), yy, '|', color=col[g], ms=14, mew=2)
    else:
        ax.plot(xs, [yy], 'o', color=col[g], ms=9)
    ax.text(0.105, yy, '  ' + ' / '.join(f'{x_:+.3f}' for x_ in xs), va='center', fontsize=8.5)
ax.set_yticks(ypos, [r[1] for r in rows])
ax.axvline(0, color='k', lw=1)
ax.axvspan(0.02 - 0.015, 0.02 + 0.015, color='tab:blue', alpha=0.08, label='overall estimate for the staple fits, +0.02 ± 0.015')
for g, c in col.items():
    ax.plot([], [], 'o', color=c, label=g.replace('\n', ' '))
ax.plot([], [], 'o-', mfc='white', color='0.4', label='two independent halves (| = their mean)')
ax.set(xlim=(-0.06, 0.15), xlabel='mean(fit − exact committor) on the plateau (exact q 0.55–0.85)',
       title='2D Brownian Z: plateau level of every fit')
for b in (3.5, 8.5):
    ax.axhline(len(rows) - 1 - b, color='0.8', lw=0.8)
ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.09), ncol=2, fontsize=8, frameon=False); ax.grid(axis='x', alpha=0.3)
fig.tight_layout(); fig.savefig(HERE + 'summary_offsets.png', dpi=110); plt.close(fig)

# --- 2. fit − exact maps -----------------------------------------------------------------------------
MAPS = ['synth_full', 'synth_cut_all', 'synth_cut_even', 'synth_cut_odd',
        'staple_forward', 'staple_both_all', 'staple_both_even', 'staple_both_odd',
        'staple_both_2250_lag50', 'md_npq_all', 'md_npne_s1-8_lag10', 'md_npne_s9-16_lag10']
have = [k for k in MAPS if os.path.exists(C + k + '/map.npz')]
nc = 4; nr = int(np.ceil(len(have) / nc))
SHORT = dict(zip(MAPS, ['synthetic, run until absorbed', 'synthetic, cut at interfaces', 'synthetic cut, even half',
                         'synthetic cut, odd half', 'staple, forward pieces only', 'staple, both pieces, lag 1',
                         'staple, both, even half', 'staple, both, odd half', 'staple, both, 2250/ens., lag 50',
                         'MD NPq, 16 seeds', 'MD NPNEq lag 10, seeds 1-8', 'MD NPNEq lag 10, seeds 9-16']))
fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.1 * nr), squeeze=False, constrained_layout=True)
for ax, k in zip(axes.flat, have):
    d = np.load(C + k + '/map.npz')
    im = ax.imshow((d['fit'] - d['exact']).T, origin='lower', extent=[d['ey'][0], d['ey'][-1], d['ex'][0], d['ex'][-1]],
                   aspect='auto', cmap='PuOr', vmin=-0.1, vmax=0.1)
    ax.set_title(f"{SHORT[k]}\nplateau {float(d['plateau']):+.3f}, RMS {float(d['rms']):.3f}", fontsize=8.5)
    ax.set(xlim=(0.04, 0.86), ylim=(-0.01, 0.31)); ax.tick_params(labelsize=7)
for ax in axes.flat[len(have):]:
    ax.axis('off')
fig.colorbar(im, ax=axes, shrink=0.6, label='fit − exact committor')
fig.supxlabel('order parameter y'); fig.supylabel('x')
fig.savefig(HERE + 'summary_maps.png', dpi=100); plt.close(fig)

# --- 3. martingale test and absorbed fraction (synth/martingale_*.log) -------------------------------
bands = ['0.55–0.7', '0.7–0.85']
tests = [  # label, (E[D], s.e.) per band
    ('synthetic walkers', [(-0.0047, 0.0043), (-0.0031, 0.0038)]),
    ('staple, rejection w.,\nboth pieces', [(0.0015, 0.0012), (0.0010, 0.0018)]),
    ('staple, rejection w.,\nforward pieces', [(0.0041, 0.0016), (0.0036, 0.0024)]),
    ('staple, rejection w.,\nmirrored pieces', [(-0.0010, 0.0017), (-0.0016, 0.0024)]),
    ('staple weights,\nboth pieces', [(-0.0003, 0.0012), (-0.0111, 0.0018)]),
    ('fitted q (+0.031)\ninstead of exact', [(0.0001, 0.0012), (0.0005, 0.0016)]),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[2.3, 1]))
w = 0.38; xx = np.arange(len(tests))
for j, (b, c) in enumerate(zip(bands, ['tab:orange', 'tab:red'])):
    m = [t[1][j][0] for t in tests]; e = [t[1][j][1] for t in tests]
    a1.bar(xx + (j - 0.5) * w, m, w, yerr=e, capsize=3, color=c, alpha=0.75, label=f'band q {b}')
a1.axhline(0, color='k', lw=0.8); a1.set_xticks(xx, [t[0] for t in tests], fontsize=8.5)
a1.set(ylabel='E[q(stop) − q(first entry)]', title='Martingale test at first band entry (0 for unbiased data, ±1 s.e.)')
a1.legend(frameon=False); a1.grid(axis='y', alpha=0.3)
a2.bar(bands, [2.9, 19.3], color=['tab:orange', 'tab:red'], alpha=0.75)
for i, f in enumerate([2.9, 19.3]):
    a2.text(i, f + 0.5, f'{f:.1f}%', ha='center')
a2.set(ylabel='% of pieces (weighted)', ylim=(0, 25), xlabel='band entered (exact q)',
       title='Staple pieces entering the plateau\nthat reach A or B before they end')
fig.tight_layout(); fig.savefig(HERE + 'summary_tests.png', dpi=110); plt.close(fig)
print('missing case values (fallback used):', sorted(set(FALLBACK) - set(val)))
print('maps:', have)
