"""Coarse flowchart of the order-parameter learning loop (docs/op_loop_report.md).

    python -m loop.flowchart        writes <OUT_ROOT>/op_loop/flowchart.png
"""
import os

import common

OUT = os.path.join(common.OUT_ROOT, 'op_loop', 'flowchart.png')

# fill tints of the report's categorical slots, by phase; text stays in ink
PHASE = {
    'init': ('#e4e3df', '#52514e'),       # initialization (neutral)
    'learn': ('#d5e4f7', '#2a78d6'),      # learning lambda from the stored paths
    'sample': ('#fbe1d3', '#eb6834'),     # sampling with lambda
    'final': ('#d2f0e3', '#1baf7a'),      # production
}
INK, MUTED = '#0b0b0b', '#52514e'

STEPS = [  # key, phase, title, one-line description
    ('pass0', 'init', '0  Pass 0: infinit (inftools)',
     'short RETIS runs with the seed order parameter (e.g. y); adds interfaces until paths reach B'),
    ('pool', 'init', '1  Path pool',
     'all stored paths, cut at the interface below their ensemble; CVs and MC weight per frame'),
    ('teacher', 'learn', '2  Teacher: NPNEq committor',
     'q = P(reach B before A) at every stored frame, from the paths alone (no formula)'),
    ('screen', 'learn', '3  CV screening (optional)',
     'keep the CVs that lower the held-out path energy of the committor'),
    ('student', 'learn', '4  Student: MLP  λ(x)',
     'a smooth formula of the CVs fitted to logit q; works on any new frame (~10 µs)'),
    ('intf', 'sample', '5  Interfaces',
     'planned at committor levels, then the spacing gate (thin / refill) on the stored paths'),
    ('seeds', 'sample', '6  Seed paths',
     'one valid initial path per ensemble, cut from the stored paths under the new λ'),
    ('run', 'sample', '7  Sampling pass: infretis StapleTIS',
     '~1000 cycles with λ as the order parameter; new paths go into the pool'),
    ('stop', 'decision', '8  Converged?',
     'λ stable · Z_q OK · interfaces unchanged · checks pass'),
    ('freeze', 'final', '9  Freeze',
     'λ fixed; short verification runs until the interface set passes the gate on its own paths'),
    ('prod', 'final', '10  Production: StapleTIS',
     'long run with frozen λ, interfaces, seed paths  →  crossing probability and rate'),
]


def draw(path=OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    fig, ax = plt.subplots(figsize=(9.6, 12.8))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.5, 13.2)
    ax.axis('off')
    x0, w, h, dy = 0.6, 8.4, 0.78, 1.18
    ys = {key: 12.4 - i * dy for i, (key, *_) in enumerate(STEPS)}
    cx = x0 + w / 2

    # the loop: steps 2-8 inside a dashed frame
    top, bot = ys['teacher'] + h / 2 + 0.28, ys['stop'] - h / 2 - 0.22
    ax.add_patch(FancyBboxPatch((x0 - 0.35, bot), w + 0.7, top - bot, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fill=False, ls=(0, (4, 3)), lw=1.2, ec=MUTED))
    ax.text(x0 - 0.2, top - 0.05, 'pass k = 1, 2, …', fontsize=9, color=MUTED, va='top', style='italic')

    for key, phase, title, desc in STEPS:
        y = ys[key]
        if phase == 'decision':
            pts = [(cx, y + h / 2 + 0.12), (x0 + w, y), (cx, y - h / 2 - 0.12), (x0, y)]
            ax.add_patch(Polygon(pts, closed=True, fc='#fff6d6', ec='#eda100', lw=1.6))
            ax.text(cx, y + 0.12, title, ha='center', va='center', fontsize=10.5,
                    weight='bold', color=INK)
            ax.text(cx, y - 0.2, desc, ha='center', va='center', fontsize=7.6, color=MUTED)
            continue
        fc, ec = PHASE[phase]
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x0 + 0.18, y + 0.15, title, ha='left', va='center', fontsize=10.5, weight='bold', color=INK)
        ax.text(x0 + 0.18, y - 0.17, desc, ha='left', va='center', fontsize=8, color=MUTED)

    def arrow(p, q, text=None, rad=0.0, tx=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13, lw=1.4, color=MUTED,
                                     connectionstyle=f'arc3,rad={rad}'))
        if text:
            ax.text(*(tx or ((p[0] + q[0]) / 2 + 0.12, (p[1] + q[1]) / 2)), text, fontsize=9, color=INK,
                    weight='bold', va='center')

    keys = [k for k, *_ in STEPS]
    for a, b in zip(keys[:-1], keys[1:]):
        lo = ys[a] - h / 2 - (0.12 if a == 'stop' else 0)
        hi = ys[b] + h / 2 + (0.12 if b == 'stop' else 0)
        arrow((cx, lo), (cx, hi), 'yes' if a == 'stop' else None)
    # "no": from the decision back to the teacher, along the right side
    xr = x0 + w + 0.55
    ax.plot([x0 + w, xr, xr], [ys['stop'], ys['stop'], ys['teacher']], color=MUTED, lw=1.4)
    arrow((xr, ys['teacher']), (x0 + w, ys['teacher']))
    ax.text(xr + 0.08, (ys['stop'] + ys['teacher']) / 2, 'no:\nnext pass', fontsize=9, color=INK, weight='bold',
            va='center')

    # legend of the phases
    lx, ly = x0, -0.45
    for label, phase in (('initialization', 'init'), ('learning λ', 'learn'), ('sampling with λ', 'sample'),
                         ('production', 'final')):
        fc, ec = PHASE[phase]
        ax.add_patch(FancyBboxPatch((lx, ly), 0.35, 0.25, boxstyle='round,pad=0.01,rounding_size=0.05',
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(lx + 0.45, ly + 0.12, label, fontsize=8.5, color=MUTED, va='center')
        lx += 2.05
    ax.set_title('Order-parameter learning loop', fontsize=13, weight='bold', color=INK, loc='left', x=0.08)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


if __name__ == '__main__':
    print(draw())
