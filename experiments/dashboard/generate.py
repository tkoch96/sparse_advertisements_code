"""SCULPTOR experiments dashboard generator.

Static-site generator for a localhost dashboard: LEFT SIDEBAR with one
tab per experiment; each tab shows the experiment's figure(s) plus an
arm x N table of per-seed own-objective scores, every value linking to
that run's convergence-over-iterations PDF when harvested.

Architecture (Tom, 2026-08-14):
  EXPERIMENTS  registry below -- one entry per experiment/tab. Each
               entry picks a RENDERER ('objective_ladder' today) and
               may carry presentation OVERRIDES (value format, extra
               intro HTML, figure list). Adding a new experiment = one
               registry entry (+ a renderer if it's a new shape).
  Data         cell scores come from a scores JSON produced by the
               experiment's scoring script (see score_hardb3.py);
               convergence figures from the experiment's artifacts dir.
  Output       <repo>/dashboard_site/index.html (+ symlinks figs/,
               plots/). Serve with: python3 -m http.server 8643
               --directory dashboard_site  (or .claude/launch.json
               'hardb3-dash'). Regenerate any time; reload browser.

Usage:
    python -m experiments.dashboard.generate
See experiments/dashboard/README.md.
"""
import glob
import json
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SITE = os.path.join(REPO, 'dashboard_site')
NS = [1, 2, 5, 10, 20, 50]
SEEDS = [1, 2, 3, 4, 5]
ARMS = [('fixed', 'no_mc', 'fixed', 'L1 no_mc+fixed'),
        ('sched', 'no_mc', 'scheduled', 'L2 no_mc+sched'),
        ('sched', 'no_direction', 'scheduled', 'L3 no_dir+sched'),
        ('sched', 'no_memory', 'scheduled', 'L4 no_mem+sched'),
        ('smart', 'no_memory', 'smart', 'L5 no_mem+smart'),
        ('smart', 'full', 'smart', 'L6 full+smart')]

# ---------------------------------------------------------------------------
# Experiment registry. kind='objective_ladder' entries are hardB3 tabs;
# 'static' entries just render figures + html. Per-experiment overrides:
#   fmt      : python format spec for cell values
#   intro    : extra HTML paragraph under the tab title
#   figures  : list of repo-relative PNGs shown above the table
# ---------------------------------------------------------------------------
HARDB3_STORE = 'cache/model_error/hardB3_scores.json'
HARDB3_FIGS = 'cache/ablation/hardB3_artifacts/figs'

EXPERIMENTS = [
    {'id': 'ablation_v3', 'title': 'Ablation: policy ladder',
     'sections': [
         {'id': 'ladder_v3', 'title': '7-arm ladder v3',
          'kind': 'ladder_links',
          'figs_dir': 'cache/ablation/policy_ladder_v3_artifacts/figs',
          'figs_url': 'figs_ladder3',
          'fixed_all_n': True,
          'arms': [
              ('fixed', 'no_mc', 'fixed', 'L1 no_mc+fixed',
               'figs_ladder3', 'cache/ablation/policy_ladder_v3_artifacts/figs', 'L1_'),
              ('sched', 'no_mc', 'scheduled', 'L2 no_mc+sched',
               'figs_ladder3', 'cache/ablation/policy_ladder_v3_artifacts/figs', 'L2_'),
              ('sched', 'no_memory', 'scheduled', 'L3 no_mem+sched',
               'figs_ladder3', 'cache/ablation/policy_ladder_v3_artifacts/figs', 'L3_'),
              ('sched', 'no_direction', 'scheduled', 'L4 no_dir+sched',
               'figs_ladder3', 'cache/ablation/policy_ladder_v3_artifacts/figs', 'L4_'),
              ('sched', 'full', 'scheduled', 'L5 full+sched',
               'figs_ladder3', 'cache/ablation/policy_ladder_v3_artifacts/figs', 'L5_'),
              ('sched', 'full', 'slotted', 'L6 slotted WHEN',
               'figs_ladder3', 'cache/ablation/policy_ladder_v3_artifacts/figs', 'L6_'),
          ],
          'progress_manifest': 'tools/v3grid_manifest.json',
          'heading': 'Policy ladder v3 — feature ladder L1-L6, '
                     'avg_latency objective (10 deployments)',
          'figures': ['figures/policy_ladder_v3_5panel_objective.png'],
          'refresh': {
              'pull': [('cache/ablation/policy_ladder_v3/',
                        'cache/ablation/policy_ladder_v3/'),
                       ('cache/ablation/policy_ladder_v3_artifacts/figs/',
                        'cache/ablation/policy_ladder_v3_artifacts/figs/')],
              'steps': [
                  {'in': ['cache/ablation/policy_ladder_v3/*/N*/'
                          'seed_*_*.json'],
                   'out': ['cache/model_error/steady/'
                           'policy_steady_v3.json'],
                   'world': 'georand',
                   'argv': ['{py}', '-m',
                            'experiments.model_error.steady_metrics',
                            '--dirs',
                            'AUTO:cache/ablation/policy_ladder_v3',
                            '--tag', 'policy_steady_v3',
                            '--seeds', '1-5']},
                  {'in': ['cache/ablation/policy_ladder_v3/*/N*/'
                          'seed_*_*.json'],
                   'out': ['cache/model_error/failure/'
                           'policy_failure_v3.json'],
                   'every': 4,
                   'world': 'georand',
                   'argv': ['{py}', '-m',
                            'experiments.model_error.failure_metrics',
                            '--dirs',
                            'AUTO:cache/ablation/policy_ladder_v3',
                            '--tag', 'policy_failure_v3',
                            '--seeds', '1-5', '--jobs', '4']},
                  {'in': ['cache/model_error/steady/'
                          'policy_steady_v3.json'],
                   'out': ['figures/policy_ladder_v3_5panel.png',
                           'figures/policy_ladder_v3_5panel_objective.png',
                           'figures/policy_ladder_v3_5panel_iters.png'],
                   'always': True,
                   'env': {'POLICY_PLOT_STAT': 'mean',
                           'POLICY_PLOT_OUT': 'policy_ladder_v3_5panel'},
                   'argv': ['{py}', '-m',
                            'experiments.model_error.plot_policy5']},
              ],
          },
          'intro': 'One rung per capability: L1 fixed probing '
                   '&rarr; L2 scheduled probing &rarr; L3 belief LP '
                   '&rarr; L4 memory &rarr; L5 direction+explore '
                   '&rarr; L6 slotted probe timing (even mean '
                   'rate, surprise-biased within slots). Lower = better; '
                   'opp and painter are reference lines. L5+ train '
                   'up to 500 iters with convergence-based early '
                   'exit; probes are budgeted to N per run.'},
     ]},
    {'id': 'ablation_v3lbc', 'title': 'Policy ladder: LB cache ON',
     'sections': [
         {'id': 'ladder_v3lbc', 'title': '7-arm ladder v3',
          'kind': 'ladder_links',
          'figs_dir': 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs',
          'figs_url': 'figs_ladder3lbc',
          'fixed_all_n': True,
          'arms': [
              ('fixed', 'no_mc', 'fixed', 'L1 no_mc+fixed',
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L1_'),
              ('sched', 'no_mc', 'scheduled', 'L2 no_mc+sched',
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L2_'),
              ('sched', 'no_memory', 'scheduled', 'L3 no_mem+sched',
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L3_'),
              ('sched', 'no_direction', 'scheduled', 'L4 no_dir+sched',
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L4_'),
              ('sched', 'full', 'scheduled', 'L5 full+sched',
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L5_'),
              ('sched', 'full', 'scheduled', "L6' decision WHAT",
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L6p_'),
              ('smart', 'full', 'smart', 'L7 +smart WHEN',
               'figs_ladder3lbc', 'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs', 'L7_'),
          ],
          'progress_manifest': 'tools/v3lbc_manifest.json',
          'progress_live': False,
          'heading': 'Policy ladder v3 + LB CACHE ON — same grid, '
                     'cache A/B',
          'figures': ['figures/policy_ladder_v3lbc_5panel_objective.png'],
          'refresh': {
              'pull': [('cache/ablation/policy_ladder_v3_LBCACHE/',
                        'cache/ablation/policy_ladder_v3_LBCACHE/'),
                       ('cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs/',
                        'cache/ablation/policy_ladder_v3_LBCACHE_artifacts/figs/')],
              'steps': [
                  {'in': ['cache/ablation/policy_ladder_v3_LBCACHE/*/N*/'
                          'seed_*_*.json'],
                   'out': ['cache/model_error/steady/'
                           'lbc_steady_v3.json'],
                   'world': 'georand',
                   'argv': ['{py}', '-m',
                            'experiments.model_error.steady_metrics',
                            '--dirs',
                            'AUTO:cache/ablation/policy_ladder_v3_LBCACHE',
                            '--tag', 'lbc_steady_v3',
                            '--seeds', '1-5']},
                  {'in': ['cache/ablation/policy_ladder_v3_LBCACHE/*/N*/'
                          'seed_*_*.json'],
                   'out': ['cache/model_error/failure/'
                           'lbc_failure_v3.json'],
                   'every': 4,
                   'world': 'georand',
                   'argv': ['{py}', '-m',
                            'experiments.model_error.failure_metrics',
                            '--dirs',
                            'AUTO:cache/ablation/policy_ladder_v3_LBCACHE',
                            '--tag', 'lbc_failure_v3',
                            '--seeds', '1-5', '--jobs', '4']},
                  {'in': ['cache/model_error/steady/'
                          'lbc_steady_v3.json'],
                   'out': ['figures/policy_ladder_v3lbc_5panel.png',
                           'figures/policy_ladder_v3lbc_5panel_objective.png',
                           'figures/policy_ladder_v3lbc_5panel_iters.png'],
                   'always': True,
                   'env': {'POLICY_PLOT_STAT': 'mean',
                           'POLICY_PLOT_TAG_PREFIX': 'lbc',
                           'POLICY_PLOT_OUT': 'policy_ladder_v3lbc_5panel'},
                   'argv': ['{py}', '-m',
                            'experiments.model_error.plot_policy5']},
              ],
          },
          'intro': 'One rung per capability: L1 fixed probing '
                   '&rarr; L2 scheduled probing &rarr; L3 belief LP '
                   '&rarr; L4 memory &rarr; L5 direction+explore '
                   '&rarr; L6 slotted probe timing (even mean '
                   'rate, surprise-biased within slots). Lower = better; '
                   'opp and painter are reference lines. L5+ train '
                   'up to 500 iters with convergence-based early '
                   'exit; probes are budgeted to N per run.'},
     ]},
    {'id': 'hardobj_v2', 'title': 'Ablation: hard objectives',
     'sections': [
         {'id': 'overview', 'title': 'overview', 'kind': 'static',
          'refresh': {
              'remote_harvest': 'REPO=/home/ubuntu/sparse_advertisements_code; A=$REPO/cache/ablation/hardB3v2_artifacts; mkdir -p $A/figs $A/logs; for ws in /home/ubuntu/hb3v2_ws_* /home/ubuntu/hb3v2R2_ws_*; do [ -d "$ws" ] || continue; case "$ws" in *smk*) continue;; esac; lab=$(basename $ws | sed -e s/hb3v2R2_ws_// -e s/hb3v2_ws_//); for d in $ws/S*/runs/ablation-*; do [ -d "$d" ] || continue; cp $d/convergence_over_iterations.pdf $A/figs/${lab}_$(basename $d | sed -e s/^ablation-small-//).pdf 2>/dev/null; rm -rf "$d"; done; done; A2=$REPO/cache/ablation/hardB3v2_L7K3_artifacts; mkdir -p $A2/figs; for ws in /home/ubuntu/l7k3_ws_fracb /home/ubuntu/l7k3_ws_mlu /home/ubuntu/l7k3_ws_prio; do [ -d "$ws" ] || continue; lab=$(basename $ws | sed s/l7k3_ws_//); for d in $ws/S*/runs/ablation-*; do [ -d "$d" ] || continue; cp $d/convergence_over_iterations.pdf $A2/figs/${lab}_$(basename $d | sed -e s/^ablation-small-//).pdf 2>/dev/null; rm -rf "$d"; done; done',
              'pull': [('cache/ablation/hardB3v2/',
                        'cache/ablation/hardB3v2/'),
                       ('cache/ablation/hardB3v2_artifacts/figs/',
                        'cache/ablation/hardB3v2_artifacts/figs/'),
                       ('cache/ablation/hardB3v2_L7K3/',
                        'cache/ablation/hardB3v2_L7K3/'),
                       ('cache/ablation/hardB3v2_L7K3_artifacts/figs/',
                        'cache/ablation/hardB3v2_L7K3_artifacts/figs/')],
              'steps': [
                  {'in': ['cache/ablation/hardB3v2/*/*/N*/'
                          'seed_*_*.json'],
                   'out': ['cache/model_error/hardB3v2_scores.json'],
                   'argv': ['{py}',
                            'experiments/dashboard/score_hardb3.py',
                            '--jobs', '4',
                            '--root', 'cache/ablation/hardB3v2',
                            '--store',
                            'cache/model_error/hardB3v2_scores.json']},
                  {'in': ['cache/ablation/hardB3v2_L7K3/*/*/N*/'
                          'seed_*_*.json'],
                   'out': ['cache/model_error/hardB3v2_L7K3_scores.json'],
                   'argv': ['{py}',
                            'experiments/dashboard/score_hardb3.py',
                            '--jobs', '4',
                            '--root', 'cache/ablation/hardB3v2_L7K3',
                            '--store',
                            'cache/model_error/hardB3v2_L7K3_scores.json']},
                  {'in': ['cache/model_error/hardB3v2_scores.json',
                          'cache/model_error/hardB3v2_L7K3_scores.json'],
                   'out': ['figures/hardB3v2_ladders_incremental.png',
                           'figures/hardB3v2_mlu_util.png'],
                   'always': True,
                   'env': {'HARDB3_STORE':
                           'cache/model_error/hardB3v2_scores.json',
                           'HARDB3_EXTRA_STORE':
                           'cache/model_error/hardB3v2_L7K3_scores.json',
                           'HARDB3_ROOT_RE': 'cache/ablation/hardB3v2',
                           'HARDB3_FIG_PREFIX': 'hardB3v2'},
                   'argv': ['{py}',
                            'experiments/dashboard/plot_hardb3.py']},
              ],
          },
          'heading': 'Hard objectives: monotone ladder + full fix stack, '
                     'all objectives',
          'figures': ['figures/hardB3v2_ladders_incremental.png'],
          'intro': 'The monotone ladder (L1 budgeted-fixed &rarr; L6 '
                   'full+smart, same semantics as policy ladder v2) '
                   'trained AND scored per hard objective under the '
                   'full fix stack (explore MC, combined-U gate, no '
                   'exit-on-budget, corrected soft objective). gamma=0; '
                   'objectives sequential fracb &rarr; mlu &rarr; prio, '
                   'deployment-major within each. Painter refs reused '
                   'from hardB3 (painter is unaffected by the solver '
                   'fixes; rescored under current code). Dataset '
                   'cache/ablation/hardB3v2.'},
         {'id': 'fracb', 'title': 'frac_beyond 10ms',
          'kind': 'objective_ladder',
          'store': 'cache/model_error/hardB3v2_scores.json',
          'figs_dir': 'cache/ablation/hardB3v2_artifacts/figs',
          'figs_url': 'figs_hb3v2', 'world': 'georand',
          'fixed_all_n': True, 'arms': [
              ('fixed', 'no_mc', 'fixed', 'L1 no_mc+fixed'),
              ('sched', 'no_mc', 'scheduled', 'L2 no_mc+sched'),
              ('sched', 'no_memory', 'scheduled', 'L3 no_mem+sched'),
              ('smart', 'no_memory', 'smart', 'L4 no_mem+smart'),
              ('smart', 'no_direction', 'smart', 'L5 no_dir+smart'),
              ('smart', 'full', 'smart', 'L6 full+smart'),
              ('smartL7', 'full', 'smart', 'L7 bern-K3+smart'),
          ],
          'extra_store': 'cache/model_error/hardB3v2_L7K3_scores.json',
          'figures': ['figures/hardB3v2_fracb.png'], 'fmt': '{:.3f}',
          'intro': 'CAPABILITY metric (2026-08-16): min achievable '
                   'volume-weighted excess-ms beyond (per-UG optimal + '
                   '10ms) over the advertisement\'s ingress options '
                   '(hinge LP, canonical Gurobi home; negated, higher = '
                   'better). Monotone in the option set, so '
                   'one-per-peering is an EXACT floor — the old '
                   'assignment-derived fraction let constrained arms '
                   '"beat" opp (25 cells, caught by Tom); it remains in '
                   'the store as the frac_beyond component. Arms still '
                   'TRAINED on the assignment-derived form (uniform '
                   'across all arms, so the ladder comparison is '
                   'fair).'},
         {'id': 'mlu', 'title': 'pure MLU',
          'kind': 'objective_ladder',
          'store': 'cache/model_error/hardB3v2_scores.json',
          'figs_dir': 'cache/ablation/hardB3v2_artifacts/figs',
          'figs_url': 'figs_hb3v2', 'world': 'georand',
          'fixed_all_n': True, 'arms': [
              ('fixed', 'no_mc', 'fixed', 'L1 no_mc+fixed'),
              ('sched', 'no_mc', 'scheduled', 'L2 no_mc+sched'),
              ('sched', 'no_memory', 'scheduled', 'L3 no_mem+sched'),
              ('smart', 'no_memory', 'smart', 'L4 no_mem+smart'),
              ('smart', 'no_direction', 'smart', 'L5 no_dir+smart'),
              ('smart', 'full', 'smart', 'L6 full+smart'),
              ('smartL7', 'full', 'smart', 'L7 bern-K3+smart'),
          ],
          'extra_store': 'cache/model_error/hardB3v2_L7K3_scores.json',
          'figures': ['figures/hardB3v2_mlu_util.png',
                      'figures/hardB3v2_mlu.png'], 'fmt': '{:.3f}',
          'intro': 'routed_lat + P*bad_frac + alpha*(MLU + bad_frac) '
                   '(Tom 2026-08-15 late). MLU = BEST-ACHIEVABLE peak '
                   'utilization for the advertisement (min-Y LP over '
                   'its per-prefix ingress options) — monotone, so '
                   'one-per-peering is a hard floor (<= 1/1.1 = 0.909 '
                   'by anycast provisioning) and stranding cannot '
                   'help. Stranded volume is charged bounded '
                   'penalties (P=50ms + alpha), never the 30s '
                   'sentinel. The pure-MLU era (gameable by stranding '
                   'via the path-dropping fallback LP) is quarantined '
                   '(PUREMLU_STRANDING_ERA_mlu, head).'},
         {'id': 'prio', 'title': 'joint priority',
          'kind': 'objective_ladder',
          'store': 'cache/model_error/hardB3v2_scores.json',
          'figs_dir': 'cache/ablation/hardB3v2_artifacts/figs',
          'figs_url': 'figs_hb3v2', 'world': 'georand',
          'fixed_all_n': True, 'arms': [
              ('fixed', 'no_mc', 'fixed', 'L1 no_mc+fixed'),
              ('sched', 'no_mc', 'scheduled', 'L2 no_mc+sched'),
              ('sched', 'no_memory', 'scheduled', 'L3 no_mem+sched'),
              ('smart', 'no_memory', 'smart', 'L4 no_mem+smart'),
              ('smart', 'no_direction', 'smart', 'L5 no_dir+smart'),
              ('smart', 'full', 'smart', 'L6 full+smart'),
              ('smartL7', 'full', 'smart', 'L7 bern-K3+smart'),
          ],
          'extra_store': 'cache/model_error/hardB3v2_L7K3_scores.json',
          'figures': ['figures/hardB3v2_prio.png'], 'fmt': '{:.2f}',
          'intro': 'Joint latency + bulk-download priority objective '
                   '(negated). CAVEAT (2026-08-16, annotated on the '
                   'panel): this metric is TWO-STAGE assignment-derived '
                   '— the low-latency split is the avg_latency '
                   'optimum, not prio-optimal — so a constrained arm '
                   'can legitimately cross the opp line by small '
                   'margins; its capability twin is a nonconvex QP '
                   '(oversubscribe x significance goes bilinear when '
                   'co-optimized). Resolution options pending Tom: '
                   '(a) redefine to a jointly-LINEAR objective '
                   '(co-optimizable, exact opp floor), or (b) keep and '
                   'accept the documented exemption.'},
     ]},
]


def load_scores(store_rel, extra_rel=None):
    """Parse a scores store into the (obj, pdir, rung, seed, N) grid.
    extra_rel (Tom 2026-08-16: L7 on the same page as L1-L6): a second
    store merged under the VIRTUAL pdir 'smartL7' — L7 cells share
    (smart, full) with L6, so only the path root distinguishes them."""
    grid, painter = {}, {}
    pat = re.compile(
        r'cache/ablation/hardB3[^/]*/([^/]+)/([^/]+)/N(\d+)/'
        r'seed_(\d+)_(.+)\.json')
    for rel, pdir_override in ((store_rel, None), (extra_rel, 'smartL7')):
        if not rel:
            continue
        store_path = os.path.join(REPO, rel)
        if not os.path.exists(store_path):
            continue
        for key, rec in json.load(open(store_path)).items():
            if rec.get('obj_val') is None:
                continue
            if key.startswith('painter:'):
                if pdir_override:
                    continue
                o = key.split(':')[1]
                s = int(re.search(r'seed_(\d+)_', key).group(1))
                painter.setdefault(o, {})[s] = rec['obj_val']
                continue
            m = pat.search(key)
            if m:
                grid[(m.group(1), pdir_override or m.group(2), m.group(5),
                      int(m.group(4)), int(m.group(3)))] = rec['obj_val']
    return grid, painter


def render_objective_ladder(exp):
    grid, painter = load_scores(exp['store'], exp.get('extra_store'))
    o, fmt = exp['id'], exp['fmt']
    arms = exp.get('arms', ARMS)
    figs_url = exp.get('figs_url', 'figs')
    fixed_all = exp.get('fixed_all_n')
    figs_dir = os.path.join(REPO, exp['figs_dir'])
    out = ['<h2>{} <small>trained &amp; scored on its own objective '
           '(higher = better) &middot; {}</small></h2>'.format(
               exp['title'], exp['world'])]
    out.append('<p class="note">{}</p>'.format(exp.get('intro', '')))
    for f in exp.get('figures', []):
        if os.path.exists(os.path.join(REPO, f)):
            out.append('<img src="plots/{}" alt="{}">'.format(
                os.path.basename(f), exp['id']))
    out.append('<div class="wrap"><table><thead><tr><th>arm</th>')
    out += ['<th>N={}</th>'.format(n) for n in NS]
    out.append('</tr></thead><tbody>')
    for pdir, rung, pname, alabel in arms:
        out.append('<tr><th>{}</th>'.format(alabel))
        ns = (NS if fixed_all else [1]) if pdir == 'fixed' else NS
        for n in NS:
            if n not in ns:
                out.append('<td class="c mut">&mdash;</td>')
                continue
            vals = []
            for s in SEEDS:
                v = grid.get((o, pdir, rung, s, n))
                if v is None:
                    continue
                fn = ('{}_fixed_{}-dep{}-N{}-fixed.pdf'.format(o, rung, s, n)
                      if (pdir == 'fixed' and fixed_all) else
                      '{}_{}_{}-dep{}-N{}-{}.pdf'.format(
                          o, pdir, rung, s, n, pname))
                txt = fmt.format(v)
                if os.path.exists(os.path.join(figs_dir, fn)):
                    vals.append('<a href="{}/{}" title="seed {} '
                                'convergence">{}</a>'.format(
                                    figs_url, fn, s, txt))
                else:
                    vals.append('<span title="seed {} (conv fig pending)">'
                                '{}</span>'.format(s, txt))
            out.append('<td class="c">{}</td>'.format(
                '<br>'.join(vals) if vals else
                '<span class="mut">&middot;&middot;&middot;</span>'))
        out.append('</tr>')
    pv = painter.get(o, {})
    out.append('<tr><th>painter (ref)</th><td class="c" colspan="6">{}'
               '</td></tr>'.format(
                   ' &nbsp; '.join('s{}: {}'.format(s, fmt.format(v))
                                   for s, v in sorted(pv.items()))
                   or '<span class="mut">pending</span>'))
    out.append('</tbody></table></div>')
    out.append('<p class="note">One value per deployment (seed); click a '
               'value for that run\'s convergence-over-iterations PDF '
               '(unlinked = not yet harvested from the VM).</p>')
    out.append(conv_grid(
        figs_url, figs_dir,
        lambda pdir, rung, s, n, pname:
            ((('{}_fixed_{}-dep{}-N{}-fixed.pdf'.format(o, rung, s, n)
               if fixed_all else
               '{}_{}_{}-dep{}-fixed.pdf'.format(o, pdir, rung, s)))
             if pdir == 'fixed' else
             '{}_{}_{}-dep{}-N{}-{}.pdf'.format(o, pdir, rung, s, n,
                                                pname)),
        arms=arms,
        fixed_ns=NS if fixed_all else (1,),
        painter_fn=lambda s:
            'painter_{}_painter-dep{}-N1-smart.pdf'.format(
                'geo' if exp['world'] == 'georand' else 'stock', s)))
    return '\n'.join(out)


def render_static(exp):
    out = ['<h2>{}</h2>'.format(exp.get('heading', exp['title']))]
    for f in exp.get('figures', []):
        if os.path.exists(os.path.join(REPO, f)):
            out.append('<img src="plots/{}" alt="">'.format(
                os.path.basename(f)))
    out.append('<p class="note">{}</p>'.format(exp.get('intro', '')))
    return '\n'.join(out)




def conv_grid(url_prefix, figs_dir_abs, fname_fn, arms=ARMS,
              painter_fn=None, fixed_ns=(1,)):
    """Compact arm x N grid of convergence-figure links (s1..s5 per
    cell); linked iff the PDF exists on disk. fname_fn(pdir, rung, s,
    n, pname) -> filename. fixed_ns: which N columns the fixed arm
    occupies ((1,) legacy; NS for budgeted-fixed L1 v2)."""
    out = ['<h3>convergence over iterations <small>every run; links '
           'appear as figures are harvested from the VM</small></h3>']
    out.append('<div class="wrap"><table><thead><tr><th>arm</th>')
    out += ['<th>N={}</th>'.format(n) for n in NS]
    out.append('</tr></thead><tbody>')
    for arm in arms:
        # optional per-arm figs override (Tom 2026-08-16: L7 lives in a
        # different artifacts dir with a filename prefix but belongs on
        # the SAME grid, one row below L6):
        # (pdir, rung, pname, label[, figs_url, figs_dir_abs, fname_prefix])
        pdir, rung, pname, alabel = arm[:4]
        a_url = arm[4] if len(arm) > 4 else url_prefix
        a_dir = os.path.join(REPO, arm[5]) if len(arm) > 5 else figs_dir_abs
        a_pfx = arm[6] if len(arm) > 6 else ''
        out.append('<tr><th>{}</th>'.format(alabel))
        ns = list(fixed_ns) if pdir == 'fixed' else NS
        for n in NS:
            if n not in ns:
                out.append('<td class="c mut">&mdash;</td>')
                continue
            links = []
            for s in SEEDS:
                fn = a_pfx + fname_fn(pdir, rung, s, n, pname)
                if os.path.exists(os.path.join(a_dir, fn)):
                    cell = '<a href="{}/{}">s{}</a>'.format(a_url, fn, s)
                    # model-error companion figure (belief vs GT + probe
                    # stats): superscript link when harvested
                    me = 'ME_' + fn
                    if os.path.exists(os.path.join(a_dir, me)):
                        cell += '<a href="{}/{}" title="model error over '                                'iterations"><sup>m</sup></a>'.format(
                                    a_url, me)
                    links.append(cell)
                else:
                    links.append('<span class="mut">s{}</span>'.format(s))
            out.append('<td class="c">{}</td>'.format(' '.join(links)))
        out.append('</tr>')
    if painter_fn:
        links = []
        for s in SEEDS:
            fn = painter_fn(s)
            if os.path.exists(os.path.join(figs_dir_abs, fn)):
                links.append('<a href="{}/{}">s{}</a>'.format(
                    url_prefix, fn, s))
            else:
                links.append('<span class="mut">s{}</span>'.format(s))
        out.append('<tr><th>painter (ref)</th><td class="c" colspan="6">'
                   '{}</td></tr>'.format(' '.join(links)))
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def _parse_seed_spec(spec):
    if '-' in str(spec):
        a, b = str(spec).split('-')
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in str(spec).split(',')]


def grid_progress_html(manifest_path, live=True):
    """Live iteration-progress bar for a running grid (Tom 2026-08-16):
    completed learning iterations / total queued. Done cells contribute
    their ACTUAL n_iters to both sides; pending cells contribute a
    per-class budget estimate (L1 = its N budget; fixed-100 arms = 100;
    stop-v2 500-cap arms = 150 nominal, marked est). Recomputed on every
    generate pass, so the refresh loop keeps it current."""
    try:
        specs = json.load(open(os.path.join(REPO, manifest_path)))
    except (OSError, ValueError):
        return ''
    done_it = est_total = done_cells = total_cells = 0
    for sp in specs:
        ns = [int(n) for n in str(sp['n_values']).split(',')]
        seeds = _parse_seed_spec(sp.get('seeds', '1-5'))
        mi = int(sp.get('max_iter', 100))
        for n in ns:
            for s in seeds:
                total_cells += 1
                hits = glob.glob(os.path.join(
                    REPO, sp['out_root'], 'N{}'.format(n),
                    'seed_{}_*.json'.format(s)))
                got = None
                if hits:
                    try:
                        got = json.load(open(hits[0])).get('n_iters')
                    except (OSError, ValueError):
                        got = None
                if got:
                    done_it += int(got)
                    est_total += int(got)
                    done_cells += 1
                else:
                    est_total += (min(n, mi) if sp['label'] == 'L1'
                                  else (100 if mi <= 100 else 150))
    if not est_total:
        return ''
    pct = 100.0 * done_it / est_total
    if not live:
        return ('<div style="margin:10px 0 18px 0">'
                '<div style="font-size:13px;margin-bottom:4px">grid '
                'progress: <b>{:,}</b> / ~{:,} learning iterations '
                '({:.1f}%) &mdash; {} / {} cells '
                '<span style="color:#888;font-size:11px">(refresh-'
                'cycle snapshot)</span></div>'
                '<div style="background:#333;border-radius:6px;'
                'height:16px;max-width:640px"><div style="background:'
                '#2a78d6;height:16px;border-radius:6px;width:{:.1f}%">'
                '</div></div></div>').format(
            done_it, est_total, pct, done_cells, total_cells,
            min(pct, 100))
    return (
        '<div style="margin:10px 0 18px 0">'
        '<div style="font-size:13px;margin-bottom:4px" id="gridprog-text">'
        'grid progress: <b>{:,}</b> / ~{:,} learning iterations ({:.1f}%) '
        '&mdash; {} / {} cells</div>'
        '<div style="background:#333;border-radius:6px;height:16px;'
        'max-width:640px"><div id="gridprog-bar" style="background:#2f9e6e;'
        'height:16px;border-radius:6px;width:{:.1f}%;'
        'transition:width 0.5s"></div></div>'
        '<div style="max-width:640px;display:flex;gap:18px;margin-top:8px" '
        'id="gridprog-sys">'
        '<div style="flex:1"><div style="font-size:12px" id="vm-ram-text">'
        'VM RAM: &mdash;</div>'
        '<div style="background:#333;border-radius:5px;height:10px;'
        'position:relative"><div id="vm-ram-bar" style="background:#2f9e6e;'
        'height:10px;border-radius:5px;width:0%;transition:width 0.5s">'
        '</div><div style="position:absolute;left:90%;top:-2px;width:2px;'
        'height:14px;background:#c02f4e"></div></div></div>'
        '<div style="flex:1"><div style="font-size:12px" id="vm-cpu-text">'
        'VM CPU: &mdash;</div>'
        '<div style="background:#333;border-radius:5px;height:10px">'
        '<div id="vm-cpu-bar" style="background:#2a78d6;height:10px;'
        'border-radius:5px;width:0%;transition:width 0.5s"></div></div>'
        '</div></div>'
        '<div style="font-size:11px;color:#888;margin-top:2px">denominator '
        'is an estimate: stop-v2 arms budgeted at 150 iters until they '
        'land; live-updated every 30s by progress_tick; red tick = 90% RAM '
        'target (<span id="gridprog-ts">generate-time snapshot</span>)</div>'
        '<script>(function(){{\n'
        'function upd(){{fetch("progress.json?ts="+Date.now(),'
        '{{cache:"no-store"}}).then(function(r){{return r.json()}})'
        '.then(function(d){{\n'
        'var pct=100.0*d.done_it/d.est_total;\n'
        'document.getElementById("gridprog-text").innerHTML='
        '"grid progress: <b>"+d.done_it.toLocaleString()+"</b> / ~"+'
        'd.est_total.toLocaleString()+" learning iterations ("+'
        'pct.toFixed(1)+"%) &mdash; "+d.done_cells+" / "+d.total_cells+'
        '" cells";\n'
        'document.getElementById("gridprog-bar").style.width='
        'Math.min(pct,100)+"%";\n'
        'if (d.ram_pct !== undefined) {{\n'
        ' document.getElementById("vm-ram-text").innerHTML="VM RAM: "+'
        'd.ram_used_gb+"G / "+d.ram_total_gb+"G ("+d.ram_pct+"%)";\n'
        ' var rb=document.getElementById("vm-ram-bar");\n'
        ' rb.style.width=Math.min(d.ram_pct,100)+"%";\n'
        ' rb.style.background=d.ram_pct>=92?"#c02f4e":'
        '(d.ram_pct>=86?"#eda100":"#2f9e6e");\n'
        ' document.getElementById("vm-cpu-text").innerHTML="VM CPU: "+'
        'd.cpu_pct+"% (load "+d.load1+" / "+d.cores+" cores, "+'
        'd.cells_running+" cells)";\n'
        ' document.getElementById("vm-cpu-bar").style.width='
        'Math.min(d.cpu_pct,100)+"%";\n'
        '}}\n'
        'document.getElementById("gridprog-ts").textContent='
        '"head-live as of "+d.ts;\n'
        '}}).catch(function(){{}})}}\n'
        'upd(); setInterval(upd, 30000);}})();</script>'
        '</div>'.format(
            done_it, est_total, pct, done_cells, total_cells, min(pct, 100)))


def render_ladder_links(exp):
    """Policy-ladder experiment: figures + a convergence-link grid over
    cache/ablation/policy_ladder_fixed_artifacts/figs (pattern
    <rung>-dep<seed>-N<n>-<pmode>.pdf; no arm prefix)."""
    out = ['<h2>{}</h2>'.format(exp.get('heading', exp['title']))]
    if exp.get('progress_manifest'):
        out.append(grid_progress_html(
            exp['progress_manifest'],
            live=exp.get('progress_live', True)))
    out.append('<p class="note">{}</p>'.format(exp.get('intro', '')))
    for f in exp.get('figures', []):
        if os.path.exists(os.path.join(REPO, f)):
            out.append('<img src="plots/{}" alt="">'.format(
                os.path.basename(f)))
    figs_abs = os.path.join(REPO, exp['figs_dir'])
    fixed_all = exp.get('fixed_all_n')
    _pfx = exp.get('fname_prefix', '')
    out.append(conv_grid(
        exp['figs_url'], figs_abs,
        lambda pdir, rung, s, n, pname:
            _pfx + (('{}-dep{}-N{}-fixed.pdf'.format(rung, s, n) if fixed_all
              else '{}-dep{}-fixed.pdf'.format(rung, s))
             if pdir == 'fixed'
             else '{}-dep{}-N{}-{}.pdf'.format(rung, s, n, pname)),
        arms=exp.get('arms', ARMS),
        fixed_ns=NS if fixed_all else (1,)))
    return '\n'.join(out)


RENDERERS = {'objective_ladder': render_objective_ladder,
             'static': render_static,
             'ladder_links': render_ladder_links}


def main():
    os.makedirs(SITE, exist_ok=True)
    for name, target in (('figs', os.path.join(REPO, HARDB3_FIGS)),
                         ('figs_hb3v2', os.path.join(
                             REPO,
                             'cache/ablation/hardB3v2_artifacts/figs')),
                         ('figs_ladder2', os.path.join(
                             REPO,
                             'cache/ablation/policy_ladder_v2_artifacts'
                             '/figs')),
                         ('figs_ladder', os.path.join(
                             REPO,
                             'cache/ablation/policy_ladder_fixed_artifacts'
                             '/figs')),
                         ('figs_l7k3', os.path.join(
                             REPO,
                             'cache/ablation/hardB3v2_L7K3_artifacts'
                             '/figs')),
                         ('figs_ladder3', os.path.join(
                             REPO,
                             'cache/ablation/policy_ladder_v3_artifacts'
                             '/figs')),
                         ('figs_ladder3lbc', os.path.join(
                             REPO,
                             'cache/ablation/policy_ladder_v3_LBCACHE_artifacts'
                             '/figs')),
                         ('figs_ladder_l7k3', os.path.join(
                             REPO,
                             'cache/ablation/policy_ladder_v2_L7K3_artifacts'
                             '/figs')),
                         ('plots', os.path.join(REPO, 'figures'))):
        lnk = os.path.join(SITE, name)
        if not os.path.islink(lnk):
            os.symlink(target, lnk)
    nav, panes = [], []
    for i, exp in enumerate(EXPERIMENTS):
        on = ' on' if i == 0 else ''
        nav.append('<button class="tab{}" data-t="e{}">{}</button>'.format(
            on, i, exp['title']))
        tabs, panels = [], []
        for j, sec in enumerate(exp['sections']):
            son = ' on' if j == 0 else ''
            tabs.append('<button class="stab{}" data-t="e{}s{}">{}'
                        '</button>'.format(son, i, j, sec['title']))
            panels.append('<section id="e{}s{}" class="spanel{}">{}'
                          '</section>'.format(
                              i, j, son, RENDERERS[sec['kind']](sec)))
        panes.append('<section id="e{}" class="panel{}">'
                     '<nav class="stabs">{}</nav>{}</section>'.format(
                         i, on, ''.join(tabs), ''.join(panels)))
    stamp = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
    html = TEMPLATE.replace('@NAV@', '\n'.join(nav)) \
                   .replace('@PANELS@', '\n'.join(panes)) \
                   .replace('@STAMP@', stamp)
    open(os.path.join(SITE, 'index.html'), 'w').write(html)
    print('wrote', os.path.join(SITE, 'index.html'))


TEMPLATE = '''<!doctype html><meta charset="utf-8">
<meta http-equiv="refresh" content="180">
<meta http-equiv="Cache-Control" content="no-store">
<title>SCULPTOR experiments</title>
<style>
:root { --bg:#f6f5f2; --ink:#22282b; --mut:#6d7478; --line:#dcdad4;
  --card:#fefdfb; --acc:#31647f; --go:#2f9e6e; }
@media (prefers-color-scheme: dark) { :root { --bg:#191d1f; --ink:#e6e3dc;
  --mut:#8b9296; --line:#33393c; --card:#202528; --acc:#6da3bf;
  --go:#3fae7e; } }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); margin:0;
  font:14px/1.5 system-ui,-apple-system,sans-serif; }
.shell { display:flex; min-height:100vh; }
aside { width:200px; flex:none; border-right:1px solid var(--line);
  padding:1.4rem .9rem; }
aside h1 { font-size:.95rem; font-weight:650; margin:0 0 .2rem; }
aside .sub { color:var(--mut); font-size:.72rem; margin-bottom:1.2rem; }
.tab { display:block; width:100%; text-align:left; font:inherit;
  font-size:.86rem; padding:.45rem .7rem; margin-bottom:.25rem;
  border:1px solid transparent; background:none; color:var(--ink);
  border-radius:7px; cursor:pointer; }
.tab:hover { border-color:var(--line); background:var(--card); }
.tab.on { background:var(--acc); color:#fff; }
.tab:focus-visible { outline:2px solid var(--go); outline-offset:2px; }
main { flex:1; min-width:0; padding:1.6rem 2rem; }
h2 { font-size:1.05rem; margin:0 0 .6rem; }
h2 small { color:var(--mut); font-weight:400; font-size:.78rem; }
.panel { display:none; } .panel.on { display:block; }
.stabs { display:flex; gap:.4rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.stab { font:inherit; font-size:.82rem; padding:.3rem .85rem;
  border:1px solid var(--line); background:var(--card); color:var(--ink);
  border-radius:999px; cursor:pointer; }
.stab.on { background:var(--go); border-color:var(--go); color:#fff; }
.stab:focus-visible { outline:2px solid var(--acc); outline-offset:2px; }
.spanel { display:none; } .spanel.on { display:block; }
.wrap { overflow-x:auto; margin-top:1rem; }
table { border-collapse:collapse; width:100%; background:var(--card);
  border:1px solid var(--line); }
th,td { border:1px solid var(--line); padding:.45rem .6rem;
  text-align:left; vertical-align:top; font-size:.82rem; }
thead th { font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--mut); }
tbody th { white-space:nowrap; font-weight:600; }
.c { font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums; font-size:.78rem; }
.c a { color:var(--acc); text-decoration:none;
  border-bottom:1px dotted var(--acc); }
.c a:hover { color:var(--go); border-bottom-style:solid; }
.mut { color:var(--mut); }
img { max-width:100%; border:1px solid var(--line); background:#fff;
  margin:.6rem 0; }
.note { color:var(--mut); font-size:.76rem; max-width:72ch; }
</style>
<div class="shell">
<aside>
  <h1>SCULPTOR experiments</h1>
  <div class="sub">updated @STAMP@<br>reload for fresh data</div>
  <nav>@NAV@</nav>
</aside>
<main>@PANELS@</main>
</div>
<script>
// Tab selection persists across the 180s self-reload and manual
// refreshes via the URL hash (#e1s2) + localStorage fallback
// (Tom, 2026-08-15).
function activate(btns, panels, b, silent) {
  btns.forEach(function (x) { x.classList.remove('on'); });
  panels.forEach(function (x) { x.classList.remove('on'); });
  b.classList.add('on');
  document.getElementById(b.dataset.t).classList.add('on');
  if (!silent) { remember(); }
}
function remember() {
  var t = document.querySelector('.tab.on');
  var pane = t && document.getElementById(t.dataset.t);
  var st = pane && pane.querySelector('.stab.on');
  var key = (st ? st.dataset.t : (t ? t.dataset.t : ''));
  if (key) {
    try { history.replaceState(null, '', '#' + key); } catch (e) {}
    try { localStorage.setItem('dashTab', key); } catch (e) {}
  }
}
var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab'));
var panels = Array.prototype.slice.call(document.querySelectorAll('.panel'));
tabs.forEach(function (b) {
  b.addEventListener('click', function () { activate(tabs, panels, b); });
});
document.querySelectorAll('.panel').forEach(function (pane) {
  var st = Array.prototype.slice.call(pane.querySelectorAll('.stab'));
  var sp = Array.prototype.slice.call(pane.querySelectorAll('.spanel'));
  st.forEach(function (b) {
    b.addEventListener('click', function () { activate(st, sp, b); });
  });
});
(function restore() {
  var key = (location.hash || '').replace('#', '');
  if (!key) { try { key = localStorage.getItem('dashTab') || ''; } catch (e) {} }
  if (!key) { return; }
  var m = key.match(/^(e\d+)(s\d+)?$/);
  if (!m) { return; }
  var tb = tabs.filter(function (b) { return b.dataset.t === m[1]; })[0];
  if (tb) { activate(tabs, panels, tb, true); }
  if (m[2]) {
    var pane = document.getElementById(m[1]);
    if (pane) {
      var st = Array.prototype.slice.call(pane.querySelectorAll('.stab'));
      var sp = Array.prototype.slice.call(pane.querySelectorAll('.spanel'));
      var sb = st.filter(function (b) { return b.dataset.t === key; })[0];
      if (sb) { activate(st, sp, sb, true); }
    }
  }
  try { history.replaceState(null, '', '#' + key); } catch (e) {}
})();
</script>
'''


if __name__ == '__main__':
    main()
