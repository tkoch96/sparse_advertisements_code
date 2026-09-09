"""THE ablation evaluation (Tom 2026-09-08): given a finished (or running)
ladder directory of cell JSONs, emit

  1. the ladder table -- per rung, the mean trusted objective over
     deployments and the % of the painter->OPP gap it closes (on the means
     and as the mean of per-deployment %), written to ladder_summary.{json,csv};
  2. the same percentage OVER ITERATIONS, per rung, from each cell's
     per-iteration ground-truth objective (`gt_objective_series`), written
     to pct_gap_closed_over_iterations.{pdf,json}.

Inputs are the cell JSONs the ladder writes (seed_<s>_<rung>.json): each
carries `repo_objective` (the cell's final ground-truth objective),
`opp_objective` (one-per-peering under the same evaluator; identical across
the cells of one deployment now that seeded deployments are deterministic)
and `gt_objective_series` ([[iter, objective], ...]; painter is one-shot and
has none). NO metric logic lives anywhere else: cdf_fork.main calls run()
after its CDF figure, and run_ablation_cdf.py reaches it through cdf_fork.

    python -m evaluations.evaluate_ablation --in-dir cache/ablation/<study>/Nprefixes
    python -m evaluations.evaluate_ablation --in-dir ... --prelim   # running study

  3. VERIFICATION that every rung used exactly the features it claims
     (--ws-root <workspace with the cells' solver logs>): per cell, from the
     solver's own log + result JSON, PASS/FAIL lines and a nonzero exit on
     any failure. run_ablation_cdf.py calls this once at the end of every
     study, so a study that violates its own ladder fails loudly.
  4. --selftest: run the real driver on 'small' in a throwaway workspace on
     this machine, then verify it (the local proof).

    python -m evaluations.evaluate_ablation --in-dir <out_root>/N<n> --ws-root <ws>
    python -m evaluations.evaluate_ablation --selftest

--prelim drops the 'rescored' gate: repo_objective/opp_objective are written
by the cell itself at completion (the rescore only adds latency/failure
columns), so preliminary numbers equal the final ones for finished cells.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# rung order / labels / colors are the ladder's single source of truth
from experiments.ablation.cdf_fork import LADDER  # noqa: E402
from evaluations.run_ablation_cdf import LADDER_PROBE_MODE  # noqa: E402  (the per-rung WHEN policy)


def _load_cells(in_dir, require_rescored=True):
    """{(seed, rung): json} for every scored cell in in_dir."""
    cells = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, 'seed_*_*.json'))):
        with open(fn) as f:
            r = json.load(f)
        if (require_rescored and not r.get('rescored')) \
                or r.get('repo_objective') is None:
            continue
        cells[(int(r['seed']), r['rung'])] = r
    return cells


def _anchors(cells):
    """Per seed: painter objective, OPP anchor (mean of the seed's per-cell
    OPP values -- identical when the world is pinned; the spread is reported
    as the anchor's noise floor), and the gap."""
    painter, opp_cells = {}, {}
    for (s, rung), r in cells.items():
        if rung == 'painter':
            painter[s] = float(r['repo_objective'])
        if r.get('opp_objective') is not None:
            opp_cells.setdefault(s, []).append(float(r['opp_objective']))
    opp = {s: float(np.mean(v)) for s, v in opp_cells.items()}
    spread = {s: float(max(v) - min(v)) for s, v in opp_cells.items()}
    return painter, opp, spread


def ladder_summary(in_dir, require_rescored=True):
    """THE headline metric (Tom 2026-09-08): per rung, the mean trusted
    objective over deployments and the cumulative percentage of the
    painter->OPP gap it closes, computed on the MEANS
    (100 * (mean_painter - mean_rung) / (mean_painter - mean_OPP)), plus
    the increment over the previous rung in capability order, the mean of
    the per-deployment percentages (scale-free companion: raw objectives
    mix per-deployment scales) and the per-deployment percentages."""
    cells = _load_cells(in_dir, require_rescored)
    by_rung = {}
    for (s, rung), r in cells.items():
        by_rung.setdefault(rung, {})[s] = float(r['repo_objective'])
    painter, opp, opp_spread = _anchors(cells)
    if 'painter' not in by_rung or not opp:
        return None, ''
    rungs = [r for r, _, _ in LADDER if r in by_rung]
    seeds = sorted(s for s in set(painter) & set(opp)
                   if all(s in by_rung[r] for r in rungs))
    if not seeds:
        return None, ''
    mean = {r: float(np.mean([by_rung[r][s] for s in seeds])) for r in rungs}
    mean_opp = float(np.mean([opp[s] for s in seeds]))
    gap = mean['painter'] - mean_opp
    rows, prev = [], 0.0
    for r in rungs:
        cum = 100.0 * (mean['painter'] - mean[r]) / gap if gap > 0 else float('nan')
        per_seed = {}
        for s in seeds:
            g = painter[s] - opp[s]
            per_seed[s] = (100.0 * (painter[s] - by_rung[r][s]) / g
                           if g > 0 else float('nan'))
        finite = [v for v in per_seed.values() if np.isfinite(v)]
        rows.append({'rung': r, 'mean_objective': mean[r],
                     'mean_minus_opp': mean[r] - mean_opp,
                     'pct_gap_closed_on_means': cum,
                     'increment_pct': cum - prev,
                     'mean_of_per_seed_pct': (float(np.mean(finite))
                                              if finite else float('nan')),
                     'n_seeds_with_positive_gap': len(finite),
                     'pct_gap_closed_per_seed': per_seed})
        prev = cum
    summary = {'seeds': seeds, 'n_deployments': len(seeds),
               'mean_opp_objective': mean_opp,
               'opp_anchor': "mean of the deployment's per-cell OPP values",
               'opp_cell_spread_per_seed': {s: opp_spread[s] for s in seeds},
               'painter_to_opp_gap_on_means': gap,
               'preliminary': not require_rescored, 'rungs': rows}
    hdr = '{:<14}{:>10}{:>12}{:>12}{:>10}{:>14}'.format(
        'rung', 'mean obj', 'mean-OPP', '% gap (cum)', 'incr', 'mean seed-%')
    lines = ['LADDER SUMMARY{} (means over {} deployments; % of painter->OPP '
             'gap closed on the means; OPP mean {:.3f}):'.format(
                 ' [PRELIMINARY, un-rescored cells]' if not require_rescored else '',
                 len(seeds), mean_opp), hdr, '-' * len(hdr)]
    for row in rows:
        lines.append('{:<14}{:>10.3f}{:>12.3f}{:>11.1f}%{:>+9.1f}{:>13.1f}%'.format(
            row['rung'], row['mean_objective'], row['mean_minus_opp'],
            row['pct_gap_closed_on_means'], row['increment_pct'],
            row['mean_of_per_seed_pct']))
    lines.append('{:<14}{:>10.3f}{:>12.3f}{:>11.1f}%'.format(
        'OPP', mean_opp, 0.0, 100.0))
    lines.append('per-deployment % gap closed: ' + '; '.join(
        '{}: {}'.format(row['rung'], ', '.join(
            '{:.0f}'.format(v) for s, v in sorted(row['pct_gap_closed_per_seed'].items())))
        for row in rows if row['rung'] != 'painter'))
    lines.append("OPP anchor = mean of each deployment's per-cell OPP; "
                 'cell-to-cell OPP spread per deployment: ' + ', '.join(
                     '{}: {:.3f}'.format(s, opp_spread[s]) for s in seeds) +
                 ' (painter->OPP gap on means {:.3f})'.format(gap))
    return summary, '\n'.join(lines)


def write_summary_files(summary, out_dir):
    with open(os.path.join(out_dir, 'ladder_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, 'ladder_summary.csv'), 'w') as f:
        f.write('rung,mean_objective,mean_minus_opp,pct_gap_closed_on_means,'
                'increment_pct,mean_of_per_seed_pct\n')
        for row in summary['rungs']:
            f.write('{},{:.6f},{:.6f},{:.3f},{:.3f},{:.3f}\n'.format(
                row['rung'], row['mean_objective'], row['mean_minus_opp'],
                row['pct_gap_closed_on_means'], row['increment_pct'],
                row['mean_of_per_seed_pct']))
        f.write('OPP,{:.6f},0,100,,100\n'.format(summary['mean_opp_objective']))


def pct_over_iterations(in_dir, require_rescored=True):
    """Per rung, the % of the painter->OPP gap closed at every iteration:
    per deployment 100*(painter_s - gt_s(it))/(painter_s - OPP_s) from the
    cell's gt_objective_series (held at its final value past the cell's
    last iteration, which is what an early-stopped cell delivers), then
    (a) the mean of the per-deployment % and (b) the means-based %
    100*(mean painter - mean gt(it))/(mean painter - mean OPP)."""
    cells = _load_cells(in_dir, require_rescored)
    painter, opp, _ = _anchors(cells)
    series = {}   # rung -> seed -> np.array indexed by iteration
    for (s, rung), r in cells.items():
        if rung == 'painter' or s not in painter or s not in opp:
            continue
        ser = r.get('gt_objective_series') or []
        if not ser:
            continue
        its = [int(p[0]) for p in ser]
        vals = [float(p[1]) for p in ser]
        arr = np.full(max(its) + 1, np.nan)
        arr[its] = vals
        # forward-fill gaps (series are recorded per iteration; be safe)
        last = vals[0]
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]
        series.setdefault(rung, {})[s] = arr
    if not series:
        return None
    rungs = [r for r, _, _ in LADDER if r in series]
    seeds = sorted(s for s in set(painter) & set(opp)
                   if all(s in series[r] for r in rungs) and painter[s] - opp[s] > 0)
    if not seeds:
        return None
    n_it = max(len(series[r][s]) for r in rungs for s in seeds)
    out = {'iterations': list(range(n_it)), 'seeds': seeds, 'rungs': {},
           'preliminary': not require_rescored}
    mean_painter = float(np.mean([painter[s] for s in seeds]))
    mean_opp = float(np.mean([opp[s] for s in seeds]))
    for r in rungs:
        held = np.array([np.concatenate([series[r][s],
                                         np.full(n_it - len(series[r][s]),
                                                 series[r][s][-1])])
                         for s in seeds])                      # seeds x n_it
        per_seed = np.array([100.0 * (painter[s] - held[i]) / (painter[s] - opp[s])
                             for i, s in enumerate(seeds)])
        means_based = 100.0 * (mean_painter - held.mean(axis=0)) / (mean_painter - mean_opp)
        out['rungs'][r] = {
            'mean_of_per_seed_pct': per_seed.mean(axis=0).tolist(),
            'means_based_pct': means_based.tolist(),
            'per_seed_pct': {s: per_seed[i].tolist() for i, s in enumerate(seeds)},
            'n_seeds': len(seeds)}
    return out


def plot_pct_over_iterations(data, out_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    color = {r: c for r, _, c in LADDER}
    label = {r: l for r, l, _ in LADDER}
    its = data['iterations']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in ((axes[0], 'mean_of_per_seed_pct',
                            'mean of per-deployment %'),
                           (axes[1], 'means_based_pct',
                            '% on the means of the objectives')):
        for r, d in data['rungs'].items():
            ax.plot(its, d[key], color=color.get(r), label=label.get(r, r), lw=1.6)
        ax.axhline(0, color=color['painter'], ls='--', lw=1, label='painter')
        ax.axhline(100, color='k', ls=':', lw=1, label='one-per-peering (OPP)')
        ax.set_title('{} ({} deployments{})'.format(
            title, len(data['seeds']),
            ', PRELIMINARY' if data.get('preliminary') else ''), fontsize=10)
        ax.set_xlabel('iteration')
        ax.grid(True, alpha=.3)
    axes[0].set_ylabel('% of painter -> OPP gap closed')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)



# ============================ verification ==================================
# Rung -> (memory, direction, explore, mc) the fork's banner must show.
# The rungs themselves are DEFINED in experiments/ablation/sculptor_fork.RUNGS
# (never edited here); this table is what the evaluation checks them against.
EXPECT_FLAGS = {'no_mc':         ('False', 'False', 'none', 'False'),
                'no_memory':     ('False', 'False', 'none', 'True'),
                'no_memory_dir': ('False', 'True',  'none', 'True'),
                'expl_none':     ('True',  'True',  'none', 'True')}
_GATE = re.compile(r'\[probe-gate\] iter=(\d+) mode=(\w+)(.*?)-> (PROBE|step)')
_BANNER = re.compile(r'\[ablation-fork\] memory=(\w+) direction=(\w+) explore=(\w+) mc=(\w+) probe_mode=(\w+)')
_SUMMARY = re.compile(r'\[ablation-assert\] SUMMARY .*?: (\{.*\})')
_NBUDGET = re.compile(r'-N(\d+)-(\w+)$')


class _Checks:
    def __init__(self):
        self.rows = []

    def check(self, ok, name, detail=''):
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    @property
    def ok(self):
        return all(r[0] for r in self.rows)

    def report(self):
        n_fail = sum(1 for r in self.rows if not r[0])
        lines = ['VERIFICATION: {} checks, {} failed -> {}'.format(
            len(self.rows), n_fail, 'PASS' if n_fail == 0 else 'FAIL')]
        for ok, name, detail in self.rows:
            if ok:
                lines.append('   PASS {}'.format(name))
            else:
                lines.append('   FAIL {}  -- {}'.format(name, detail))
        return '\n'.join(lines)


def _cell_log(ws_root, seed, rung):
    hits = [f for f in glob.glob(os.path.join(ws_root, 'S*', 'logs', '*.log'))
            if f.endswith('_s{}_{}.log'.format(seed, rung))]
    return hits[0] if len(hits) == 1 else None


def verify(in_dir, ws_root, dpsize=None, deployments=None, max_iter=None,
           probe_n=None, rungs=None):
    """Prove, per cell, that each rung used exactly the features it claims.

    Evidence is the cell's OWN solver log (ws_root/S*/logs/<label>_N<n>_s<seed>_<rung>.log)
    and its result JSON; nothing is recomputed here.

    Every fork rung (L2-L5), with (memory, direction, explore, mc) per EXPECT_FLAGS:
      banner    '[ablation-fork] memory= direction= explore= mc= probe_mode=' shows
                exactly those flags and probe_mode=scheduled;
      WHEN      every '[probe-gate]' decision is mode=scheduled (zero smart
                decisions), probes sit on the fixed grid (first >= period, every
                gap >= period, count == min(N, n_iters//period) within the retry
                tolerance), and JSON probes_spent equals the PROBE decisions;
      WHAT      explore=none: measured advertisements <= 1 initial grounding +
                the scheduled probes -> the max-information phase issued nothing;
      binding   '[ablation-assert] SUMMARY {...}' printed (each check raises at
                the offending iteration, so a printed count proves the flag held
                every iteration): memory OFF <=> iter_start_binary & step_binary > 0
                (advertisement binary at every iteration start and after every
                step), direction OFF <=> grad_single & step_single > 0 (<=1
                nonzero gradient component, <=1 changed coordinate per step),
                MC OFF <=> mc_off_workers > 0 (every worker answered the MC-off
                RPC with point-mass pdfs every iteration).
    L6 'full':  the MAINLINE solver (fork env scrubbed, no fork banner), every
                gate decision mode=smart, budget honored.
    painter:    no probe-gate decisions (one-shot baseline).

    CONTRACT (Tom 2026-09-08): when dpsize / deployments / max_iter / probe_n /
    rungs are given, the study must match them EXACTLY or the verification
    fails: the solver logs are named for the dpsize; the seeds present are
    exactly 1..deployments; the rungs present are exactly the requested set;
    every trained cell ran max_iter iterations; every cell's budget N is the
    requested number (or, for 'prefixes', the deployment's prefix count, the
    same for every rung of that deployment)."""
    C = _Checks()
    cells = _load_cells(in_dir, require_rescored=False)
    seeds = sorted({s for s, _ in cells})
    rungs_present = {r for _, r in cells}
    C.check(cells, 'cells found in {}'.format(in_dir), 'none')
    # ---- contract
    if deployments is not None:
        C.check(seeds == list(range(1, int(deployments) + 1)),
                'contract: deployments present == 1..{}'.format(deployments), 'seeds={}'.format(seeds))
    if rungs is not None:
        want = set(r for r in str(rungs).split(',') if r)
        C.check(rungs_present == want,
                'contract: rungs present == {}'.format(sorted(want)),
                'present={} missing={} extra={}'.format(sorted(rungs_present), sorted(want - rungs_present), sorted(rungs_present - want)))
    if dpsize is not None:
        logs = glob.glob(os.path.join(ws_root, 'S*', 'logs', '*.log'))
        tag = 'cdf_{}_N'.format(str(dpsize).replace('/', '_'))
        bad = [os.path.basename(f) for f in logs if not os.path.basename(f).startswith(tag)]
        C.check(logs and not bad, 'contract: every solver log is a {} cell'.format(dpsize),
                'n_logs={} not matching {!r}: {}'.format(len(logs), tag, bad[:5]))
    budget_by_seed = {}
    for s in seeds:
        for rung in [r for r, _, _ in LADDER if r in rungs_present]:
            cell = 'seed {} {}'.format(s, rung)
            r = cells.get((s, rung))
            if r is None:
                C.check(False, cell + ': result JSON present', 'missing')
                continue
            lf = _cell_log(ws_root, s, rung)
            if not C.check(lf is not None, cell + ': solver log present in {}'.format(ws_root), 'not found'):
                continue
            with open(lf, errors='replace') as f:
                text = f.read()
            gates = [m for m in (_GATE.search(l) for l in text.splitlines()) if m]
            modes = sorted(set(m.group(2) for m in gates))
            probes = [int(m.group(1)) for m in gates if m.group(4) == 'PROBE']
            if rung == 'painter':
                C.check(not gates, cell + ': painter has no probe-gate decisions', str(modes))
                continue
            nb = _NBUDGET.search(os.path.basename(str(r.get('save_run_dir', ''))))
            N = int(nb.group(1)) if nb else None
            n_iters = int(r.get('n_iters') or r.get('max_iter') or 0)
            if max_iter is not None:
                C.check(int(r.get('max_iter') or -1) == int(max_iter),
                        cell + ': contract: max_iter == {}'.format(max_iter), 'json max_iter={}'.format(r.get('max_iter')))
            if probe_n is not None:
                if str(probe_n).isdigit():
                    C.check(N == int(probe_n), cell + ': contract: budget N == {}'.format(probe_n), 'N={}'.format(N))
                else:   # 'prefixes': one per prefix of THIS deployment, same for every rung
                    npf = re.search(r'n_prefixes=(\d+)', text)
                    C.check(N is not None and (npf is None or int(npf.group(1)) == N),
                            cell + ': contract: budget N == prefix count of deployment {}'.format(s),
                            'N={} n_prefixes={}'.format(N, npf.group(1) if npf else '?'))
                    budget_by_seed.setdefault(s, set()).add(N)
            if rung == 'full':
                C.check(gates and modes == [LADDER_PROBE_MODE['full']], cell + ': every gate decision mode={}'.format(LADDER_PROBE_MODE['full']), 'modes={}'.format(modes))
                C.check(_BANNER.search(text) is None and 'rung=full: scrubbed fork env' in text,
                        cell + ': full is the MAINLINE solver (no fork banner; fork env scrubbed)', '')
                C.check(N is not None and int(r.get('probes_spent', -1)) <= N,
                        cell + ': budget honored (probes_spent {} <= N {})'.format(r.get('probes_spent'), N), '')
                continue
            if rung not in EXPECT_FLAGS:
                C.check(False, cell + ': unknown fork rung', rung)
                continue
            exp = EXPECT_FLAGS[rung]
            bm = _BANNER.search(text)
            want_mode = LADDER_PROBE_MODE.get(rung)
            C.check(bm is not None and bm.groups()[:4] == exp and bm.group(5) == want_mode,
                    cell + ': banner memory/direction/explore/mc == {} and probe_mode={}'.format(exp, want_mode),
                    'banner={}'.format(bm.groups() if bm else None))
            C.check(gates and modes == [want_mode],
                    cell + ': every gate decision mode={} (zero other-mode decisions)'.format(want_mode),
                    'modes={} n={}'.format(modes, len(gates)))
            if N:
                tconv = int(r.get('max_iter') or n_iters)   # SCULPTOR_ABLATION_PROBE_TCONV = max_iter (queue)
                period = max(1, int(round(tconv / N)))
                exp_cnt = min(N, n_iters // period)
                gaps = [b - a for a, b in zip(probes, probes[1:])]
                C.check(exp_cnt - 1 <= len(probes) <= exp_cnt,
                        cell + ': probe count {} == min(N={}, n_iters//period={}) (retry tolerance 1)'.format(len(probes), N, exp_cnt),
                        'probes at {}'.format(probes))
                C.check(not probes or probes[0] >= period, cell + ': first probe at iteration >= period {}'.format(period), 'first={}'.format(probes[:1]))
                C.check(all(g >= period for g in gaps), cell + ': every gap between probes >= period {}'.format(period), 'gaps={}'.format(gaps))
            else:
                C.check(False, cell + ': budget N parsed from save_run_dir', str(r.get('save_run_dir')))
            C.check(int(r.get('probes_spent', -1)) == len(probes),
                    cell + ': JSON probes_spent == PROBE decisions in the log ({})'.format(len(probes)),
                    'json={}'.format(r.get('probes_spent')))
            # A probe at an advertisement measured before (memory ON can
            # revisit the initial adv) adds no new entry, so the bound is <=:
            # any measured adv beyond initial + probes would be an
            # exploration measurement.
            C.check(0 < int(r.get('n_advs_measured', -1)) <= len(probes) + 1,
                    cell + ': explore=none: measured advs ({}) <= 1 initial + {} scheduled probes (max-info phase issued nothing)'.format(r.get('n_advs_measured'), len(probes)),
                    'n_advs_measured={}'.format(r.get('n_advs_measured')))
            sm = _SUMMARY.search(text)
            if not C.check(sm is not None, cell + ': fork binding checks held to the end (SUMMARY printed)', ''):
                continue
            import ast as _ast
            chk = _ast.literal_eval(sm.group(1))
            mem_off, dir_off, mc_off = exp[0] == 'False', exp[1] == 'False', exp[3] == 'False'
            C.check((chk.get('mc_off_workers', 0) > 0) == mc_off,
                    cell + ': MC {} (mc_off_workers={})'.format('OFF: workers answered MC-off RPC every iteration' if mc_off else 'ON: no MC-off checks', chk.get('mc_off_workers')), str(chk))
            C.check((chk.get('iter_start_binary', 0) > 0) == mem_off and (chk.get('step_binary', 0) > 0) == mem_off,
                    cell + ': memory {} (iter_start_binary={}, step_binary={})'.format('OFF: advertisement binary every iteration' if mem_off else 'ON: no binary checks', chk.get('iter_start_binary'), chk.get('step_binary')), str(chk))
            C.check((chk.get('grad_single', 0) > 0) == dir_off and (chk.get('step_single', 0) > 0) == dir_off,
                    cell + ': direction {} (grad_single={}, step_single={})'.format('OFF: <=1 coordinate per gradient/step' if dir_off else 'ON: no single-coordinate checks', chk.get('grad_single'), chk.get('step_single')), str(chk))
    for s_, ns in budget_by_seed.items():
        C.check(len(ns) == 1, 'contract: every rung of deployment {} ran the same budget'.format(s_), 'N values={}'.format(sorted(ns)))
    return C


def selftest(iters=16, probe_n=4, deployments=2, keep=False):
    """Local proof: run the REAL driver on 'small' in a throwaway workspace
    (the driver ends by calling this module's table + curves + verify), then
    verify again here and return the verdict."""
    import subprocess
    import tempfile
    root = tempfile.mkdtemp(prefix='evaluate_ablation_selftest_')
    out_root, ws_root = os.path.join(root, 'out'), os.path.join(root, 'ws')
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT, MPLBACKEND='Agg', RAY_ADDRESS='local',
               RAY_TMPDIR=tempfile.mkdtemp(prefix='/tmp/rt_'), SCULPTOR_DEPSTORE='0',
               SCULPTOR_ABLATION_ASSERTS='1')
    rungs = 'painter,no_mc,no_memory,no_memory_dir,expl_none,full'
    print('[selftest] small x {} deployments x {} iters, N={}, rungs {}\n[selftest] scratch {}'.format(
        deployments, iters, probe_n, rungs, root), flush=True)
    log = os.path.join(root, 'driver.log')
    with open(log, 'w') as lf:
        rc = subprocess.call([sys.executable, '-u', os.path.join(_REPO_ROOT, 'evaluations', 'run_ablation_cdf.py'),
                              '--dpsize', 'small', '--deployments', str(deployments),
                              '--max-iter', str(iters), '--probe-n', str(probe_n), '--rungs', rungs,
                              '--slots', '3', '--workers-per-run', '2',
                              '--out-root', out_root, '--ws-root', ws_root, '--queue-only'],
                             cwd=_REPO_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print('[selftest] driver rc={} (log {})'.format(rc, log), flush=True)
    C = verify(os.path.join(out_root, 'N{}'.format(probe_n)), ws_root, dpsize='small',
               deployments=deployments, max_iter=iters, probe_n=probe_n, rungs=rungs)
    C.check(rc == 0, 'driver exit code 0', 'rc={}'.format(rc))
    print(C.report(), flush=True)
    if C.ok and not keep:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    else:
        print('[selftest] kept scratch: {}'.format(root))
    return C.ok


def run(in_dir, out_dir=None, require_rescored=True, plot=True, ws_root=None,
        contract=None):
    """Table + files + over-iterations figure (+ verification when ws_root is
    given). Returns (summary, curves, verification-or-None)."""
    out_dir = out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)
    summary, table = ladder_summary(in_dir, require_rescored)
    if summary is None:
        print('[evaluate_ablation] no complete painter/OPP deployment in {} '
              '-> no ladder summary'.format(in_dir), flush=True)
        summary = None
    if summary is not None:
        print('\n' + table + '\n', flush=True)
        write_summary_files(summary, out_dir)
        print('[evaluate_ablation] wrote {}/ladder_summary.{{json,csv}}'.format(out_dir),
              flush=True)
    curves = pct_over_iterations(in_dir, require_rescored) if plot else None
    if curves is not None:
        with open(os.path.join(out_dir, 'pct_gap_closed_over_iterations.json'), 'w') as f:
            json.dump(curves, f)
        pdf = os.path.join(out_dir, 'pct_gap_closed_over_iterations.pdf')
        plot_pct_over_iterations(curves, pdf)
        print('[evaluate_ablation] wrote {} (+ .json; {} rungs x {} iterations x '
              '{} deployments)'.format(pdf, len(curves['rungs']),
                                       len(curves['iterations']),
                                       len(curves['seeds'])), flush=True)
    elif plot:
        print('[evaluate_ablation] no gt_objective_series -> no over-iterations '
              'figure', flush=True)
    ver = None
    if ws_root:
        ver = verify(in_dir, ws_root, **(contract or {}))
        print('\n' + ver.report() + '\n', flush=True)
        with open(os.path.join(out_dir, 'verification.txt'), 'w') as f:
            f.write(ver.report() + '\n')
    return summary, curves, ver


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', help='ladder dir of seed_<s>_<rung>.json (= <out_root>/N<n>)')
    ap.add_argument('--out-dir', default=None, help='default: --in-dir')
    ap.add_argument('--ws-root', default=None,
                    help='queue workspace holding the cells\' solver logs (S*/logs/) -> '
                         'runs the per-rung feature/probe-policy verification')
    ap.add_argument('--dpsize', default=None, help='contract: the study must be this size')
    ap.add_argument('--deployments', type=int, default=None, help='contract: exactly deployments 1..D present')
    ap.add_argument('--max-iter', type=int, default=None, help='contract: every trained cell ran this many iterations')
    ap.add_argument('--probe-n', default=None, help="contract: budget per cell (int, or 'prefixes')")
    ap.add_argument('--rungs', default=None, help='contract: exactly this comma list of rungs present')
    ap.add_argument('--prelim', action='store_true',
                    help='running study: drop the rescored gate')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--selftest', action='store_true',
                    help="run the real driver on 'small' here, then verify (local proof)")
    ap.add_argument('--selftest-iters', type=int, default=16)
    ap.add_argument('--selftest-probe-n', type=int, default=4)
    ap.add_argument('--selftest-deployments', type=int, default=2)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.selftest_iters, a.selftest_probe_n,
                             a.selftest_deployments, a.keep) else 1
    if not a.in_dir:
        ap.error('--in-dir is required (or --selftest)')
    contract = {'dpsize': a.dpsize, 'deployments': a.deployments, 'max_iter': a.max_iter,
                'probe_n': a.probe_n, 'rungs': a.rungs}
    if a.ws_root and any(v is None for v in contract.values()):
        ap.error('--ws-root verification requires the full contract: --dpsize --deployments --max-iter --probe-n --rungs')
    summary, _, ver = run(a.in_dir, a.out_dir, require_rescored=not a.prelim,
                          plot=not a.no_plot, ws_root=a.ws_root, contract=contract)
    # exit code: verification is the gate. A missing table (un-rescored
    # study, e.g. --queue-only or --prelim on a run with no complete
    # deployment) is reported, not fatal, unless nothing at all was evaluated.
    if ver is not None:
        return 0 if ver.ok else 1
    return 0 if summary is not None else 1


if __name__ == '__main__':
    sys.exit(main())
