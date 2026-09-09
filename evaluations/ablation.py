"""THE ablation script (Tom 2026-09-08: one file). Subcommands:

  run       the study driver: pinned/seeded deployments + inits -> queue manifest
            (one spec per WHEN policy from LADDER_PROBE_MODE) -> run_n_sweep_queue
            (cells via run_cell below) -> trusted rescore -> CDF figure -> `evaluate`
            with this study's own contract.
  evaluate  THE evaluation of a ladder directory: % of painter->OPP gap closed per
            rung (table + files), the same % over iterations (pdf + json), and --
            given --ws-root -- VERIFICATION that every rung used exactly the
            features it claims, under an explicit contract (--dpsize --deployments
            --max-iter --probe-n --rungs [--full-probe-mode]); mismatch fails.
  selftest  run `run` on 'small' here, then verify (the local proof).
  tree      file the harvested per-cell figures/state pickles as
            deployment_NN/L#_<rung>/... (--from-harvest).

    python -m evaluations.ablation run --dpsize testing_feature-actual-10 --deployments 10 \
        --max-iter 150 --probe-n prefixes --dep-dir cache/ablation/cdf_a10_deps \
        --out-root cache/ablation/<study> --ws-root /home/ubuntu/<study>_ws
    python -m evaluations.ablation evaluate --in-dir cache/ablation/<study>/Nprefixes \
        --ws-root <ws> --dpsize ... --deployments ... --max-iter ... --probe-n ... --rungs ...
    python -m evaluations.ablation selftest
    python -m evaluations.ablation tree --from-harvest <figs dir> --out-dir <tree>

The per-cell wrapper (harvest_figs / clean_cell / run_cell) lives here too; the
queue (experiments/ablation/run_n_sweep_queue.py) imports it from this module.
WHAT each rung optimizes with is defined ONCE in experiments/ablation/sculptor_fork.RUNGS.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = _REPO_ROOT
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.ablation.cdf_fork import LADDER  # noqa: E402  (rung order/labels/colors)


# ============================ per-cell wrapper ==============================
def harvest_figs(ws, figs_dir, label, dpsize):
    """Copy this cell's convergence/model-error PDFs from ws/runs/* into
    figs_dir under '<label>_<rundir-suffix>.pdf'. Each slot runs ONE cell
    at a time, so any dir in this slot's runs/ belongs to the cell that
    just finished (timestamped actual-N run dirs escape labeled globs --
    found 2026-08-18)."""
    if not figs_dir:
        return
    os.makedirs(figs_dir, exist_ok=True)
    for d in glob.glob(os.path.join(ws, 'runs', '*')):
        suffix = os.path.basename(d).replace(
            'ablation-{}-'.format(dpsize), '', 1)
        src = os.path.join(d, 'convergence_over_iterations.pdf')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(
                figs_dir, '{}_{}.pdf'.format(label, suffix)))
        me = os.path.join(d, 'model_error_over_iterations.pdf')
        if os.path.exists(me):
            shutil.copy(me, os.path.join(
                figs_dir, 'ME_{}_{}.pdf'.format(label, suffix)))
        # final state pickle rides along (Tom 2026-08-27): highest-N
        # state-*.pkl is the hot-start / post-mortem artifact -- ~1MB,
        # part of the keep set, everything else in the run dir is fluff.
        states = []
        for f in glob.glob(os.path.join(d, 'state-*.pkl')):
            m = re.search(r'state-(\d+)\.pkl$', f)
            if m:
                states.append((int(m.group(1)), f))
        if states:
            n, f = max(states)
            shutil.copy(f, os.path.join(
                figs_dir, '{}_{}_state-{}.pkl'.format(label, suffix, n)))


def clean_cell(ws, ray_tmp=None):
    """Delete the cell's disk fluff: every dir under ws/runs (run state
    pickles ride inside) and the slot's ray tmp sessions. Runs on EVERY
    exit path -- the old inline harvest only fired on rc==0, so failed
    cells leaked their run dirs, and ray_q_S* grew one dead session per
    cell until the box hit ENOSPC (2026-08-27)."""
    for d in glob.glob(os.path.join(ws, 'runs', '*')):
        shutil.rmtree(d, ignore_errors=True)
    if ray_tmp and os.path.isdir(ray_tmp):
        # the cell's driver has exited, so its raylet tree is dead; only
        # session dirs live here and the next cell creates a fresh one.
        for d in glob.glob(os.path.join(ray_tmp, 'session_*')):
            if os.path.islink(d):
                try:
                    os.remove(d)
                except OSError:
                    pass
            else:
                shutil.rmtree(d, ignore_errors=True)


def run_cell(cmd, ws, env, log_path, timeout_s,
             figs_dir=None, label=None, dpsize=None):
    """Run one ablation cell to completion and ALWAYS leave the slot
    clean. Returns the subprocess rc (-99 on timeout, matching the queue
    convention). Figures are harvested before cleanup regardless of rc so
    a failed cell still leaves its convergence evidence."""
    try:
        with open(log_path, 'w') as lf:
            try:
                rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT,
                                     timeout=timeout_s)
            except subprocess.TimeoutExpired:
                rc = -99
    finally:
        try:
            harvest_figs(ws, figs_dir, label, dpsize)
        finally:
            clean_cell(ws, env.get('RAY_TMPDIR'))
    return rc


# ================================ study driver ==============================
sys.path.insert(0, _REPO)

# active ladder for the paper CDF: no_direction retired 2026-08-18,
# expl_random retired 2026-08-12
# WHEN-to-measure policy PER RUNG (Tom 2026-09-08: the ablation overrides the
# solver defaults for every rung explicitly; nothing rides on a default).
# WHAT each rung optimizes with (memory / direction / explore / MC) is defined
# once in experiments/ablation/sculptor_fork.RUNGS; this table is the probing
# policy of the same ladder, and `ablation evaluate` verifies
# both from every cell's own log.
LADDER_PROBE_MODE = {
    'painter':       None,          # one-shot baseline, never probes
    'no_mc':         'scheduled',   # L2: fixed schedule, every ~TCONV/N iterations
    'no_memory':     'scheduled',   # L3
    'no_memory_dir': 'scheduled',   # L4
    'expl_none':     'scheduled',   # L5: no exploration of any kind
    'full':          'smart',       # L6 = SCULPTOR: uncertainty-gated probing
}
PAPER_RUNGS = 'full,expl_none,no_memory_dir,no_memory,no_mc,painter'


def _export_deps(dep_dir, dpsize, k):
    """dep_seed<1..k>.pkl from depstore trainings whose key dpsize
    matches. Ordered by ingest provenance sim (so seed i is always the
    same deployment across runs); refuses if fewer than k exist."""
    import json as _json
    import pickle
    import shutil
    from core import depstore
    st = depstore.Depstore()
    cands = []
    for e in st.index():
        if e.get('kind') != 'training':
            continue
        d = os.path.join(st.root, e['path'])
        mfn = os.path.join(d, 'manifest.json')
        dpp = os.path.join(d, 'deployment.pkl')
        if not (os.path.exists(mfn) and os.path.exists(dpp)):
            continue
        mf = _json.load(open(mfn))
        if mf.get('key', {}).get('dpsize') != str(dpsize):
            continue
        sim = mf.get('provenance', {}).get('sim')
        cands.append((int(sim) if sim is not None else 10**6, dpp,
                      mf.get('fp')))
    # index can list a fp twice (touches); dedupe by path
    cands = sorted({p: (s, p, fp) for s, p, fp in cands}.values())
    assert len(cands) >= k, \
        'only {} depstore trainings for dpsize {} (need {})'.format(
            len(cands), dpsize, k)
    os.makedirs(dep_dir, exist_ok=True)
    for i, (sim, dpp, fp) in enumerate(cands[:k]):
        dst = os.path.join(dep_dir, 'dep_seed{}.pkl'.format(i + 1))
        if not os.path.exists(dst):
            shutil.copy(dpp, dst)
        print('[cdf] seed {} <- training fp {} (ingest sim {})'.format(
            i + 1, fp, sim), flush=True)


def _ensure_inits_from_deps(seeds, dep_tpl, init_dir):
    """Canonical per-seed inits built from the PINNED deployments
    (mirrors _ensure_inits, which draws by seed)."""
    import pickle
    import numpy as np
    os.makedirs(init_dir, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(
        os.path.join(init_dir, 'init_dep{}.npy'.format(s)))]
    if not todo:
        return
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers.helpers import deployment_to_prefixes
    for s in todo:
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(s)
        dep = pickle.load(open(dep_tpl.format(seed=s), 'rb'))
        sas = Sparse_Advertisement_Eval(
            dep, verbose=False, lambduh=0,
            using_resilience_benefit=False, gamma=0,
            n_prefixes=deployment_to_prefixes(dep))
        a0 = sas.init_advertisement()
        np.save(os.path.join(init_dir, 'init_dep{}.npy'.format(s)), a0)
        print('[cdf] canonical init seed {} (from dep-file, shape {})'
              .format(s, a0.shape), flush=True)


def _ensure_inits(seeds, dpsize, init_dir):
    """Canonical per-seed inits for SEEDED deployments (no dep-dir): draw the
    deployment by SCULPTOR_DEPLOYMENT_SEED (the queue's warm-up has cached it)
    and save the solver's canonical initial advertisement."""
    os.makedirs(init_dir, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(
        os.path.join(init_dir, 'init_dep{}.npy'.format(s)))]
    if not todo:
        return
    from core.deployment_setup import get_random_deployment
    from core.sparse_advertisements_v3 import Sparse_Advertisement_Eval
    from helpers.helpers import deployment_to_prefixes
    for s in todo:
        os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(s)
        dep = get_random_deployment(dpsize)
        sas = Sparse_Advertisement_Eval(
            dep, verbose=False, lambduh=0,
            using_resilience_benefit=False, gamma=0,
            n_prefixes=deployment_to_prefixes(dep))
        a0 = sas.init_advertisement()
        np.save(os.path.join(init_dir, 'init_dep{}.npy'.format(s)), a0)
        print('[ablation] canonical init seed {} -> init_dep{}.npy (shape {})'
              .format(s, s, a0.shape), flush=True)


def _parse_budget(token):
    """--probe-n: int | 'prefixes' (one per prefix) | '<float>x' (measurement
    multiplier: round(f * n_prefixes) per deployment, e.g. 0.5x = half)."""
    t = str(token).strip()
    if t.isdigit():
        return 'int', int(t)
    if t == 'prefixes':
        return 'prefixes', None
    if t.endswith('x'):
        return 'mult', float(t[:-1])
    raise ValueError("--probe-n must be an int, 'prefixes' or '<float>x' (got {!r})".format(token))


def _n_prefixes_by_seed(seeds, dpsize, dep_tpl):
    """Prefix count of each deployment (pinned pickle or seeded draw)."""
    from helpers.helpers import deployment_to_prefixes
    out = {}
    for s in seeds:
        if dep_tpl:
            import pickle
            dep = pickle.load(open(dep_tpl.format(seed=s), 'rb'))
        else:
            from core.deployment_setup import get_random_deployment
            os.environ['SCULPTOR_DEPLOYMENT_SEED'] = str(s)
            dep = get_random_deployment(dpsize)
        out[int(s)] = int(deployment_to_prefixes(dep))
    return out


def run_main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dpsize', required=True)
    ap.add_argument('--deployments', type=int, default=10,
                    help='seeds 1..K')
    ap.add_argument('--max-iter', type=int, default=100)
    ap.add_argument('--gamma', default='4',
                    help="training gamma; '4' matches wrapper_eval / the "
                         'paper evals (choke_config parity)')
    ap.add_argument('--probe-n', default='prefixes',
                    help="measurement budget per cell: int | 'prefixes' (default: one "
                         "per prefix of each deployment) | '<float>x' (measurement "
                         "multiplier, e.g. 0.5x = half the prefix count, per deployment)")
    ap.add_argument('--rungs', default=PAPER_RUNGS)
    ap.add_argument('--full-probe-mode', default=None, choices=['smart', 'scheduled'],
                    help="override the 'full' rung's WHEN policy (LADDER_PROBE_MODE) for a "
                         "probing-policy experiment; the evaluation verifies against it")
    ap.add_argument('--cell-env', action='append', default=[], metavar='K=V',
                    help='extra env for every cell (repeatable), e.g. gate knobs '
                         'SCULPTOR_SMART_STALE_FRAC=2.0 for a probing-policy arm')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--ws-root', required=True)
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--port0', type=int, default=57000)
    ap.add_argument('--paper-fig', default=None,
                    help='paper CDF pdf (default figures/paper/'
                         'ablation_ladder_cdf_<dpsize>.pdf)')
    ap.add_argument('--queue-only', action='store_true',
                    help='stop after the queue+rescore (no figures)')
    ap.add_argument('--figs-only', action='store_true',
                    help='skip the queue; just (re)draw from rescores')
    ap.add_argument('--dep-dir', default=None,
                    help='dep-file mode: run cells + rescore on pinned '
                         'deployments dep_seed<K>.pkl in this dir (e.g. '
                         'exported from depstore trainings) instead of '
                         'seeded draws')
    ap.add_argument('--export-deps-from-depstore', action='store_true',
                    help='populate --dep-dir from depstore trainings '
                         'whose key dpsize matches --dpsize (ordered by '
                         'ingest sim), then proceed')
    ap.add_argument('--cell-timeout', type=int, default=None,
                    help='per-cell timeout seconds (SCULPTOR_CELL_TIMEOUT)')
    ap.add_argument('--continue-from', default=None,
                    help='CONTINUATION (Tom 2026-09-02): prior N-dir (e.g. '
                         'cache/ablation/cdf_a10/N10). Non-painter arms '
                         'hot-start from their own final advs there and run '
                         '--max-iter MORE iterations at --probe-n budget '
                         '(the remaining 0.5x); painter re-runs fresh at '
                         '--painter-probe-n (the full 1.5x).')
    ap.add_argument('--painter-probe-n', default='15',
                    help='painter measurement cap in continuation mode')
    ap.add_argument('--init-src', default=None,
                    help='canonical inits dir (default <ws-root>/inits)')
    ap.add_argument('--artifacts-figs', default=None,
                    help='per-cell figure harvest dir (default <out-root>'
                         '_artifacts/figs). Every spec carries it, so '
                         'harvest_figs copies each cell\'s '
                         'convergence/model-error PDFs + final state pickle '
                         'as <label>_seed<seed>_<rung>.* BEFORE clean_cell '
                         'wipes the run dir. Without it (the 2026-09-02 '
                         'campaign) the genuine make_plots PDFs were '
                         'generated and then discarded.')
    ap.add_argument('--workers-per-run', default='auto',
                    help="Ray workers per cell, passed to run_n_sweep_queue: "
                         "an int, or 'auto' (default; Tom 2026-09-08: always "
                         "max out the cores -- ncores / concurrent cells, "
                         "widening in the tail of the study)")
    a = ap.parse_args(argv)

    os.environ.setdefault('MPLBACKEND', 'Agg')
    out_root = os.path.abspath(a.out_root)
    ws_root = os.path.abspath(a.ws_root)
    figs_dir = os.path.abspath(a.artifacts_figs or
                               out_root + '_artifacts/figs')
    os.makedirs(ws_root, exist_ok=True)
    if a.cell_timeout:
        os.environ['SCULPTOR_CELL_TIMEOUT'] = str(a.cell_timeout)

    dep_tpl = None
    if a.dep_dir:
        dep_dir = os.path.abspath(a.dep_dir)
        if a.export_deps_from_depstore:
            _export_deps(dep_dir, a.dpsize, a.deployments)
        dep_tpl = os.path.join(dep_dir, 'dep_seed{seed}.pkl')
        missing = [s for s in range(1, a.deployments + 1)
                   if not os.path.exists(dep_tpl.format(seed=s))]
        assert not missing, 'dep-dir missing seeds {}'.format(missing)
        # cells + inits + rescore must all see the pinned deployments
        os.environ['SCULPTOR_ABLATION_DEP_FILE'] = dep_tpl

    if not a.figs_only:
        init_dir = a.init_src or os.path.join(ws_root, 'inits')
        if dep_tpl:
            _ensure_inits_from_deps(range(1, a.deployments + 1), dep_tpl,
                                    init_dir)
        else:
            _ensure_inits(range(1, a.deployments + 1), a.dpsize, init_dir)
        # drop the Ray connection the init builds opened: a lingering GCS
        # client hard-exits THIS process if any cell cleanup ever touches
        # its Ray session (killed the first actual-10 driver, 2026-08-31)
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

        # per-deployment prefix counts (budget multiplier + evaluation contract)
        npfx = _n_prefixes_by_seed(range(1, a.deployments + 1), a.dpsize, dep_tpl)
        kind, val = _parse_budget(a.probe_n)
        per_seed = {s_: {'n_prefixes': n_,
                         'N': (n_ if kind == 'prefixes' else
                               val if kind == 'int' else max(1, int(round(val * n_))))}
                    for s_, n_ in npfx.items()}
        study = {'dpsize': a.dpsize, 'deployments': a.deployments, 'max_iter': a.max_iter,
                 'probe_n': str(a.probe_n), 'rungs': a.rungs, 'gamma': str(a.gamma),
                 'full_probe_mode': a.full_probe_mode or LADDER_PROBE_MODE['full'],
                 'cell_env': a.cell_env, 'per_seed': per_seed}
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, 'study.json'), 'w') as fh:
            json.dump(study, fh, indent=1)
        print('[ablation] study contract written: {} (budgets {})'.format(
            os.path.join(out_root, 'study.json'),
            {s_: v['N'] for s_, v in per_seed.items()}), flush=True)
        base_env = ({'SCULPTOR_ABLATION_DEP_FILE': dep_tpl}
                    if dep_tpl else {})
        for kv in a.cell_env:
            k, v = kv.split('=', 1)
            base_env[k] = v
        # the per-rung WHEN policy for THIS study (full may be overridden)
        policy = dict(LADDER_PROBE_MODE)
        policy['full'] = a.full_probe_mode or LADDER_PROBE_MODE['full']
        if a.continue_from:
            assert _parse_budget(a.probe_n)[0] != 'mult', '--continue-from does not support a multiplier budget'
            prior = os.path.abspath(a.continue_from)
            # RESUME_FROM: hot-start each arm from its prior final adv. One
            # spec per WHEN policy from LADDER_PROBE_MODE (same table as the
            # fresh branch below). SCULPTOR_PROBE_N/TCONV are the MAINLINE
            # names so the scrubbed 'full' rung gets the same remaining budget.
            _rungs = [r for r in a.rungs.split(',') if r]
            specs = []
            for mode in ('smart', 'scheduled'):
                group = [r for r in _rungs if policy.get(r) == mode]
                if not group:
                    continue
                specs.append({
                    'label': 'cdfext_main',
                    'out_root': out_root,
                    'init_src': init_dir,
                    'probe_mode': mode,
                    'rungs': ','.join(group),
                    'seeds': '1-{}'.format(a.deployments),
                    'n_values': str(a.probe_n),
                    'gamma': str(a.gamma),
                    'max_iter': a.max_iter,
                    'dpsize': a.dpsize,
                    'env': dict(base_env,
                                SCULPTOR_ABLATION_RESUME_FROM=prior,
                                SCULPTOR_PROBE_MODE=mode,
                                SCULPTOR_PROBE_N=str(a.probe_n),
                                SCULPTOR_PROBE_TCONV=str(a.max_iter)),
                    'artifacts_figs': figs_dir,
                })
            if 'painter' in _rungs:
                specs.append({
                    'label': 'cdfext_painter',
                    'out_root': out_root,
                    'init_src': init_dir,
                    'probe_mode': 'scheduled',
                    'rungs': 'painter',
                    'seeds': '1-{}'.format(a.deployments),
                    'n_values': str(a.painter_probe_n),
                    'gamma': str(a.gamma),
                    'max_iter': a.max_iter,
                    'dpsize': a.dpsize,
                    'env': dict(base_env),
                    'artifacts_figs': figs_dir,
                })
        else:
            # One spec per WHEN-to-measure policy (Tom 2026-09-08) x budget
            # group: L1-L5 (+ painter) on the fixed schedule, 'full' (L6 =
            # SCULPTOR) smart unless overridden. A measurement MULTIPLIER
            # budget is per deployment, so it makes one spec per seed.
            _rungs = [r for r in a.rungs.split(',') if r]
            _unknown = [r for r in _rungs if r not in policy]
            assert not _unknown, 'rungs without a LADDER_PROBE_MODE entry: {}'.format(_unknown)
            kind, val = _parse_budget(a.probe_n)
            if kind == 'mult':
                seed_groups = [(str(s_), str(max(1, int(round(val * npfx[s_])))))
                               for s_ in sorted(npfx)]
            else:
                seed_groups = [('1-{}'.format(a.deployments), str(a.probe_n))]
            specs = []
            for seeds_tok, n_tok in seed_groups:
                _base = {
                    'label': 'cdf_{}'.format(a.dpsize.replace('/', '_')),
                    'out_root': out_root,
                    'init_src': init_dir,
                    'seeds': seeds_tok,
                    'n_values': n_tok,
                    'gamma': str(a.gamma),
                    'max_iter': a.max_iter,
                    'dpsize': a.dpsize,
                    'artifacts_figs': figs_dir,
                }
                for mode in ('smart', 'scheduled', None):
                    group = [r for r in _rungs if policy[r] == mode]
                    if not group:
                        continue
                    if 'full' in group:
                        # SCULPTOR_PROBE_N/MODE are the MAINLINE names (the fork
                        # env is scrubbed for 'full'), so L6 gets the same budget
                        # as L1-L5 ('prefixes' resolves per deployment).
                        env = dict(base_env, SCULPTOR_PROBE_MODE=mode,
                                   SCULPTOR_PROBE_N=n_tok)
                    else:
                        env = dict(base_env)
                    specs.append(dict(_base, rungs=','.join(group),
                                      probe_mode=mode or 'scheduled', env=env))
            for sp in specs:
                print('[cdf] spec rungs={} probe_mode={}'.format(
                    sp['rungs'], sp['probe_mode']), flush=True)
        mf = os.path.join(ws_root, 'cdf_manifest.json')
        with open(mf, 'w') as fh:
            json.dump(specs, fh, indent=1)
        print('[cdf] manifest {} ({} rungs x {} deployments = {} cells)'
              .format(mf, len(a.rungs.split(',')), a.deployments,
                      len(a.rungs.split(',')) * a.deployments), flush=True)
        print('[cdf] per-cell figures harvested to {}'.format(figs_dir),
              flush=True)
        qargs = [sys.executable, '-u', '-m',
                 'experiments.ablation.run_n_sweep_queue',
                 '--manifest', mf, '--ws-root', ws_root,
                 '--slots', str(a.slots), '--port0', str(a.port0)]
        if a.workers_per_run:
            qargs += ['--workers-per-run', str(a.workers_per_run)]
        rc = subprocess.call(qargs, cwd=_REPO)
        if rc != 0:
            print('[cdf] queue rc={} -- not drawing figures from a '
                  'partial sweep'.format(rc), flush=True)
            return rc
        if _parse_budget(a.probe_n)[0] == 'mult':
            # per-seed budgets land in N<N_s>/; unify into N<mult>x/ for the
            # evaluation (copies: the queue's resume-skip keeps seeing its own).
            # Done before the --queue-only return so that mode is evaluable too.
            import glob as _glob
            import shutil as _shutil
            _udir = os.path.join(out_root, 'N{}'.format(a.probe_n))
            os.makedirs(_udir, exist_ok=True)
            for fn in _glob.glob(os.path.join(out_root, 'N[0-9]*', 'seed_*_*.json')):
                _shutil.copy(fn, os.path.join(_udir, os.path.basename(fn)))
            print('[ablation] unified per-deployment budget dirs into {}'.format(_udir), flush=True)
        if a.queue_only:
            return 0

    in_dir = os.path.join(out_root, 'N{}'.format(a.probe_n))
    if a.continue_from:
        # unify: painter cells land in N<painter_probe_n>; copy them (post-
        # rescore, so trusted fields ride along) into the main N-dir so the
        # table/CDF tooling sees one directory
        import glob as _glob
        import shutil as _shutil
        pdir = os.path.join(out_root, 'N{}'.format(a.painter_probe_n))
        for fn in _glob.glob(os.path.join(pdir, 'seed_*_painter.json')):
            _shutil.copy(fn, os.path.join(in_dir, os.path.basename(fn)))
            print('[cdf] merged {}'.format(os.path.basename(fn)))
    paper_fig = a.paper_fig or os.path.join(
        _REPO, 'figures', 'paper',
        'ablation_ladder_cdf_{}.pdf'.format(
            a.dpsize.replace('testing_feature-', '').replace('/', '_')))
    rc_cdf = subprocess.call(
        [sys.executable, '-u', '-m', 'experiments.ablation.cdf_fork',
         '--in-dir', in_dir, '--gamma', str(a.gamma),
         '--paper-out', paper_fig],
        cwd=_REPO)
    # THE ablation evaluation (Tom 2026-09-08, one script): ladder % table +
    # % over iterations + per-rung feature/probe-policy VERIFICATION from the
    # cells' own logs. A study that violates its own ladder fails here.
    rc_eval = subprocess.call(
        [sys.executable, '-u', '-m', 'evaluations.ablation', 'evaluate',
         '--in-dir', in_dir, '--ws-root', ws_root,
         '--dpsize', a.dpsize, '--deployments', str(a.deployments),
         '--max-iter', str(a.max_iter), '--probe-n', str(a.probe_n),
         '--rungs', a.rungs,
         '--full-probe-mode', a.full_probe_mode or LADDER_PROBE_MODE['full'],
         '--gamma', str(a.gamma)],
        cwd=_REPO)
    if rc_eval != 0:
        print('[cdf] EVALUATION/VERIFICATION FAILED (rc={}) -- see verification.txt in {}'.format(rc_eval, in_dir), flush=True)
    # convergence figures over iterations, filed by deployment and rung
    try:
        tree_dir = out_root.rstrip('/') + '_tree'
        organize_harvest(figs_dir, tree_dir)
        print('[ablation] convergence figures filed by deployment: {}'.format(tree_dir), flush=True)
    except Exception as e:  # never lose the study over a filing error
        print('[ablation] figure tree filing failed (non-fatal): {}'.format(e), flush=True)
    return rc_cdf or rc_eval


# ================================ evaluation ================================
# rung order / labels / colors are the ladder's single source of truth


OBJECTIVES = ('full', 'latency')


def _cell_objective(r, objective='full', gamma=None):
    """(cell objective, OPP objective) on the chosen metric.

    full    (Tom 2026-09-09, THE ablation metric): latency + gamma * SUM over
            peering-failure scenarios of the latency under that failure, all
            from the trusted rescore (rescore_fork: avg_lat, fail_popp.
            avg_lat_under_failure_abs = mean over the n_popps single-peering
            failures, times n_popps = the sum; same for OPP). gamma defaults to
            the gamma the cell trained with (recorded in its JSON).
    latency the cell's own measured_objective (repo_objective / opp_objective):
            the LP objective under the training gamma, resilience term 0."""
    if objective == 'latency':
        return (float(r['repo_objective']) if r.get('repo_objective') is not None else None,
                float(r['opp_objective']) if r.get('opp_objective') is not None else None)
    fp = r.get('fail_popp') or {}
    if r.get('avg_lat') is None or fp.get('avg_lat_under_failure_abs') is None:
        return None, None
    g = float(gamma if gamma is not None else r.get('gamma', 0.0))
    n = len(r['adv']) if isinstance(r.get('adv'), list) else 0
    obj = float(r['avg_lat']) + g * n * float(fp['avg_lat_under_failure_abs'])
    opp = (float(r['opp_avg_lat']) + g * n * float(fp['opp_avg_lat_under_failure_abs'])
           if r.get('opp_avg_lat') is not None and fp.get('opp_avg_lat_under_failure_abs') is not None else None)
    return obj, opp


def _load_cells(in_dir, require_rescored=True, objective='full', gamma=None):
    """{(seed, rung): json} for every scored cell in in_dir; each carries
    '_obj' / '_opp' on the chosen objective (None when not computable)."""
    cells = {}
    for fn in sorted(glob.glob(os.path.join(in_dir, 'seed_*_*.json'))):
        with open(fn) as f:
            r = json.load(f)
        if (require_rescored and not r.get('rescored')) \
                or r.get('repo_objective') is None:
            continue
        r['_obj'], r['_opp'] = _cell_objective(r, objective, gamma)
        cells[(int(r['seed']), r['rung'])] = r
    return cells


def _anchors(cells):
    """Per seed: painter objective, OPP anchor (mean of the seed's per-cell
    OPP values -- identical when the world is pinned; the spread is reported
    as the anchor's noise floor), and the gap."""
    painter, opp_cells = {}, {}
    for (s, rung), r in cells.items():
        if rung == 'painter' and r.get('_obj') is not None:
            painter[s] = float(r['_obj'])
        if r.get('_opp') is not None:
            opp_cells.setdefault(s, []).append(float(r['_opp']))
    opp = {s: float(np.mean(v)) for s, v in opp_cells.items()}
    spread = {s: float(max(v) - min(v)) for s, v in opp_cells.items()}
    return painter, opp, spread


def ladder_summary(in_dir, require_rescored=True, objective='full', gamma=None):
    """THE headline metric (Tom 2026-09-08): per rung, the mean trusted
    objective over deployments and the cumulative percentage of the
    painter->OPP gap it closes, computed on the MEANS
    (100 * (mean_painter - mean_rung) / (mean_painter - mean_OPP)), plus
    the increment over the previous rung in capability order, the mean of
    the per-deployment percentages (scale-free companion: raw objectives
    mix per-deployment scales) and the per-deployment percentages."""
    cells = _load_cells(in_dir, require_rescored, objective, gamma)
    by_rung = {}
    for (s, rung), r in cells.items():
        if r.get('_obj') is None:
            continue
        by_rung.setdefault(rung, {})[s] = float(r['_obj'])
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
    g_used = sorted({float(r.get('gamma', 0)) for r in cells.values()}) if gamma is None else [float(gamma)]
    summary = {'seeds': seeds, 'n_deployments': len(seeds),
               'objective': objective, 'gamma': g_used,
               'mean_opp_objective': mean_opp,
               'opp_anchor': "mean of the deployment's per-cell OPP values",
               'opp_cell_spread_per_seed': {s: opp_spread[s] for s in seeds},
               'painter_to_opp_gap_on_means': gap,
               'preliminary': not require_rescored, 'rungs': rows}
    hdr = '{:<14}{:>10}{:>12}{:>12}{:>10}{:>14}'.format(
        'rung', 'mean obj', 'mean-OPP', '% gap (cum)', 'incr', 'mean seed-%')
    _objdesc = ('latency + gamma*SUM(peering-failure latencies), rescored, gamma={}'.format(
                    ','.join('{:g}'.format(g) for g in g_used))
                if objective == 'full' else "cell's own training objective (latency LP)")
    lines = ['LADDER SUMMARY{} (objective = {}; means over {} deployments; % of '
             'painter->OPP gap closed on the means; OPP mean {:.3f}):'.format(
                 ' [PRELIMINARY, un-rescored cells]' if not require_rescored else '',
                 _objdesc, len(seeds), mean_opp), hdr, '-' * len(hdr)]
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


def pct_over_iterations(in_dir, require_rescored=True, objective='full', gamma=None):
    """Per rung, the % of the painter->OPP gap closed at every iteration:
    per deployment 100*(painter_s - gt_s(it))/(painter_s - OPP_s) from the
    cell's gt_objective_series (held at its final value past the cell's
    last iteration, which is what an early-stopped cell delivers), then
    (a) the mean of the per-deployment % and (b) the means-based %
    100*(mean painter - mean gt(it))/(mean painter - mean OPP)."""
    # NOTE: the per-iteration series is the cell's own ground-truth objective
    # (gt_objective_series = the latency LP objective during training; no
    # failure sweep per iteration exists), so the curve is always on the
    # 'latency' metric and is anchored on that metric's painter/OPP. The
    # table above is on the requested objective.
    cells = _load_cells(in_dir, require_rescored, 'latency', gamma)
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
    import matplotlib; matplotlib.use('Agg')
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
           probe_n=None, rungs=None, full_probe_mode=None):
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
                elif str(probe_n).endswith('x'):
                    # multiplier: expected N per deployment from the study's
                    # own contract file (written by `run`)
                    sj = os.path.join(os.path.dirname(os.path.abspath(in_dir)), 'study.json')
                    exp_n = None
                    if os.path.exists(sj):
                        with open(sj) as f:
                            exp_n = json.load(f).get('per_seed', {}).get(str(s), {}).get('N')
                    C.check(exp_n is not None and N == int(exp_n),
                            cell + ': contract: budget N == round({} * n_prefixes) = {}'.format(probe_n, exp_n), 'N={}'.format(N))
                else:   # 'prefixes': one per prefix of THIS deployment, same for every rung
                    npf = re.search(r'n_prefixes=(\d+)', text)
                    C.check(N is not None and (npf is None or int(npf.group(1)) == N),
                            cell + ': contract: budget N == prefix count of deployment {}'.format(s),
                            'N={} n_prefixes={}'.format(N, npf.group(1) if npf else '?'))
                    budget_by_seed.setdefault(s, set()).add(N)
            if rung == 'full':
                _fm = full_probe_mode or LADDER_PROBE_MODE['full']
                C.check(gates and modes == [_fm], cell + ': every gate decision mode={}'.format(_fm), 'modes={}'.format(modes))
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
    root = tempfile.mkdtemp(prefix='ablation_selftest_')
    out_root, ws_root = os.path.join(root, 'out'), os.path.join(root, 'ws')
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT, MPLBACKEND='Agg', RAY_ADDRESS='local',
               RAY_TMPDIR=tempfile.mkdtemp(prefix='/tmp/rt_'), SCULPTOR_DEPSTORE='0',
               SCULPTOR_ABLATION_ASSERTS='1')
    rungs = 'painter,no_mc,no_memory,no_memory_dir,expl_none,full'
    print('[selftest] small x {} deployments x {} iters, N={}, rungs {}\n[selftest] scratch {}'.format(
        deployments, iters, probe_n, rungs, root), flush=True)
    log = os.path.join(root, 'driver.log')
    with open(log, 'w') as lf:
        rc = subprocess.call([sys.executable, '-u', '-m', 'evaluations.ablation', 'run',
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


def evaluate(in_dir, out_dir=None, require_rescored=True, plot=True, ws_root=None,
             contract=None, objective='full', gamma=None):
    """Table + files + over-iterations figure (+ verification when ws_root is
    given). Returns (summary, curves, verification-or-None)."""
    out_dir = out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)
    summary, table = ladder_summary(in_dir, require_rescored, objective, gamma)
    if summary is None:
        print('[ablation evaluate] no complete painter/OPP deployment in {} '
              '-> no ladder summary'.format(in_dir), flush=True)
        summary = None
    if summary is not None:
        print('\n' + table + '\n', flush=True)
        write_summary_files(summary, out_dir)
        print('[ablation evaluate] wrote {}/ladder_summary.{{json,csv}}'.format(out_dir),
              flush=True)
    curves = pct_over_iterations(in_dir, require_rescored, objective, gamma) if plot else None
    if curves is not None:
        with open(os.path.join(out_dir, 'pct_gap_closed_over_iterations.json'), 'w') as f:
            json.dump(curves, f)
        pdf = os.path.join(out_dir, 'pct_gap_closed_over_iterations.pdf')
        plot_pct_over_iterations(curves, pdf)
        print('[ablation evaluate] wrote {} (+ .json; {} rungs x {} iterations x '
              '{} deployments)'.format(pdf, len(curves['rungs']),
                                       len(curves['iterations']),
                                       len(curves['seeds'])), flush=True)
    elif plot:
        print('[ablation evaluate] no gt_objective_series -> no over-iterations '
              'figure', flush=True)
    ver = None
    if ws_root:
        ver = verify(in_dir, ws_root, **(contract or {}))
        print('\n' + ver.report() + '\n', flush=True)
        with open(os.path.join(out_dir, 'verification.txt'), 'w') as f:
            f.write(ver.report() + '\n')
    return summary, curves, ver


def evaluate_main(argv=None):
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
    ap.add_argument('--full-probe-mode', default=None, choices=['smart', 'scheduled'],
                    help="contract: the 'full' rung's WHEN policy (default LADDER_PROBE_MODE)")
    ap.add_argument('--objective', default='full', choices=OBJECTIVES,
                    help="ladder metric: 'full' (default; latency + gamma*SUM of peering-"
                         "failure latencies, from the trusted rescore) or 'latency' (the "
                         "cell's own training objective)")
    ap.add_argument('--gamma', type=float, default=None,
                    help='gamma for the full objective (default: each cell\'s training gamma)')
    ap.add_argument('--prelim', action='store_true',
                    help='running study: drop the rescored gate (table then needs --objective latency)')
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args(argv)
    if not a.in_dir:
        ap.error('--in-dir is required')
    contract = {'dpsize': a.dpsize, 'deployments': a.deployments, 'max_iter': a.max_iter,
                'probe_n': a.probe_n, 'rungs': a.rungs}
    if a.ws_root and any(v is None for v in contract.values()):
        ap.error('--ws-root verification requires the full contract: --dpsize --deployments --max-iter --probe-n --rungs')
    contract['full_probe_mode'] = a.full_probe_mode
    summary, _, ver = evaluate(a.in_dir, a.out_dir, require_rescored=not a.prelim,
                               plot=not a.no_plot, ws_root=a.ws_root, contract=contract,
                               objective=a.objective, gamma=a.gamma)
    # exit code: verification is the gate. A missing table (un-rescored
    # study, e.g. --queue-only or --prelim on a run with no complete
    # deployment) is reported, not fatal, unless nothing at all was evaluated.
    if ver is not None:
        return 0 if ver.ok else 1
    return 0 if summary is not None else 1


# ============================ figure tree filing ============================
import matplotlib; matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# capability-ascending; (rung, L-rank, human label, color)
TREE_LADDER = [
    ('no_mc',         2, 'monte-carlo off (painter + one-flip search)', 'tab:brown'),
    ('no_memory',     3, 'monte-carlo on, no memory',                   'tab:orange'),
    ('no_memory_dir', 4, '+ direction (flip-threshold step)',           'tab:purple'),
    ('expl_none',     5, '+ memory (continuous advertisement)',         'tab:blue'),
    ('full',          6, '+ entropic exploration = SCULPTOR',           'tab:green'),
]


L_OF = {rung: L for rung, L, _lab, _c in TREE_LADDER}
L_OF['painter'] = 1


def armslug(rung):
    return 'L{}_{}'.format(L_OF[rung],
                           'full_SCULPTOR' if rung == 'full' else rung)


# harvested names (evaluations.ablation.harvest_figs: '<label>_<run-dir
# suffix>'; the fork's run dir is '<rung>-dep<seed>-N<n>-<mode>', or
# '<rung>-dep<seed>-fixed' for the scrubbed full rung), e.g.
#   cdf_testing_feature-actual-10_no_mc-dep1-N10-smart.pdf
#   ME_cdf_testing_feature-actual-10_full-dep2-fixed.pdf
#   cdf_testing_feature-actual-10_expl_none-dep1-N10-smart_state-12.pkl
_HARVEST = re.compile(r'^(?P<me>ME_)?(?P<label>.+?)_(?P<rung>[a-z_]+)-dep(?P<seed>\d+)-'
                      r'(?P<rest>[^_]+?)(?:_state-(?P<n>\d+))?\.(?P<ext>pdf|pkl)$')


def organize_harvest(figs_dir, out_dir):
    """The REAL runtime figures (Tom 2026-09-07 rerun): file every
    harvested per-cell PDF / state pickle from ablation run's
    artifacts_figs dir into the same deployment_NN/L#_<rung>/ tree the
    series-rebuilt figures use, so the two are drop-in comparable."""
    n = 0
    skipped = []
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(glob.glob(os.path.join(figs_dir, '*'))):
        m = _HARVEST.match(os.path.basename(fn))
        if not m or m.group('rung') not in L_OF:
            skipped.append(os.path.basename(fn))
            continue
        seed = int(m.group('seed'))
        adir = os.path.join(out_dir, 'deployment_{:02d}'.format(seed),
                            armslug(m.group('rung')))
        os.makedirs(adir, exist_ok=True)
        if m.group('ext') == 'pkl':
            dst = 'final_state-{}.pkl'.format(m.group('n'))
        elif m.group('me'):
            dst = 'model_error_over_iterations.pdf'
        else:
            dst = 'convergence_over_iterations.pdf'
        shutil.copy(fn, os.path.join(adir, dst))
        n += 1
    if skipped:
        print('[tree] skipped {} unrecognized: {}'.format(
            len(skipped), skipped[:8]))
    with open(os.path.join(out_dir, 'README.txt'), 'w') as fh:
        fh.write(
            'Per-deployment ablation convergence figures -- the GENUINE '
            'runtime make_plots output of each cell, harvested by\n'
            'ablation run --artifacts-figs (not rebuilt from series).\n\n'
            'deployment_NN/L<rank>_<rung>/convergence_over_iterations.pdf\n'
            '                            model_error_over_iterations.pdf '
            '(when the cell produced one)\n'
            '                            final_state-<N>.pkl (hot-start / '
            'post-mortem artifact)\n\n'
            'Capability order (ascending; L6 = SCULPTOR):\n'
            '  L1 painter        one-shot baseline\n'
            '  L2 no_mc          monte-carlo off (painter + one-flip search)\n'
            '  L3 no_memory      monte-carlo on, no memory\n'
            '  L4 no_memory_dir  + direction (flip-threshold step)\n'
            '  L5 expl_none      + memory (continuous advertisement)\n'
            '  L6 full           + entropic exploration = SCULPTOR '
            '(depstore cache-hit cells have no runtime figure)\n')
    print('[tree] filed {} artifacts under {}'.format(n, out_dir))
    return n


def tree_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default=None,
                    help='ablation_series_export.json (arms/painter/opp)')
    ap.add_argument('--full-curves', default=None,
                    help='full_curves_extracted_a10.json')
    ap.add_argument('--out-dir', default='figures/actual-10-ablation')
    ap.add_argument('--from-harvest', default=None,
                    help='instead of rebuilding from series: file the '
                         'harvested per-cell PDFs/pickles in this '
                         'artifacts_figs dir into the same tree')
    a = ap.parse_args(argv)

    if a.from_harvest:
        return organize_harvest(a.from_harvest, a.out_dir)
    assert a.series and a.full_curves, '--series and --full-curves required'

    S = json.load(open(a.series))
    full_curves = json.load(open(a.full_curves))
    arms, painter, opp = S['arms'], S['painter'], S['opp']
    seeds = sorted(arms, key=int)
    os.makedirs(a.out_dir, exist_ok=True)

    def series_for(seed, rung):
        if rung == 'full':
            fc = full_curves.get(seed)
            return list(zip(fc['iters'], fc['objs'])) if fc else None
        pts = arms.get(seed, {}).get(rung)
        return [(p[0], p[1]) for p in pts] if pts else None

    n_fig = 0
    for seed in seeds:
        dep_dir = os.path.join(a.out_dir, 'deployment_{:02d}'.format(int(seed)))
        os.makedirs(dep_dir, exist_ok=True)
        pv = painter.get(seed)
        ov = opp.get(seed)
        overlay, oleg = plt.subplots(figsize=(7.2, 4.6))

        for rung, L, label, color in TREE_LADDER:
            pts = series_for(seed, rung)
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            # overlay
            oleg.plot(xs, ys, color=color, lw=1.6,
                      label='L{} {} ({})'.format(L, rung, label))
            # single-arm figure
            armslug = 'L{}_{}'.format(
                L, 'full_SCULPTOR' if rung == 'full' else rung)
            adir = os.path.join(dep_dir, armslug)
            os.makedirs(adir, exist_ok=True)
            f, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(xs, ys, color=color, lw=1.8,
                    label='L{} {}'.format(L, rung))
            if pv is not None:
                ax.axhline(pv, color='tab:red', lw=.9, ls=':',
                           label='painter (L1, one-shot)')
            if ov is not None:
                ax.axhline(ov, color='k', lw=.9, ls=':',
                           label='one-per-peering (optimal)')
            ax.set_xlabel('training iteration')
            ax.set_ylabel('ground-truth objective (lower better)')
            ax.set_title('deployment {} — L{} {}\n{}'.format(
                seed, L, rung, label), fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=.3)
            f.tight_layout()
            f.savefig(os.path.join(adir, 'convergence_over_iterations.pdf'),
                      bbox_inches='tight')
            plt.close(f)
            n_fig += 1

        if pv is not None:
            oleg.axhline(pv, color='tab:red', lw=1.0, ls=':',
                         label='L1 painter (one-shot baseline)')
        if ov is not None:
            oleg.axhline(ov, color='k', lw=1.0, ls=':',
                         label='one-per-peering (optimal)')
        oleg.set_xlabel('training iteration')
        oleg.set_ylabel('ground-truth objective (lower better)')
        oleg.set_title('deployment {} — all ablation arms'.format(seed))
        oleg.legend(fontsize=7); oleg.grid(alpha=.3)
        overlay.tight_layout()
        overlay.savefig(os.path.join(dep_dir, '_overlay_all_arms.pdf'),
                        bbox_inches='tight')
        plt.close(overlay)

    with open(os.path.join(a.out_dir, 'README.txt'), 'w') as fh:
        fh.write(
            'Per-deployment ablation convergence figures.\n\n'
            'deployment_NN/  -- one folder per actual-10 paper deployment '
            '(NN = seed 1..20).\n'
            '  _overlay_all_arms.pdf  -- ALL arms on one axis; the '
            'same-deployment comparison.\n'
            '  L<rank>_<rung>/convergence_over_iterations.pdf  -- one arm, '
            'with painter + OPP reference lines.\n\n'
            'Capability order (ascending; L6 = SCULPTOR):\n'
            '  L1 painter        one-shot baseline (no training; ref line only)\n'
            '  L2 no_mc          monte-carlo off (painter + one-flip search)\n'
            '  L3 no_memory      monte-carlo on, no memory\n'
            '  L4 no_memory_dir  + direction (flip-threshold step)\n'
            '  L5 expl_none      + memory (continuous advertisement)\n'
            '  L6 full           + entropic exploration = SCULPTOR\n\n'
            'y = ground-truth objective (lower better); the trained-objective\n'
            'series (steady + gamma*resilience) each arm optimized. full/L6\n'
            'curves recovered from the paper convergence figures; the rest\n'
            'from the ablation cell JSONs.\n')
    print('wrote {} single-arm figures + {} overlays under {}'.format(
        n_fig, len(seeds), a.out_dir))


# ================================ entry point ================================
_SUBCOMMANDS = {'run': None, 'evaluate': None, 'selftest': None, 'tree': None}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        print(__doc__)
        print('usage: python -m evaluations.ablation {run|evaluate|selftest|tree} [args]')
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == 'run':
        return run_main(rest)
    if cmd == 'evaluate':
        return evaluate_main(rest)
    if cmd == 'selftest':
        ap = argparse.ArgumentParser(prog='ablation selftest')
        ap.add_argument('--iters', type=int, default=16)
        ap.add_argument('--probe-n', type=int, default=4)
        ap.add_argument('--deployments', type=int, default=2)
        ap.add_argument('--keep', action='store_true')
        a = ap.parse_args(rest)
        return 0 if selftest(a.iters, a.probe_n, a.deployments, a.keep) else 1
    return tree_main(rest)


if __name__ == '__main__':
    sys.exit(main())
