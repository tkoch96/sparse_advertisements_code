"""Ablation ladder CDF over many deployments, PAPER-PARITY config
(Tom 2026-08-30: "CDF of final objective values for each L-value over
many deployments").

This deliberately does NOT go through run_ablation_grid: the grid pins
world=maxhard + SCULPTOR_XOBJS=1 (the hardness study), while this
experiment must run the DEFAULT world with mainline probing defaults
(smart, N=10) so the 'full' rung is byte-for-byte the paper's SCULPTOR
-- and can therefore consult the depstore for trainings the paper evals
already computed (run_fork_ladder's L6 depstore hook).

    python evaluations/run_ablation_cdf.py \
        --dpsize testing_feature-actual-3 --deployments 10 \
        --max-iter 100 --slots 8 --out-root cache/ablation/cdf_a3_smoke \
        --ws-root /tmp/abl_cdf_ws

Pipeline: canonical inits -> run_n_sweep_queue (single-spec manifest,
audit + trusted rescore included) -> cdf_fork (diagnostic 2-panel +
--paper-out figure) -> quantile table. Everything after the queue is
idempotent re-runnable.
"""
import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# active ladder for the paper CDF: no_direction retired 2026-08-18,
# expl_random retired 2026-08-12
# WHEN-to-measure policy PER RUNG (Tom 2026-09-08: the ablation overrides the
# solver defaults for every rung explicitly; nothing rides on a default).
# WHAT each rung optimizes with (memory / direction / explore / MC) is defined
# once in experiments/ablation/sculptor_fork.RUNGS; this table is the probing
# policy of the same ladder, and evaluations/evaluate_ablation.py verifies
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
    (mirrors run_ablation_grid._ensure_inits, which draws by seed)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dpsize', required=True)
    ap.add_argument('--deployments', type=int, default=10,
                    help='seeds 1..K')
    ap.add_argument('--max-iter', type=int, default=100)
    ap.add_argument('--gamma', default='4',
                    help="training gamma; '4' matches wrapper_eval / the "
                         'paper evals (choke_config parity)')
    ap.add_argument('--probe-n', default='10',
                    help='measurement budget; 10 = what the 2026-09 paper '
                         'evals and the cdf_a10 campaign trained with (kept '
                         'explicit for depstore/choke_config parity). The '
                         "mainline has no constant default any more: unset "
                         "= 'prefixes', one per prefix of each deployment.")
    ap.add_argument('--rungs', default=PAPER_RUNGS)
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
                         'ablation_cell.harvest_figs copies each cell\'s '
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
    a = ap.parse_args()

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
            from run_ablation_grid import _ensure_inits
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

        base_env = ({'SCULPTOR_ABLATION_DEP_FILE': dep_tpl}
                    if dep_tpl else {})
        if a.continue_from:
            prior = os.path.abspath(a.continue_from)
            # RESUME_FROM: hot-start each arm from its prior final adv. One
            # spec per WHEN policy from LADDER_PROBE_MODE (same table as the
            # fresh branch below). SCULPTOR_PROBE_N/TCONV are the MAINLINE
            # names so the scrubbed 'full' rung gets the same remaining budget.
            _rungs = [r for r in a.rungs.split(',') if r]
            specs = []
            for mode in ('smart', 'scheduled'):
                group = [r for r in _rungs if LADDER_PROBE_MODE.get(r) == mode]
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
            # One spec per WHEN-to-measure policy (Tom 2026-09-08): the fork
            # rungs L1-L5 (+ painter, one-shot, no probing) run on the fixed
            # schedule; 'full' (L6 = SCULPTOR) gets the smart gate. Same
            # label/out_root, so harvested names and the result dir are
            # unchanged. PAPER PARITY: default world, no XOBJS. Dep-file
            # mode rides in spec env so cells AND the queue's rescore see it.
            _rungs = [r for r in a.rungs.split(',') if r]
            _unknown = [r for r in _rungs if r not in LADDER_PROBE_MODE]
            assert not _unknown, 'rungs without a LADDER_PROBE_MODE entry: {}'.format(_unknown)
            _base = {
                'label': 'cdf_{}'.format(a.dpsize.replace('/', '_')),
                'out_root': out_root,
                'init_src': init_dir,
                'seeds': '1-{}'.format(a.deployments),
                'n_values': str(a.probe_n),
                'gamma': str(a.gamma),
                'max_iter': a.max_iter,
                'dpsize': a.dpsize,
                'artifacts_figs': figs_dir,
            }
            specs = []
            # one queue spec per (probe policy); same label/out_root, so the
            # harvested names and the result dir are unchanged
            for mode in ('smart', 'scheduled', None):
                group = [r for r in _rungs if LADDER_PROBE_MODE[r] == mode]
                if not group:
                    continue
                if 'full' in group:
                    # SCULPTOR_PROBE_N/MODE are the MAINLINE names (the fork env
                    # is scrubbed for 'full'), so L6 gets the same budget as
                    # L1-L5 for numeric --probe-n too ('prefixes' resolves per
                    # deployment).
                    env = dict(base_env, SCULPTOR_PROBE_MODE=mode,
                               SCULPTOR_PROBE_N=str(a.probe_n))
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
        [sys.executable, '-u', '-m', 'evaluations.evaluate_ablation',
         '--in-dir', in_dir, '--ws-root', ws_root,
         '--dpsize', a.dpsize, '--deployments', str(a.deployments),
         '--max-iter', str(a.max_iter), '--probe-n', str(a.probe_n),
         '--rungs', a.rungs],
        cwd=_REPO)
    if rc_eval != 0:
        print('[cdf] EVALUATION/VERIFICATION FAILED (rc={}) -- see verification.txt in {}'.format(rc_eval, in_dir), flush=True)
    return rc_cdf or rc_eval


if __name__ == '__main__':
    sys.exit(main())
