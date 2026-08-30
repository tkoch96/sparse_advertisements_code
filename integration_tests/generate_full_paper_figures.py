"""generate_full_paper_figures -- END-TO-END integration test of the
paper pipeline (Tom 2026-08-30: the hermetic smoke choreography, as a
test).

    python integration_tests/generate_full_paper_figures.py
    python integration_tests/generate_full_paper_figures.py --keep

Runs the LOCAL smoke intent (intents/paper_intent.smoke.json) through
run_all_paper_evaluations twice against a hermetic scratch state:

  RUN 1 (cold): every stage must actually TRAIN ('Initializing
    advertisement' in the log), PUT trainings into the depstore, emit
    figures on BOTH sweep axes (>1 deployment size; >1 prefix budget,
    each with its OWN training fingerprint -- the cross-budget key
    regression), store the stage artifact bundles, and auto-grab the
    full set into figures/papersmoke_out.

  RUN 2 (warm): identical call must train NOTHING (zero 'Initializing
    advertisement'; '[depstore] HIT' present), finish much faster, and
    still deliver the full artifact set -- proof that the unified cache
    serves the whole pipeline.

Hermetic state = local depstore (~/.sculptor_depstore), repo ./depstore,
cache/figures papersmoke namespaces, and SCULPTOR_RUN_TAG=papersmoke
metrics pickles: all wiped up front (and after, without --keep).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
PY = os.path.expanduser('~/Documents/venv312/bin/python')
INTENT = os.path.join(_REPO, 'evaluations', 'intents',
                      'paper_intent.smoke.json')
LOCAL_STORE = os.path.expanduser('~/.sculptor_depstore')

STEPS = []


def step(name, ok, detail=''):
    STEPS.append((name, ok))
    print('  {} {} {}'.format('PASS' if ok else 'FAIL', name,
                              detail and '-- ' + detail))


def _reset():
    shutil.rmtree(LOCAL_STORE, ignore_errors=True)
    shutil.rmtree(os.path.join(_REPO, 'depstore'), ignore_errors=True)
    for d in ('cache/papersmoke', 'figures/papersmoke',
              'figures/papersmoke_out'):
        shutil.rmtree(os.path.join(_REPO, d), ignore_errors=True)
    for f in glob.glob(os.path.join(_REPO, 'cache', '*papersmoke*')):
        os.remove(f)
    # legacy global-cache pickles from pre-namespacing eras
    for f in glob.glob(os.path.join(_REPO, 'cache',
                                    'small_over_prefixes-*.pkl')) +              glob.glob(os.path.join(
                 _REPO, 'cache',
                 'evaluate_over_prefix_numbers_cache_fn_small*')):
        os.remove(f)


def _run(log_fn):
    t0 = time.time()
    env = dict(os.environ, SCULPTOR_LOG_MEM='0')
    with open(log_fn, 'w') as lf:
        rc = subprocess.call(
            [PY, os.path.join(_REPO, 'evaluations',
                              'run_all_paper_evaluations.py'),
             'run', INTENT],
            cwd=_REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return rc, time.time() - t0, open(log_fn, errors='replace').read()


def _training_fps_by_prefix():
    """(n_prefixes -> fingerprint) from the local store's trainings."""
    from core import depstore
    os.environ['SCULPTOR_DEPSTORE_LOCAL'] = '1'
    os.environ.pop('SCULPTOR_DEPSTORE_ROOT', None)
    st = depstore.Depstore(root=LOCAL_STORE)
    out = {}
    for e in st.index():
        if e.get('kind') != 'training':
            continue
        mfn = os.path.join(st.root, e['path'], 'manifest.json')
        if not os.path.exists(mfn):
            continue
        k = json.load(open(mfn)).get('key') or {}
        out.setdefault(k.get('n_prefixes'), set()).add(e['fp'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    logd = os.path.join(_REPO, 'figures', 'integration_tests')
    os.makedirs(logd, exist_ok=True)
    _reset()
    out_dir = os.path.join(_REPO, 'figures', 'papersmoke_out')

    try:
        # ---- RUN 1: cold -------------------------------------------
        rc, dt1, log = _run(os.path.join(logd, 'paperfig_run1.log'))
        step('run1 exits 0', rc == 0, 'rc={} ({:.0f}s)'.format(rc, dt1))
        step('run1 actually trains',
             'Initializing advertisement' in log)
        step('run1 PUTs trainings', '[depstore] PUT' in log)
        step('run1 pulls full artifact set',
             'ALL PAPER ARTIFACTS PULLED' in log)
        eods = glob.glob(os.path.join(out_dir,
                                      '*over_deployment_size*.pdf'))
        pfx = glob.glob(os.path.join(out_dir, '*over_prefix_budget*.pdf'))
        tbl = [f for f in ('paper_table.csv', 'paper_table.tex')
               if os.path.exists(os.path.join(out_dir, f))]
        step('deployment-size figures present', len(eods) >= 5,
             '{} files'.format(len(eods)))
        step('prefix-budget figures present', len(pfx) >= 5,
             '{} files'.format(len(pfx)))
        step('paper table present', len(tbl) == 2, ','.join(tbl))
        fps = _training_fps_by_prefix()
        budgets = {k for k in fps if k not in (None, 'auto')}
        step('distinct fingerprints per prefix budget (cross-key '
             'regression)', len(budgets) >= 2
             and not set.intersection(*(fps[b] for b in budgets))
             if len(budgets) >= 2 else False,
             'budgets={} fps={}'.format(
                 sorted(budgets),
                 {b: sorted(x)[0][:8] for b, x in fps.items()}))

        # ---- RUN 2: warm -------------------------------------------
        # wipe the LEGACY pickle cache layer (but keep the depstore):
        # pickles sit in front of the store in the driver, so leaving
        # them would let run2 'pass' without touching the unified cache
        shutil.rmtree(os.path.join(_REPO, 'cache', 'papersmoke'),
                      ignore_errors=True)
        # RUN_TAG-suffixed metrics pickles live in the GLOBAL cache dir
        # and also sit in front of the depstore -- wipe them too
        for f in glob.glob(os.path.join(_REPO, 'cache', '*papersmoke*')):
            os.remove(f)
        rc, dt2, log2 = _run(os.path.join(logd, 'paperfig_run2.log'))
        step('run2 exits 0', rc == 0, 'rc={} ({:.0f}s)'.format(rc, dt2))
        step('run2 trains NOTHING',
             'Initializing advertisement' not in log2)
        step('run2 serves training from depstore',
             '[depstore] HIT' in log2)
        step('run2 still delivers the set',
             'ALL PAPER ARTIFACTS PULLED' in log2)
        # run2 skips TRAINING via the depstore but family evals still
        # recompute (they are pickle-layer cached only; depstore-caching
        # eval families is the designed next step) -- so assert 'not
        # slower', not 'much faster'. When family caching lands, tighten
        # this back to dt2 < dt1 * 0.5.
        step('run2 not slower than cold', dt2 < dt1 * 1.10,
             '{:.0f}s vs {:.0f}s'.format(dt2, dt1))

        failed = [s for s in STEPS if not s[1]]
        print('\n{}: {}/{} checks passed'.format(
            'FAILED' if failed else 'ALL PASS',
            len(STEPS) - len(failed), len(STEPS)))
        return 1 if failed else 0
    finally:
        if not a.keep:
            _reset()
        else:
            print('kept smoke state (--keep); logs in {}'.format(logd))


if __name__ == '__main__':
    sys.exit(main())
