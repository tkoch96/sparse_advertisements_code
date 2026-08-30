"""Global deployment/eval artifact store (Tom 2026-08-30).

Once a deployment is trained under a given semantic configuration, the
trained advertisement and its evaluation results are cached globally and
looked up by CONFIG FINGERPRINT instead of by pickle-path convention.
This kills the whole class of failures from the 2026-08 campaigns:
May-era caches silently mismatching, run-tag namespace confusion,
placeholder evals surviving because nothing knew they were stale, env
knobs silently changing semantics between runs.

Two-level cache
---------------
  trainings/<fp>/          one trained artifact per (fingerprint, n_iters)
  evals/<fp>/<family>@<eval_era>/
                           eval results are keyed by the TRAINING
                           fingerprint x eval family x eval-code era, so
                           an eval-code fix invalidates evals without
                           invalidating training ("re-evaluate, don't
                           retrain" as a first-class cache op).

Fingerprint policy: OVER-KEY, never under-key (prefer-underfit rule: a
wrong cache hit is silent and permanent; a wrong miss just retrains).
The hash covers a DECLARED list of semantic knobs; any SCULPTOR_* env
var present at fingerprint time that is neither declared semantic nor
declared operational is folded into the hash too, with a loud warning to
go classify it.

Iteration semantics: n_iters is NOT part of the hash. Artifacts record
their actual n_iters; get_training(min_iters=150) returns the smallest
stored artifact with n_iters >= 150 for that fingerprint.

Local mode (SCULPTOR_DEPSTORE_LOCAL=1): a fully local store for smoke
tests on small sizes, hard-capped (default 50 MB total): oversize puts
are refused loudly and the store LRU-evicts to stay under budget.

Write safety: temp file -> sha256 -> atomic rename -> append-only index
line. Reads verify checksums. `cluster/depstorectl.py` adds fsck,
ls, why-miss, ingest, and the S3 mirror.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np

# Bump when TRAINING semantics change in a way the knobs don't capture
# (e.g. the 2026-08-28 penalty contract would have bumped this).
CORE_ERA = 'era1-20260830-peruser-penalty'

# Bump per eval family when that family's EVAL code changes meaning.
EVAL_ERAS = {
    'compare_ret': 'e1-20260830',
    'failure_grids': 'e2-20260830-highlat-split',
    'flash_bisect': 'e2-20260823-reference-first',
    'diurnal_bisect': 'e2-20260823-reference-first',
    'latencies': 'e1',
    'volume': 'e1',
    'pct_vol': 'e1',
    'mlu': 'e1',
}

# ---- knob classification (from the 2026-08-30 audit of core/) ----------
# Semantic: changes what a trained advertisement IS. name -> default (the
# value assumed when the env var is unset; defaults mirror the code).
SEMANTIC_KNOBS = {
    'SCULPTOR_DEPLOYMENT_SEED': '',
    'SCULPTOR_GENERIC_OBJECTIVE': 'avg_latency',
    'SCULPTOR_LAT_MODEL': '',
    'SCULPTOR_PREF_MODEL': '',
    'SCULPTOR_GEO_NOISE': '',
    'SCULPTOR_VOL_SPREAD': '',
    'SCULPTOR_SCALE_FACTOR': '',
    'SCULPTOR_LAT_SPREAD': '',
    'SCULPTOR_ZIPF': '',
    'SCULPTOR_CAPACITY_HEADROOM': '',
    'SCULPTOR_ALPHA': '',
    'SCULPTOR_ALPHA0': '',
    'SCULPTOR_ALPHA_POP': '',
    'SCULPTOR_ALPHA_POP_ANNEAL_END_ITER': '',
    'SCULPTOR_RMSPROP_BETA': '',
    'SCULPTOR_ADAGRAD_WARMUP_SKIP': '',
    'SCULPTOR_ADAPTIVE_PROBE_BUDGET': '',
    'SCULPTOR_GRAD_SCALE': '',
    'SCULPTOR_LB_GRAD_BUDGET_SCALE': '1.0',
    'SCULPTOR_LB_CACHE': '1',
    'SCULPTOR_MC_NUM': '',
    'SCULPTOR_MC_NUM_EXPLORE': '',
    'SCULPTOR_BELIEF_MEMO': '',
    'SCULPTOR_SIGMA_REFRESH': '',
    'SCULPTOR_SIG_CUTOFF': '',
    'SCULPTOR_SURPRISE_THETA': '',
    'SCULPTOR_STOP_RULE': '',
    'SCULPTOR_STOP_V2_IMP': '',
    'SCULPTOR_STOP_V2_INIT': '',
    'SCULPTOR_STOP_V2_PATIENCE': '',
    'SCULPTOR_STOP_V2_REL': '',
    'SCULPTOR_STOP_V2_TREND_EPS': '',
    'SCULPTOR_STOP_CONFIRM_PROBE': '',
    'SCULPTOR_STOP_DROP_ADV_DELTA': '',
    'SCULPTOR_REMEASURE_STOP': '',
    'SCULPTOR_PROBE_MODE': '',
    'SCULPTOR_PROBE_N': '',
    'SCULPTOR_PROBE_TCONV': '',
    'SCULPTOR_MAXINFO_TARGET': '',
    'SCULPTOR_STARTUP_RB': '',
    'SCULPTOR_OPP_ONCE': '',
    'SCULPTOR_GT_RB': '0',
    'SCULPTOR_USE_RESILIENCE': '1',
    'SCULPTOR_XOBJS': '',
    'SCULPTOR_OBJ_ROUND': '',
    'SCULPTOR_OBJ_MAXUTIL_ALPHA': '',
    'SCULPTOR_MLU_WEIGHT_MULT': '',
    'SCULPTOR_LATMLU_TERM': '',
    'SCULPTOR_LATMLU_STRAND_MULT': '',
    'SCULPTOR_HINGE_NOROUTE_MS': '',
    'SCULPTOR_FRACB_SCALAR': '',
    'SCULPTOR_FRAC_BEYOND_REL': '',
    'SCULPTOR_FROZEN_GAMMA': '',
    'SCULPTOR_FROZEN_WHICH': '',
    'SCULPTOR_NO_ROUTE_LATENCY': '30000',
    'SCULPTOR_NO_ROUTE_PENALTY_MULT': '2.0',
    'SCULPTOR_CONGESTED_PENALTY_MULT': '1.5',
    'SCULPTOR_NO_ROUTE_PENALTY_MS': '800',
    'SCULPTOR_CONGESTED_PENALTY_MS': '350',
    'SCULPTOR_BULK_SLACK_DOM': '1e3',
    'SCULPTOR_SOFT_CONG_PENALTY': '50',
    'SCULPTOR_CONGESTION_AWARE_OBJ': '1',
    'SCULPTOR_ROUTE_VIOLATION': '',
    'SCULPTOR_LP_BACKEND': 'gurobi',
    'SCULPTOR_OPP_FROZEN': '',
    'SCULPTOR_PAINTER_MEASURE_CAP': '',
    'SCULPTOR_ABLATION_ALPHA': '',
    'SCULPTOR_ABLATION_ALPHA0': '',
    'SCULPTOR_ABLATION_DOG_EPS': '',
    'SCULPTOR_ABLATION_GRAD_SCALE': '',
    'SCULPTOR_ABLATION_MC': '',
    'SCULPTOR_ABLATION_SIGMA_REFRESH': '',
    'SCULPTOR_ABLATION_SURPRISE_THETA': '',
    'SCULPTOR_ABLATION_OBJECTIVE': '',
    'SCULPTOR_ABLATION_PROBE_MODE': '',
    'SCULPTOR_ABLATION_PROBE_N': '',
    'SCULPTOR_ABLATION_PROBE_TCONV': '',
    'SCULPTOR_EVAL_VOLSCEN': '',
}

# Operational: affects wall time, memory, logging -- never the result.
OPERATIONAL_KNOBS = {
    'SCULPTOR_N_WORKERS', 'SCULPTOR_MIN_WORKERS',
    'SCULPTOR_N_WORKERS_DURING_PARALLEL', 'SCULPTOR_AUTOSCALE_WORKERS',
    'SCULPTOR_WORKER_INIT_STAGGER_SEC', 'SCULPTOR_WORKER_MEMPROF',
    'SCULPTOR_WORKER_MEM_LOG_DIR', 'SCULPTOR_WORKER_NODE_HEADROOM_MB',
    'SCULPTOR_WORKER_REBIRTH_EVERY', 'SCULPTOR_WORKER_REFRESH_FULL_EVERY',
    'SCULPTOR_LOG_MEM', 'SCULPTOR_LOG_OBJSIZE', 'SCULPTOR_VERBOSE_ERRORS',
    'SCULPTOR_VERBOSE_PLOT_ERRORS', 'SCULPTOR_VOLDEBUG',
    'SCULPTOR_MEM_HEADROOM_MB', 'SCULPTOR_HEAD_RAM_HEADROOM_MB',
    'SCULPTOR_STRATEGY_RSS_ESTIMATE_MB', 'SCULPTOR_MALLOC_TRIM',
    'SCULPTOR_LAT_SHARDS', 'SCULPTOR_LAT_SHARDS_AUTOBUILD',
    'SCULPTOR_DEPSETUP_ARRAYS', 'SCULPTOR_COMPACT_PT', 'SCULPTOR_COMPACT_RB',
    'SCULPTOR_DISABLE_PARALLEL_STRATEGIES', 'SCULPTOR_DISABLE_RAW_X_BATCH',
    'SCULPTOR_LP_FORCE_NONPERSISTENT', 'SCULPTOR_LP_INCREMENTAL',
    'SCULPTOR_LP_INCR_MLU', 'SCULPTOR_LP_ADAPTIVE_MLU',
    'SCULPTOR_LP_SOLVE_DEBUG', 'SCULPTOR_GPSHIM_AUDIT',
    'SCULPTOR_GPSHIM_DUAL', 'SCULPTOR_GPSHIM_DUAL_OUT', 'SCULPTOR_GRB_DUMP',
    'SCULPTOR_ADAGRAD_LOG', 'SCULPTOR_STARTUP_TIMELOG',
    'SCULPTOR_RAY_NUM_CPUS', 'SCULPTOR_RECOVER_NODE_TIMEOUT_S',
    'SCULPTOR_REQUIRE_SOLNS', 'SCULPTOR_MIN_MLU_TIMELIMIT',
    'SCULPTOR_OBJSIZE_BUDGET', 'SCULPTOR_OBJSIZE_SAMPLE_K',
    'SCULPTOR_OBJSIZE_SAMPLE_MIN', 'SCULPTOR_EVAL_SEED',
    'SCULPTOR_CELL_TIMEOUT', 'SCULPTOR_RECALC', 'SCULPTOR_QUEUE_PASSES',
    'SCULPTOR_MAX_ITER', 'SCULPTOR_MIN_ITER',   # iters handled separately
    'SCULPTOR_DEPSTORE', 'SCULPTOR_DEPSTORE_LOCAL',
    'SCULPTOR_DEPSTORE_ROOT', 'SCULPTOR_DEPSTORE_BUDGET_MB',
    'SCULPTOR_DEPSTORE_S3', 'SCULPTOR_EODS_INCLUDE_SIZE3',
    'SCULPTOR_EODS_HOTSTART_DIR', 'SCULPTOR_DEPLOYMENT_SWEEP_NSIM',
}


def deployment_id(deployment):
    """Content identity of a deployment: hash of its popps, ugs and
    volumes. Guarantees a fingerprint can never collide across two
    different (e.g. randomly drawn) deployments -- seeded runs reproduce
    the same deployment and therefore the same id."""
    try:
        popps = sorted(str(p) for p in deployment.get('popps', []))
        ugs = sorted(str(u) for u in deployment.get('ugs', []))
        vols = deployment.get('ug_to_vol') or {}
        vol_sig = sorted((str(u), round(float(v), 6))
                         for u, v in vols.items())
        blob = json.dumps([popps, ugs, vol_sig])
    except Exception:
        blob = repr(sorted(deployment.keys()))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fingerprint(config=None, _warn=True):
    """sha256 over CORE_ERA + declared semantic knobs (env view merged
    with explicit `config` overrides) + any UNKNOWN SCULPTOR_* env vars
    (over-keying). Returns (fp_hex, key_dict)."""
    key = {'CORE_ERA': CORE_ERA}
    for k, dflt in SEMANTIC_KNOBS.items():
        key[k] = os.environ.get(k, dflt)
    unknown = [k for k in os.environ
               if k.startswith('SCULPTOR_')
               and k not in SEMANTIC_KNOBS and k not in OPERATIONAL_KNOBS]
    for k in sorted(unknown):
        key[k] = os.environ[k]
        if _warn:
            print('[depstore] WARNING: unclassified knob {} folded into '
                  'the fingerprint (over-keying). Classify it in '
                  'core/depstore.py.'.format(k), flush=True)
    for k, v in (config or {}).items():
        key[str(k)] = '' if v is None else str(v)
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:20], key


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class DeploymentArtifact:
    """The globally understood result of training one deployment: the
    deployment itself, the trained advertisement, provenance, and any
    attached eval families."""

    def __init__(self, fp, key, n_iters, adv, deployment=None,
                 provenance=None, evals=None):
        self.fp = fp
        self.key = key
        self.n_iters = int(n_iters)
        self.adv = np.asarray(adv) if adv is not None else None
        self.deployment = deployment
        self.provenance = provenance or {}
        self.evals = evals or {}

    def __repr__(self):
        return ('DeploymentArtifact(fp={}, n_iters={}, evals={})'
                .format(self.fp, self.n_iters, sorted(self.evals)))


class Depstore:
    def __init__(self, root=None):
        if root is None:
            if os.environ.get('SCULPTOR_DEPSTORE_LOCAL', '0') == '1':
                root = os.environ.get(
                    'SCULPTOR_DEPSTORE_ROOT',
                    os.path.expanduser('~/.sculptor_depstore'))
                self.local = True
            else:
                root = os.environ.get(
                    'SCULPTOR_DEPSTORE_ROOT',
                    os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), 'depstore'))
                self.local = False
        else:
            self.local = os.environ.get('SCULPTOR_DEPSTORE_LOCAL',
                                        '0') == '1'
        self.root = root
        self.budget_mb = float(os.environ.get(
            'SCULPTOR_DEPSTORE_BUDGET_MB', '50' if self.local else '0'))
        os.makedirs(os.path.join(root, 'trainings'), exist_ok=True)
        os.makedirs(os.path.join(root, 'evals'), exist_ok=True)

    # ---- index -----------------------------------------------------
    @property
    def index_path(self):
        return os.path.join(self.root, 'index.jsonl')

    def _index_append(self, entry):
        entry['ts'] = time.time()
        with open(self.index_path, 'a') as f:
            f.write(json.dumps(entry, sort_keys=True) + '\n')

    def index(self):
        out = []
        if os.path.exists(self.index_path):
            for line in open(self.index_path):
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        return out

    # ---- atomic write helpers --------------------------------------
    def _write_atomic(self, path, writer):
        tmp = path + '.tmp{}'.format(os.getpid())
        writer(tmp)
        os.replace(tmp, path)
        return _sha256_file(path)

    def _dir_size_mb(self):
        total = 0
        for dirpath, _d, files in os.walk(self.root):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total / 1e6

    def _enforce_budget(self, incoming_mb):
        """Local mode: refuse oversize puts, LRU-evict to stay <budget."""
        if not self.budget_mb:
            return True
        if incoming_mb > self.budget_mb * 0.5:
            print('[depstore] REFUSED: artifact {:.1f} MB exceeds half '
                  'the local budget ({} MB). Local mode is for small-size '
                  'smokes; use the primary store for this.'.format(
                      incoming_mb, self.budget_mb), flush=True)
            return False
        while self._dir_size_mb() + incoming_mb > self.budget_mb:
            entries = [e for e in self.index() if e.get('path')]
            entries = [e for e in entries
                       if os.path.exists(os.path.join(self.root, e['path']))]
            if not entries:
                break
            victim = min(entries, key=lambda e: e.get('ts', 0))
            vpath = os.path.join(self.root, victim['path'])
            print('[depstore] local budget: evicting {} ({})'.format(
                victim['path'], victim.get('kind')), flush=True)
            shutil.rmtree(vpath, ignore_errors=True)
        return True

    # ---- trainings -------------------------------------------------
    def put_training(self, adv, n_iters, deployment=None, config=None,
                     provenance=None, state_path=None):
        fp, key = fingerprint(config)
        rel = os.path.join('trainings', '{}_it{}'.format(fp, int(n_iters)))
        d = os.path.join(self.root, rel)
        os.makedirs(d, exist_ok=True)
        adv = np.asarray(adv)
        est_mb = adv.nbytes / 1e6 + 0.02   # manifest/deployment overhead
        if state_path and os.path.exists(state_path):
            est_mb += os.path.getsize(state_path) / 1e6
        if not self._enforce_budget(est_mb):
            return None
        checks = {}
        # np.save appends .npy to bare paths; write through an explicit
        # handle so the atomic tmp->rename protocol keeps the exact name
        def _save_adv(p):
            with open(p, 'wb') as f:
                np.save(f, adv)
        checks['adv.npy'] = self._write_atomic(
            os.path.join(d, 'adv.npy'), _save_adv)
        if deployment is not None:
            import pickle
            self._write_atomic(
                os.path.join(d, 'deployment.pkl'),
                lambda p: pickle.dump(deployment, open(p, 'wb')))
            checks['deployment.pkl'] = _sha256_file(
                os.path.join(d, 'deployment.pkl'))
        if state_path and os.path.exists(state_path) and not self.local:
            shutil.copy(state_path, os.path.join(d, 'state-final.pkl'))
            checks['state-final.pkl'] = _sha256_file(
                os.path.join(d, 'state-final.pkl'))
        manifest = {
            'fp': fp, 'key': key, 'n_iters': int(n_iters),
            'checksums': checks,
            'provenance': dict(provenance or {},
                               host=os.uname().nodename,
                               created=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                     time.gmtime())),
        }
        self._write_atomic(
            os.path.join(d, 'manifest.json'),
            lambda p: open(p, 'w').write(json.dumps(manifest, indent=1)))
        self._index_append({'kind': 'training', 'fp': fp,
                            'n_iters': int(n_iters), 'path': rel})
        return fp

    def get_training(self, min_iters=0, config=None, verify=True):
        fp, key = fingerprint(config, _warn=False)
        cands = [e for e in self.index()
                 if e.get('kind') == 'training' and e.get('fp') == fp
                 and e.get('n_iters', 0) >= min_iters]
        if not cands:
            return None
        best = min(cands, key=lambda e: e['n_iters'])
        d = os.path.join(self.root, best['path'])
        mfn = os.path.join(d, 'manifest.json')
        if not os.path.exists(mfn):
            return None
        manifest = json.load(open(mfn))
        if verify:
            for fn, want in manifest.get('checksums', {}).items():
                p = os.path.join(d, fn)
                if not os.path.exists(p) or _sha256_file(p) != want:
                    print('[depstore] checksum FAIL on {}/{} -- refusing '
                          'to serve'.format(best['path'], fn), flush=True)
                    return None
        adv = np.load(os.path.join(d, 'adv.npy'))
        deployment = None
        dpp = os.path.join(d, 'deployment.pkl')
        if os.path.exists(dpp):
            import pickle
            deployment = pickle.load(open(dpp, 'rb'))
        self._index_append({'kind': 'touch', 'fp': fp,
                            'path': best['path']})
        return DeploymentArtifact(fp, manifest['key'],
                                  manifest['n_iters'], adv, deployment,
                                  manifest.get('provenance'))

    # ---- evals -----------------------------------------------------
    def put_eval(self, train_fp, family, payload, config=None,
                 provenance=None):
        era = EVAL_ERAS.get(family, 'e0')
        rel = os.path.join('evals', train_fp, '{}@{}'.format(family, era))
        d = os.path.join(self.root, rel)
        os.makedirs(d, exist_ok=True)
        import pickle
        blob = pickle.dumps(payload)
        if not self._enforce_budget(len(blob) / 1e6):
            return None
        self._write_atomic(os.path.join(d, 'payload.pkl'),
                           lambda p: open(p, 'wb').write(blob))
        manifest = {
            'train_fp': train_fp, 'family': family, 'eval_era': era,
            'checksums': {'payload.pkl': _sha256_file(
                os.path.join(d, 'payload.pkl'))},
            'provenance': dict(provenance or {},
                               host=os.uname().nodename,
                               created=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                     time.gmtime())),
        }
        self._write_atomic(
            os.path.join(d, 'manifest.json'),
            lambda p: open(p, 'w').write(json.dumps(manifest, indent=1)))
        self._index_append({'kind': 'eval', 'fp': train_fp,
                            'family': family, 'eval_era': era,
                            'path': rel})
        return rel

    def get_eval(self, train_fp, family, verify=True):
        era = EVAL_ERAS.get(family, 'e0')
        rel = os.path.join('evals', train_fp, '{}@{}'.format(family, era))
        d = os.path.join(self.root, rel)
        mfn = os.path.join(d, 'manifest.json')
        if not os.path.exists(mfn):
            return None
        manifest = json.load(open(mfn))
        p = os.path.join(d, 'payload.pkl')
        if verify:
            want = manifest['checksums'].get('payload.pkl')
            if not os.path.exists(p) or _sha256_file(p) != want:
                print('[depstore] checksum FAIL on {} -- refusing to '
                      'serve'.format(rel), flush=True)
                return None
        import pickle
        return pickle.load(open(p, 'rb'))

    # ---- diagnostics -----------------------------------------------
    def why_miss(self, min_iters=0, config=None):
        """Explain a training-cache miss: nearest stored fingerprint and
        which key fields differ. Kills the 'why did it retrain?!' class."""
        fp, key = fingerprint(config, _warn=False)
        trainings = {}
        for e in self.index():
            if e.get('kind') != 'training':
                continue
            mfn = os.path.join(self.root, e['path'], 'manifest.json')
            if os.path.exists(mfn):
                trainings[e['fp']] = (json.load(open(mfn)), e)
        if fp in trainings:
            m, e = trainings[fp]
            return ('fingerprint MATCHES {} but n_iters {} < requested {}'
                    .format(fp, e.get('n_iters'), min_iters))
        if not trainings:
            return 'store has no trainings at all'
        best, bdiff = None, None
        for ofp, (m, _e) in trainings.items():
            okey = m.get('key', {})
            diff = {k: (key.get(k), okey.get(k))
                    for k in set(key) | set(okey)
                    if key.get(k) != okey.get(k)}
            if bdiff is None or len(diff) < len(bdiff):
                best, bdiff = ofp, diff
        return ('no match for {}; nearest is {} differing on: {}'
                .format(fp, best, json.dumps(bdiff, sort_keys=True)))
