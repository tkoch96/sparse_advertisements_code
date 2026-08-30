"""Unit tests for core/depstore.py: hash stability, over-keying,
min-iters lookup, eval-era keying, checksum refusal, local budget."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import depstore  # noqa: E402


def _clean_env():
    for k in list(os.environ):
        if k.startswith('SCULPTOR_'):
            del os.environ[k]


def test_fingerprint_stability_and_sensitivity():
    _clean_env()
    fp1, _ = depstore.fingerprint()
    fp2, _ = depstore.fingerprint()
    assert fp1 == fp2, 'same env must hash identically'
    os.environ['SCULPTOR_DEPLOYMENT_SEED'] = '7'
    fp3, _ = depstore.fingerprint()
    assert fp3 != fp1, 'semantic knob must change the hash'
    del os.environ['SCULPTOR_DEPLOYMENT_SEED']
    os.environ['SCULPTOR_N_WORKERS'] = '64'
    fp4, _ = depstore.fingerprint()
    assert fp4 == fp1, 'operational knob must NOT change the hash'
    del os.environ['SCULPTOR_N_WORKERS']
    print('PASS fingerprint stability/sensitivity')


def test_overkeying_unknown_knob():
    _clean_env()
    fp1, _ = depstore.fingerprint(_warn=False)
    os.environ['SCULPTOR_SOME_FUTURE_KNOB'] = 'x'
    fp2, _ = depstore.fingerprint(_warn=False)
    assert fp2 != fp1, 'unknown SCULPTOR_* must over-key (unique hash)'
    del os.environ['SCULPTOR_SOME_FUTURE_KNOB']
    print('PASS over-keying of unknown knobs')


def test_config_override():
    _clean_env()
    fp1, _ = depstore.fingerprint({'dpsize': 'small'}, _warn=False)
    fp2, _ = depstore.fingerprint({'dpsize': 'actual-32'}, _warn=False)
    assert fp1 != fp2
    print('PASS explicit config keys')


def test_put_get_min_iters(root):
    _clean_env()
    st = depstore.Depstore(root=root)
    adv = np.random.rand(51, 6)
    cfg = {'dpsize': 'small'}
    fp = st.put_training(adv, 200, deployment={'toy': 1}, config=cfg)
    assert fp
    # 150-iter request satisfied by the 200-iter artifact
    art = st.get_training(min_iters=150, config=cfg)
    assert art is not None and art.n_iters == 200
    assert np.allclose(art.adv, adv)
    assert art.deployment == {'toy': 1}
    # 250-iter request must MISS
    assert st.get_training(min_iters=250, config=cfg) is None
    msg = st.why_miss(min_iters=250, config=cfg)
    assert 'n_iters' in msg, msg
    # smaller artifact preferred when both satisfy
    st.put_training(adv, 300, config=cfg)
    art2 = st.get_training(min_iters=150, config=cfg)
    assert art2.n_iters == 200, 'must serve the smallest sufficient'
    print('PASS put/get + min-iters semantics')


def test_eval_era_keying(root):
    _clean_env()
    st = depstore.Depstore(root=root)
    fp = st.put_training(np.zeros((3, 2)), 50, config={'dpsize': 'small'})
    st.put_eval(fp, 'flash_bisect', {'crit': 33.6})
    got = st.get_eval(fp, 'flash_bisect')
    assert got == {'crit': 33.6}
    # era bump invalidates
    old = depstore.EVAL_ERAS['flash_bisect']
    depstore.EVAL_ERAS['flash_bisect'] = 'e99-test'
    try:
        assert st.get_eval(fp, 'flash_bisect') is None, \
            'era bump must invalidate stored evals'
    finally:
        depstore.EVAL_ERAS['flash_bisect'] = old
    print('PASS eval-era keying')


def test_checksum_refusal(root):
    _clean_env()
    st = depstore.Depstore(root=root)
    cfg = {'dpsize': 'corrupt-test'}
    st.put_training(np.ones((4, 4)), 10, config=cfg)
    art = st.get_training(min_iters=10, config=cfg)
    p = os.path.join(root, 'trainings')
    tdir = [d for d in os.listdir(p) if d.endswith('_it10')]
    with open(os.path.join(p, tdir[0], 'adv.npy'), 'ab') as f:
        f.write(b'CORRUPT')
    assert st.get_training(min_iters=10, config=cfg) is None, \
        'corrupted artifact must be refused'
    print('PASS checksum refusal')


def test_local_budget(root):
    _clean_env()
    os.environ['SCULPTOR_DEPSTORE_LOCAL'] = '1'
    os.environ['SCULPTOR_DEPSTORE_BUDGET_MB'] = '2'
    try:
        st = depstore.Depstore(root=root)
        big = np.zeros((600, 600))  # ~2.9 MB > half of 2 MB budget
        assert st.put_training(big, 5, config={'x': 'big'}) is None, \
            'oversize put must be refused in local mode'
        # several small puts must trigger LRU eviction, staying <budget
        for i in range(10):
            fp = st.put_training(np.zeros((180, 180)), 5,
                                 config={'x': 'n{}'.format(i)})
            assert fp is not None, 'small put {} wrongly refused'.format(i)
        assert st._dir_size_mb() <= 2.2, st._dir_size_mb()
        n_dirs = len(os.listdir(os.path.join(root, 'trainings')))
        assert n_dirs < 10, 'LRU eviction never fired ({} dirs)'.format(n_dirs)
        # newest artifact must still be servable
        assert st.get_training(min_iters=5, config={'x': 'n9'}) is not None
        print('PASS local budget (refusal + LRU eviction, {} survivors)'
              .format(n_dirs))
    finally:
        del os.environ['SCULPTOR_DEPSTORE_LOCAL']
        del os.environ['SCULPTOR_DEPSTORE_BUDGET_MB']


def main():
    base = tempfile.mkdtemp(prefix='depstore_test_')
    try:
        test_fingerprint_stability_and_sensitivity()
        test_overkeying_unknown_knob()
        test_config_override()
        test_put_get_min_iters(os.path.join(base, 'a'))
        test_eval_era_keying(os.path.join(base, 'b'))
        test_checksum_refusal(os.path.join(base, 'c'))
        test_local_budget(os.path.join(base, 'd'))
        print('\nALL DEPSTORE TESTS PASS')
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    main()
