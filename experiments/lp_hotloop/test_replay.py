"""Replay the fixture against the current code: bitwise output match +
per-solve timing. Run under PYTHONHASHSEED=0 (gen and replay both)."""
import os, sys, pickle, time
import numpy as np
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _R)
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
from experiments.lp_hotloop.gen_fixture import build_worker

if __name__ == '__main__':
    recs = pickle.load(open(os.path.join(_R,'experiments/lp_hotloop/fixture_decent.pkl'),'rb'))
    w = build_worker()
    ok, times = True, []
    for i, r in enumerate(recs):
        np.random.seed(5000 + i)
        t0 = time.time()
        b, (x, px) = w.latency_benefit(r['adv'].copy(), retnow=True, generic_obj='avg_latency')
        times.append(time.time() - t0)
        match = (float(b) == r['benefit'] and float(np.sum(px)) == r['pdf_sum']
                 and float(np.sum(x*px)) == r['exp'])
        ok = ok and match
        if not match:
            print('MISMATCH call {}: got {} want {}'.format(i, float(b), r['benefit']))
    t = np.array(times[1:])  # first call pays one-time warmup
    print('replay: {} calls, match={}, per-solve p50={:.2f}s p90={:.2f}s'.format(
        len(recs), ok, np.percentile(t,50), np.percentile(t,90)))
    sys.exit(0 if ok else 1)
