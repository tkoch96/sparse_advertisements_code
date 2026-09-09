"""Per-objective nsim / env for generate_paper_table (Tom 2026-09-07): a new
objective joins an existing campaign at ITS deployment count and cell env
without re-running the covered cells."""
import os, pytest
os.environ.setdefault('SCULPTOR_XOBJS', '1')
from evaluations import generate_paper_table as gpt

@pytest.mark.unit
def test_nsim_by_objective_parsing_and_lookup():
    t = gpt.parse_nsim_by_objective(1, 'frozen_prefix:3, mlu:2')
    assert t == {'*': 1, 'frozen_prefix': 3, 'max_util': 2}   # alias resolved
    assert gpt.nsim_for(t, 'frozen_prefix') == 3
    assert gpt.nsim_for(t, 'avg_latency') == 1                 # default
    assert gpt.nsim_for(3, 'anything') == 3                    # plain int
    assert gpt.parse_nsim_by_objective(2, {'frozen': 3}) == {'*': 2, 'frozen_prefix': 3}

@pytest.mark.unit
def test_env_by_objective_parsing():
    e = gpt.parse_env_by_objective('{"frozen": {"SCULPTOR_N_WORKERS": 24}}')
    assert e == {'frozen_prefix': {'SCULPTOR_N_WORKERS': '24'}}
    assert gpt.parse_env_by_objective('') == {}

@pytest.mark.unit
def test_pipeline_passes_per_objective_flags():
    from evaluations import run_all_paper_evaluations as rape
    import json
    intent = json.load(open(os.path.join(os.path.dirname(gpt.__file__), 'intents', 'paper_intent.por.json')))
    spec = intent['stages']['paper_table']
    assert spec['nsim_by_objective'] == {'frozen_prefix': 3}
    fn = [f for f in dir(rape) if 'cmd' in f.lower() and callable(getattr(rape, f))]
    assert fn, 'no cmd builder found'


def test_coverage_requires_computed_metric_per_sim(monkeypatch):
    """A sim with stored advertisements but an EMPTY required-metric entry is
    trained-but-unevaluated and must NOT count as covered (2026-09-09: the
    size-32 frozen cell killed mid-eval read as 3/3 covered, the relaunch
    skipped to tabulation and emitted '-')."""
    import numpy as np
    need = gpt.OBJECTIVE_REQUIRED_KEY['frozen_prefix']
    adv = np.ones((4, 2))
    fake = {
        'adv': {0: {'sparse': [adv]}, 1: {'sparse': [adv]}, 2: {'sparse': [adv]}},
        'compare_rets': {0: {'n_advs': 6}, 1: {'n_advs': 6}, 2: {'n_advs': 6}},
        need: {0: {'sparse': 1.0}, 1: {}, 2: {}},
    }
    monkeypatch.setattr(gpt, 'load_metrics', lambda dpsize, obj, tag: (fake, '/fake.pkl'))
    cov = gpt.coverage('32', ['frozen_prefix'], {'*': 3}, 'tag')
    m, p, n, failed = cov['frozen_prefix']
    assert n == 1 and failed == []
    # fully evaluated -> fully covered
    fake[need] = {0: {'sparse': 1.0}, 1: {'sparse': 1.0}, 2: {'sparse': 1.0}}
    assert gpt.coverage('32', ['frozen_prefix'], {'*': 3}, 'tag')['frozen_prefix'][2] == 3
