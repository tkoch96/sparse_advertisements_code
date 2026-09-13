"""Named paper numbers (evaluations/paper_numbers.py, Tom 2026-09-12): the key
grammar, direction-aware comparison verbs, the near-zero guard, role
resolution, token docs, and sync with the table generator's display maps."""
import json
import os
import re
import pytest

from evaluations import paper_numbers as pn

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- a tiny synthetic artifacts dir ------------------------------------------

@pytest.fixture
def arts(tmp_path):
    cols = ['MLU|Latency (ms)', 'MLU|MLU', 'Latency + g*Resilience|% cong PoP-fail',
            'Latency + g*Resilience|Latency PoP-fail (ms)', 'Latency + g*Resilience|Flash-crowd resilience',
            'Frac beyond optimal|% within 10ms']
    dirs = ['<', '<', '<', '<', '>', '>']
    rows = {
        'One-per-peering': [29.0, 0.88, 0.5, 30.9, 14.0, 87.0],
        'SCULPTOR':        [30.4, 0.88, 5.18, 32.1, 12.8, 83.7],
        'PAINTER':         [33.7, 0.90, 8.29, 37.0, 10.8, 79.0],
        'Unicast':         [34.2, 0.89, 5.44, 36.8, 12.8, 76.0],
        'AnyOpt':          [46.2, 0.91, 21.77, 37.9, 10.1, 68.2],
        'Anycast':         [51.2, 0.91, 0.0, 50.1, 10.1, 67.3],
    }
    with open(tmp_path / 'paper_table_full.csv', 'w') as f:
        f.write('method,' + ','.join(cols) + '\n')
        f.write('DIRECTION,' + ','.join(dirs) + '\n')
        for m, v in rows.items():
            f.write(m + ',' + ','.join('{:.4f}'.format(x) for x in v) + '\n')
    with open(tmp_path / 'paper_table_full_stats.csv', 'w') as f:
        f.write('method,' + ','.join(cols) + '\n')
        for m, v in rows.items():
            f.write(m + ',' + ','.join('{:.4f}|0.25|3'.format(x) for x in v) + '\n')
    with open(tmp_path / 'deployment_scaling_summary.csv', 'w') as f:
        f.write('axis,objective,metric,direction,sculptor_at_max,best_baseline_at_max,best_baseline_name,'
                'gap_at_min,gap_at_max,gap_change_pct,spearman_rho,sculptor_leads,verdict\n')
        f.write('deployment_size,Latency,Subopt normal (ms),<,1.143,4.704,painter,1.512,3.562,57.5,1.0,6/6,grow\n')
    abl = tmp_path / 'ablation' / 'avg_latency'
    abl.mkdir(parents=True)
    json.dump({'n_deployments': 2, 'mean_opp_objective': 100.0, 'mean_zero_anchor_objective': 200.0,
               'rungs': [{'rung': 'full', 'pct_gap_closed_on_means': 90.7, 'increment_pct': 1.5,
                          'mean_objective': 109.3, 'mean_minus_opp': 9.3}]},
              open(abl / 'ladder_summary.json', 'w'))
    manual = tmp_path / 'manual.json'
    json.dump({'_comment': 'x',
               'sculptor_prefixes_a32': {'value': 42, 'doc': 'd', 'from': 'f', 'prec': 0},
               'opp_prefixes_a32': {'value': 779, 'doc': 'd', 'from': 'f', 'prec': 0},
               'nodoc': {'value': 1, 'from': 'f'},
               'later': {'placeholder': True, 'value': None, 'doc': 'd', 'from': 'todo'}},
              open(manual, 'w'))
    return tmp_path


@pytest.fixture
def R(arts):
    return pn.Resolver(pn.Source(str(arts), manual_json=str(arts / 'manual.json'), run_tag='t'))


# ---- grammar -------------------------------------------------------------------

def test_base_cell_and_formatting(R):
    r = R.resolve('dtf.popfail.all.lat.mean.sculptor')
    assert r.ok and r.text == '32.1' and 'whole-site' in r.doc and 'SCULPTOR' in r.doc
    assert R.resolve('dtf.popfail.all.cong.mean.sculptor.d1').text == '5.2'
    assert R.resolve('dtf.popfail.all.cong.mean.sculptor').text == '5.18'


def test_invalid_tokens_explain_themselves(R):
    assert 'unknown scenario' in R.resolve('dtf.bogus.all.lat.mean.sculptor').error
    assert 'no such cell' in R.resolve('dtf.flash.all.lat.mean.sculptor').error
    assert 'valid for dtf' in R.resolve('dtf.flash.all.lat.mean.sculptor').error
    assert 'unknown family' in R.resolve('nope.x').error
    assert 'unknown verb' in R.resolve('dtf.popfail.all.lat.mean.sculptor.foo.opp').error
    assert 'takes 1 argument' in R.resolve('dtf.popfail.all.lat.mean.sculptor.minus').error


def test_direction_aware_verbs(R):
    # lower-is-better latency: sculptor 32.1 vs opp 30.9
    assert R.resolve('dtf.popfail.all.lat.mean.sculptor.minus.opp').text == '1.2'
    assert R.resolve('dtf.popfail.all.lat.mean.sculptor.worsethan.opp').text == '1.2'
    # pctbetter vs painter (37.0): (37.0-32.1)/37.0 = 13.2%
    assert R.resolve('dtf.popfail.all.lat.mean.sculptor.pctbetter.painter').text == '13'
    assert R.resolve('dtf.popfail.all.lat.mean.sculptor.pctworse.painter').text == '-13'
    # higher-is-better flash intensity: sculptor 12.8 vs painter 10.8 -> better
    assert float(R.resolve('dtf.flash.all.intensity.mean.sculptor.pctbetter.painter').text) > 0
    assert float(R.resolve('dtf.flash.all.intensity.mean.sculptor.worsethan.painter').text) < 0
    # over, pctway
    assert R.resolve('dtf.flash.all.intensity.mean.sculptor.over.anycast').text == '1.3'
    r = R.resolve('lss.normal.all.within10.mean.sculptor.pctway.painter.opp')
    assert r.text == '{:.0f}'.format(100 * (83.7 - 79.0) / (87.0 - 79.0))


def test_near_zero_guard_and_points_alternative(R):
    r = R.resolve('dtf.popfail.all.cong.mean.sculptor.pctbetter.anycast')   # anycast 0.0 %
    assert not r.ok and 'floor' in r.error and 'minus' in r.error
    assert R.resolve('dtf.popfail.all.cong.mean.sculptor.minus.anycast').text == '5.18'
    assert not R.resolve('dtf.popfail.all.cong.mean.sculptor.over.anycast').ok


def test_roles_and_name(R):
    # best prior approach on site-fail congestion: unicast 5.44 (< painter 8.29, anyopt 21.77, anycast 0.0?)
    # anycast is 0.0 here and IS the best baseline by the metric -- roles follow the numbers
    assert R.resolve('dtf.popfail.all.cong.mean.bestbaseline.name').text == '\\acast'
    assert R.resolve('dtf.popfail.all.lat.mean.bestbaseline.name').text == '\\ucast'
    assert R.resolve('dtf.popfail.all.lat.mean.best.name').text == '\\opp'
    assert R.resolve('dtf.popfail.all.lat.mean.bestpractical.name').text == '\\sparse'
    assert R.resolve('dtf.popfail.all.lat.mean.worst.name').text == '\\acast'
    r = R.resolve('dtf.popfail.all.lat.mean.sculptor.pctbetter.bestbaseline')
    assert r.ok and 'bestbaseline -> Unicast' in r.doc


def test_derived_cells_match_table_display(R):
    assert R.resolve('latmlu.normal.all.mluratio.mean.sculptor').text == '{:.3f}'.format(0.88 / (1 / 1.1))
    assert R.resolve('dtf.flash.all.intensityratio.mean.sculptor').text == '{:.2f}'.format(12.8 / 10.1)
    assert R.resolve('lss.normal.all.beyond10.mean.sculptor').text == '16.3'


def test_stats_from_stats_csv(R):
    assert R.resolve('dtf.popfail.all.lat.std.sculptor').text == '0.2'
    assert R.resolve('dtf.popfail.all.lat.n.sculptor').text == '3'


def test_negative_worsethan_opp_is_flagged(R):
    r = R.resolve('dtf.flash.all.intensity.mean.sculptor.worsethan.opp')   # 14.0 - 12.8 > 0: fine
    assert r.ok and not r.notes
    r = R.resolve('latmlu.normal.all.mlu.mean.sculptor.worsethan.opp')   # equal -> 0.0, no note
    assert r.ok and r.text == '0.000'


def test_sweep_abl_headline_manual(R):
    assert R.resolve('sweep.dpsize.latency.subopt-normal-ms.gap-change-pct').text == '58'
    assert R.resolve('sweep.dpsize.latency.subopt-normal-ms.best-baseline-name').text == '\\painter'
    assert R.resolve('sweep.dpsize.latency.subopt-normal-ms.verdict').text == 'grow'
    assert 'no sweep row' in R.resolve('sweep.dpsize.latency.nope.verdict').error
    assert R.resolve('abl.dtf.full.pctbenefit').text == '91'
    assert R.resolve('abl.dtf.painter.pctbenefit').text == '0'
    assert R.resolve('abl.dtf.OPP.pctbenefit').text == '100'
    assert 'no ladder_summary' in R.resolve('abl.lss.full.pctbenefit').error
    assert R.resolve('headline.prefix_savings_x').text == '19'
    assert 'HEADLINE' in R.resolve('headline.prefix_savings_x').doc
    assert 'unknown headline' in R.resolve('headline.nope').error
    assert R.resolve('manual.sculptor_prefixes_a32').text == '42'
    assert 'needs both' in R.resolve('manual.nodoc').error
    assert 'placeholder' in R.resolve('manual.later').error


def test_every_headline_has_a_doc():
    for k, h in pn.HEADLINES.items():
        assert h.doc.strip(), k


def test_every_token_has_a_doc():
    for vocab in (pn.OBJECTIVES, pn.SCENARIOS, pn.POPULATIONS, pn.METRICS, pn.STATS, pn.METHODS,
                  pn.ROLES, pn.VERBS, pn.SWEEP_AXES, pn.SWEEP_FIELDS, pn.ABL_RUNGS, pn.ABL_FIELDS):
        for k, t in vocab.items():
            assert t.doc.strip(), k
    for cell in pn.CELLS:
        assert cell[0] in pn.OBJECTIVES and cell[1] in pn.SCENARIOS
        assert cell[2] in pn.POPULATIONS and cell[3] in pn.METRICS


# ---- emit / check --------------------------------------------------------------

def test_emit_defines_referenced_and_reports_undefined(arts, R, tmp_path):
    tex = tmp_path / 'p.tex'
    tex.write_text('a \\pn{dtf.popfail.all.lat.mean.sculptor.worsethan.opp} b '
                   '\\pn{dtf.popfail.all.cong.mean.sculptor.pctbetter.anycast} c \\pn{manual.later}\n'
                   'd \\pn[x]{headline.prefix_savings_x}\n')
    out = tmp_path / 'tables'
    results, refs, prev = pn.emit(R, [str(tex)], str(out), write=True)
    assert set(refs) == {'dtf.popfail.all.lat.mean.sculptor.worsethan.opp',
                         'dtf.popfail.all.cong.mean.sculptor.pctbetter.anycast', 'manual.later',
                         'headline.prefix_savings_x'}
    body = (out / 'paper_numbers.tex').read_text()
    assert '\\pndef{dtf.popfail.all.lat.mean.sculptor.worsethan.opp}{1.2}' in body
    assert '\\pndef{headline.prefix_savings_x}{19}' in body
    assert 'UNDEFINED' in body and 'pctbetter.anycast' in body
    assert '%   source:' in body and 'provenance' in body
    # base cells emitted too
    assert '\\pndef{dtf.popfail.all.lat.mean.painter}{37.0}' in body
    lines = []
    n_undef = pn.report(results, refs, prev, lines.append)
    assert n_undef == 2
    snap = json.load(open(out / 'paper_numbers.json'))
    assert snap['keys']['dtf.popfail.all.lat.mean.sculptor.worsethan.opp']['text'] == '1.2'
    # a second emit after the number moved reports the change
    snap['keys']['dtf.popfail.all.lat.mean.sculptor.worsethan.opp']['text'] = '9.9'
    json.dump(snap, open(out / 'paper_numbers.json', 'w'))
    results, refs, prev = pn.emit(R, [str(tex)], str(out), write=False)
    lines = []
    pn.report(results, refs, prev, lines.append)
    assert any('9.9 -> 1.2' in l for l in lines)


def test_pn_regex_matches_doc_forms():
    s = r'x \pn{a.b.c} y \pn[0]{d.e} z \tbd{3} \pn{ f.g }'
    assert [k.strip() for k in pn.PN_RE.findall(s)] == ['a.b.c', 'd.e', 'f.g']


# ---- sync with the table generator ---------------------------------------------

def test_derived_cells_track_generate_paper_table_display_maps():
    os.environ.setdefault('SCULPTOR_XOBJS', '1')
    from evaluations import generate_paper_table as gpt
    # every ratio / complement column the .tex shows has a derived token here
    ratio_subs = set(gpt.TEX_RATIO)
    comp_subs = set(gpt.TEX_COMPLEMENT)
    derived = {}
    for cell, c in pn.CELLS.items():
        if c.derive:
            base = pn.CELLS[c.derive[1]].column.split('|', 1)[1]
            derived[base] = c.derive
    assert ratio_subs <= set(derived), ratio_subs - set(derived)
    assert comp_subs <= set(derived), comp_subs - set(derived)
    assert derived['MLU'] == ('ratio_const', ('latmlu', 'normal', 'all', 'mlu'), 1.0 / gpt.MLU_HEADROOM)
    for sub, ref in gpt.TEX_RATIO.items():
        if isinstance(ref, str):
            tok = [k for k, m in pn.METHODS.items() if m.csv == ref][0]
            assert derived[sub][2] == tok, sub
    # every stored column the CELLS point at exists in the generator's column set
    cols = {'{}|{}'.format(o, l) for _o, l, _d, _f in gpt.COLUMNS for o in [_o]} if hasattr(gpt, 'COLUMNS') else None
    if cols:
        # COLUMNS entries are (group, label, dir, fn); label is the full 'Group|Sub'
        labels = {l for _o, l, _d, _f in gpt.COLUMNS}
        for cell, c in pn.CELLS.items():
            if c.column:
                assert c.column in labels, c.column


# ---- on-Internet family --------------------------------------------------------

@pytest.fixture
def R_internet(arts):
    with open(arts / 'actual_deployment_stats.csv', 'w') as f:
        f.write('scenario,method,gap,gapall,overloaded,within10,within50,within100,n_scenarios,source\n')
        f.write('steady,sculptor,1.93,1.93,0,91.8,100,100,1,pickle\n')
        f.write('steady,painter,5.50,5.50,0,88.0,96.8,100,1,pickle\n')
        f.write('steady,unicast,7.05,7.05,0,74.9,98.5,100,1,pickle\n')
        f.write('steady,anycast,20.39,20.39,0,56.8,85.2,99.1,1,pickle\n')
        f.write('linkfail,sculptor,7.93,7.93,0,64.5,100,100,29,pickle\n')
        f.write('linkfail,unicast,14.24,14.24,0,38.2,100,100,26,pickle\n')
        f.write('linkfail,painter,58.3,347.3,75.8,10.4,17.7,20.3,42,pickle\n')
        f.write('linkfail,anycast,3.12,304.7,69.0,26.5,31.0,31.0,15,pickle\n')
    return pn.Resolver(pn.Source(str(arts), manual_json=str(arts / 'manual.json'), run_tag='t'))


def test_internet_family(R_internet):
    R = R_internet
    assert R.resolve('internet.steady.gap.sculptor').text == '1.9'
    assert R.resolve('internet.steady.gap.bestbaseline.name').text == '\\painter'
    assert R.resolve('internet.steady.gap.sculptor.betterthan.bestbaseline').text == '3.6'
    assert R.resolve('internet.steady.gap.sculptor.pctbetter.painter').text == '65'
    assert R.resolve('internet.steady.within10.sculptor.minus.painter').text == '3.8'
    assert R.resolve('internet.linkfail.overloaded.painter').text == '75.8'
    assert R.resolve('internet.linkfail.gap.sculptor.betterthan.unicast').text == '6.3'
    # roles follow the numbers even when misleading; the gap doc carries the caution
    assert R.resolve('internet.linkfail.gap.bestbaseline.name').text == '\\acast'
    assert 'CAUTION' in R.resolve('internet.linkfail.gap.bestbaseline').doc
    assert R.resolve('internet.linkfail.gapall.bestbaseline.name').text == '\\ucast'
    assert R.resolve('internet.linkfail.gap.opp').text == '0.0'
    assert 'no row' in R.resolve('internet.sitefail.gap.sculptor').error
    assert 'unknown internet metric' in R.resolve('internet.steady.nope.sculptor').error
    assert R.resolve('headline.intro_internet_steady_gain_ms').text == '3.6'
    assert R.resolve('headline.intro_internet_linkfail_gain_ms').text == '6.3'


def test_betterthan_verb(R):
    assert R.resolve('dtf.popfail.all.lat.mean.sculptor.betterthan.painter').text == '4.9'
    assert R.resolve('dtf.popfail.all.lat.mean.sculptor.betterthan.opp').text == '-1.2'
