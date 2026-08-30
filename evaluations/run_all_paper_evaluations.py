"""run_all_paper_evaluations -- one intent file drives every paper eval.

    python evaluations/run_all_paper_evaluations.py evaluations/intents/paper_intent.example.json
    python evaluations/run_all_paper_evaluations.py intent.json --only paper_table
    python evaluations/run_all_paper_evaluations.py intent.json --dry-run

SKELETON (Tom 2026-08-30): the intent JSON configures three stages --
(a) evaluate_over_deployment_sizes, (b) evaluate_over_n_prefixes,
(c) generate_paper_table. Each stage builds a command from its intent
block; 'where': 'local' runs it as a local subprocess, 'where': 'vm'
launches it through cluster.expctl on the storage VM (one run per stage,
sequential, stop-on-failure). Stage 'env' merges over global 'env'.

Iteration TODOs (agreed skeleton scope):
  - n_prefixes CLI passthrough once its flags are settled
  - depstore modes per stage (train_only pre-pass, eval_only enforcement)
  - post-run: chain grab_paper_artifacts + push to paper_artifacts/ canon
"""
import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

PY_LOCAL = os.path.expanduser('~/Documents/venv312/bin/python')
PY_VM = '/home/ubuntu/venv312/bin/python'


def _stage_cmd(name, spec, where):
    """Build the argv for one stage from its intent block."""
    py = PY_VM if where == 'vm' else PY_LOCAL
    if name == 'deployment_sizes':
        cmd = [py, '-u', 'evaluations/evaluate_over_deployment_sizes.py',
               '--port', str(spec.get('port', 31600)),
               '--cache-fn', spec['cache_fn'],
               '--figures-subdir', spec['figures_subdir'],
               '--dpsizes', ','.join(str(d) for d in spec['dpsizes']),
               '--nsim', ','.join(str(n) for n in spec['nsim'])]
        if spec.get('plot'):
            cmd.append('--plot')
        return cmd
    if name == 'n_prefixes':
        # SKELETON: verbatim passthrough pending CLI settlement
        cmd = [py, '-u', 'evaluations/evaluate_over_n_prefixes.py',
               '--port', str(spec.get('port', 31602))]
        for k, v in spec.items():
            if k in ('enabled', 'env', 'port', '_comment'):
                continue
            cmd += ['--{}'.format(k),
                    ','.join(str(x) for x in v) if isinstance(v, list)
                    else str(v)]
        return cmd
    if name == 'paper_table':
        return [py, '-u', 'evaluations/generate_paper_table.py',
                '--dpsize', str(spec['dpsize']),
                '--number_of_deployments', str(spec['nsim']),
                '--num_training_iter', str(spec['iters']),
                '--run_id', spec['run_tag'],
                '--objectives', ','.join(spec['objectives']),
                '--out', spec['out']]
    raise SystemExit('unknown stage: {}'.format(name))


def _run_local(name, cmd, env):
    print('[{}] local: {}'.format(name, ' '.join(cmd)))
    full_env = dict(os.environ, **env)
    return subprocess.call(cmd, cwd=_REPO, env=full_env)


def _run_vm(name, cmd, env, vm, run_label):
    argv = [PY_LOCAL, '-m', 'cluster.expctl', 'launch', vm,
            '--label', run_label]
    for k, v in env.items():
        argv += ['--env', '{}={}'.format(k, v)]
    argv += ['--'] + cmd
    print('[{}] vm launch: {}'.format(name, ' '.join(argv)))
    return subprocess.call(argv, cwd=_REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('intent')
    ap.add_argument('--only', default=None,
                    help='comma list of stage names')
    ap.add_argument('--dry-run', action='store_true',
                    help='print every command; run nothing')
    a = ap.parse_args()
    intent = json.load(open(a.intent))
    where = intent.get('where', 'local')
    vm = intent.get('storage_vm')
    genv = intent.get('env') or {}
    stages = intent.get('stages') or {}
    want = ([k.strip() for k in a.only.split(',')] if a.only
            else [k for k, v in stages.items() if v.get('enabled')])

    results = {}
    for name in want:
        spec = stages.get(name)
        if spec is None:
            raise SystemExit('stage {!r} not in intent file'.format(name))
        if spec.get('kind') == 'local_artifact':
            print('[{}] artifact-only stage -- nothing to run '
                  '(see its runbook / grab_paper_artifacts)'.format(name))
            results[name] = 0
            continue
        env = dict(genv, **(spec.get('env') or {}))
        cmd = _stage_cmd(name, spec, where)
        if a.dry_run:
            print('[{}] DRY: env={} cmd={}'.format(
                name, env, ' '.join(cmd)))
            results[name] = 0
            continue
        if where == 'vm':
            rc = _run_vm(name, cmd, env, vm,
                         '{}-{}'.format(intent.get('run_id', 'paper'),
                                        name))
        else:
            rc = _run_local(name, cmd, env)
        results[name] = rc
        if rc != 0:
            print('[{}] FAILED rc={} -- stopping (stop-on-failure)'
                  .format(name, rc))
            break

    print('\n== summary ==')
    for name, rc in results.items():
        print('  {:<20s} {}'.format(name, 'ok' if rc == 0 else
                                    'rc={}'.format(rc)))
    return 1 if any(results.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
