"""grab_paper_artifacts -- pull the paper's artifact set from the
central storage VM, or fail with a runbook of what still needs to run.

    python evaluations/grab_paper_artifacts.py            # pull everything
    python evaluations/grab_paper_artifacts.py --check    # report only
    python evaluations/grab_paper_artifacts.py --only eods_figures

Run LOCALLY on the Mac. Talks to the storage VM (cluster/vmlib transport,
same key/instance resolution as vmctl -- no hand-rolled ssh). For every
artifact in PAPER_MANIFEST it checks the VM-side source files, pulls the
satisfied ones into figures/paper/ (and figures/paper_table/), and for
anything missing prints WHY plus the ready-to-paste expctl command that
would produce it. Exit code 0 only when the whole set landed.
"""
import argparse
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from cluster import vmlib as V  # noqa: E402

STORAGE_INSTANCE = os.environ.get('SCULPTOR_STORAGE_VM',
                                  'i-0428c395787bc3ca0')
VM_REPO = '/home/ubuntu/sparse_advertisements_code'

# Each artifact: VM-side sources (glob-free, explicit), local destination
# dir, and the runbook command that produces it when absent.
PAPER_MANIFEST = {
    'eods_figures': {
        'desc': 'evaluate_over_deployment_sizes figure set (full ramp)',
        'src_dir': VM_REPO + '/paper_artifacts/eods',
        'src_files': [
            'average_latency_over_deployment_size_normal.pdf',
            'average_latency_over_deployment_size_fail_ingress_mlu.pdf',
            'average_latency_over_deployment_size_fail_site_mlu.pdf',
            'average_congestion_over_deployment_size_fail_ingress_mlu.pdf',
            'average_congestion_over_deployment_size_fail_site_mlu.pdf',
            'average_high_latency_over_deployment_size_fail_ingress_mlu.pdf',
            'average_high_latency_over_deployment_size_fail_site_mlu.pdf',
            'flash_crowd_blowup_before_congestion_over_deployment_size.pdf',
            'diurnal_blowup_before_congestion_over_deployment_size.pdf',
            'percent_traffic_within_10_ms_site_failure_over_deployment_size.pdf',
            'percent_traffic_within_50_ms_normal_over_deployment_size.pdf',
        ],
        'dst': 'figures/paper',
        'runbook': ("python -m cluster.expctl launch {vm} --label eodsredo "
                    "--port 31600 -- /home/ubuntu/venv312/bin/python -u "
                    "evaluations/evaluate_over_deployment_sizes.py --port 31600 "
                    "--cache-fn cache/cluster_runs/20260822_220131-prefixbudget3/"
                    "metrics_by_dpsize.pkl --figures-subdir "
                    "cluster/20260822_220131-prefixbudget3 "
                    "--dpsizes 5,10,15,20,25,actual-32 --nsim 20,20,12,5,4,3 "
                    "--plot"),
    },
    'paper_table': {
        'desc': 'methods x metrics supersection table (nsim=3, healed)',
        'src_dir': VM_REPO + '/paper_artifacts/paper_table',
        'src_files': ['paper_table.csv', 'paper_table.tex',
                      'paper_table_key.csv', 'paper_table_key.tex'],
        'dst': 'figures/paper_table',
        'runbook': ("python -m cluster.expctl launch {vm} --label tableredo "
                    "--port 31601 -- /home/ubuntu/venv312/bin/python -u "
                    "evaluations/generate_paper_table.py --dpsize 32 "
                    "--number_of_deployments 3 --num_training_iter 150 "
                    "--run_id 20260823_130342_papertable32b "
                    "--objectives per_site_cost,max_util,frac_beyond_optimal,"
                    "joint_priority --out figures/cluster/"
                    "20260823_130342-papertable32b/paper_table"),
    },
    'hardness_figures': {
        'desc': 'maxhard ablation grid figures (local store, synced to dash)',
        'src_dir': None,   # local-only artifact
        'local_dir': 'figures/dashboards/ablation_scout',
        'src_files': ['grid_objdim_5panel.png', 'ablation_scout_grid_bars.png',
                      'ablation_scout_difficulty_scatter.png'],
        'dst': 'figures/paper',
        'runbook': ("SCULPTOR_LOG_MEM=0 python run_ablation_grid.py "
                    "--number_measurements_allowed '[5,10]' --deployments 3 "
                    "--num_iters 250 --objectives all --dpsize small && "
                    "python -m dashboard.plot_ablation_scout"),
    },
}


def _vm_ip():
    d = V.resolve(STORAGE_INSTANCE)
    if isinstance(d, str):
        d = V.describe(d)[0]
    if d['state'] != 'running':
        return None, d['state']
    return d['ip'], 'running'


def _remote_stat(ip, paths):
    """One ssh round-trip: mtime+size per path ('MISSING' when absent)."""
    script = ';'.join(
        'stat -c "%Y %s" "{p}" 2>/dev/null || echo MISSING'.format(p=p)
        for p in paths)
    rc, out, _err = V.ssh(ip, script, timeout=60)
    lines = (out or '').strip().split('\n')
    return {p: (None if line.strip() == 'MISSING' else line.strip())
            for p, line in zip(paths, lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='report only; pull nothing')
    ap.add_argument('--only', default=None,
                    help='comma list of manifest keys')
    a = ap.parse_args()
    want = ([k.strip() for k in a.only.split(',')] if a.only
            else list(PAPER_MANIFEST))
    bad = [k for k in want if k not in PAPER_MANIFEST]
    if bad:
        raise SystemExit('unknown artifact(s): {} (have: {})'.format(
            bad, sorted(PAPER_MANIFEST)))

    ip = state = None
    if any(PAPER_MANIFEST[k].get('src_dir') for k in want):
        ip, state = _vm_ip()
        if ip is None:
            print('storage VM {} is {} -- start it first:\n'
                  '  python -m cluster.vmctl start {}'.format(
                      STORAGE_INSTANCE, state, STORAGE_INSTANCE))
            return 2

    missing = {}
    pulled = 0
    for k in want:
        spec = PAPER_MANIFEST[k]
        print('== {} -- {}'.format(k, spec['desc']))
        if spec.get('src_dir'):
            paths = [os.path.join(spec['src_dir'], f)
                     for f in spec['src_files']]
            st = _remote_stat(ip, paths)
            absent = [f for f, p in zip(spec['src_files'], paths)
                      if st[p] is None]
            if absent:
                missing[k] = absent
                print('   MISSING on VM: {}'.format(', '.join(absent)))
                continue
            if a.check:
                print('   ok ({} files on VM)'.format(len(paths)))
                continue
            dst = os.path.join(_REPO, spec['dst'])
            os.makedirs(dst, exist_ok=True)
            src = 'ubuntu@{}:{}/'.format(ip, spec['src_dir'])
            includes = []
            for f in spec['src_files']:
                includes += ['--include', f]
            rc, _o, err = V.rsync(src, dst + '/', ip=ip,
                                  extra=tuple(includes + ['--exclude', '*']))
            if rc != 0:
                print('   rsync FAILED: {}'.format((err or '')[:200]))
                missing[k] = ['<rsync failure>']
                continue
            pulled += len(paths)
            print('   pulled {} file(s) -> {}'.format(len(paths),
                                                      spec['dst']))
        else:
            ld = os.path.join(_REPO, spec['local_dir'])
            absent = [f for f in spec['src_files']
                      if not os.path.exists(os.path.join(ld, f))]
            if absent:
                missing[k] = absent
                print('   MISSING locally: {}'.format(', '.join(absent)))
                continue
            if a.check:
                print('   ok ({} local files)'.format(len(spec['src_files'])))
                continue
            dst = os.path.join(_REPO, spec['dst'])
            os.makedirs(dst, exist_ok=True)
            import shutil
            for f in spec['src_files']:
                shutil.copy(os.path.join(ld, f), os.path.join(dst, f))
            pulled += len(spec['src_files'])
            print('   copied {} file(s) -> {}'.format(
                len(spec['src_files']), spec['dst']))

    if missing:
        print('\nINCOMPLETE -- {} artifact set(s) need runs:'.format(
            len(missing)))
        for k, files in missing.items():
            print('\n  [{}] missing: {}'.format(k, ', '.join(files)))
            print('  produce it with:\n    {}'.format(
                PAPER_MANIFEST[k]['runbook'].format(vm=STORAGE_INSTANCE)))
        return 1
    print('\nALL PAPER ARTIFACTS {} ({} files)'.format(
        'PRESENT' if a.check else 'PULLED', pulled))
    return 0


if __name__ == '__main__':
    sys.exit(main())
