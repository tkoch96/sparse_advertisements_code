"""Print the mesh grid (rung x N) for each seed present in
cache/ablation/mesh_georand: combined score (+collapse tag), probes
spent, and steady avg_lat. Usage: python -m experiments.model_error.mesh_table [seed ...]
"""
import glob
import json
import os
import sys

RUNGS = ['painter', 'no_mc', 'no_memory', 'no_direction', 'full']
NS = [1, 2, 5, 10, 20, 50]
ROOT = 'cache/ablation/mesh_georand'


def comb(r):
    return r['diff_vs_opp'] + 4 * (
        r['fail_popp']['avg_lat_under_failure_abs']
        - r['fail_popp']['opp_avg_lat_under_failure_abs'])


def seed_table(seed):
    cells = {}
    for fn in glob.glob(os.path.join(ROOT, 'N*', 'seed_{}_*.json'.format(seed))):
        r = json.load(open(fn))
        if not r.get('rescored'):
            continue
        cells[(r['rung'], int(fn.split(os.sep)[3][1:]))] = r
    if not cells:
        return False
    print('== SEED {} == combined [probes] (C>10k collapsed, D>100 degraded)'.format(seed))
    print('{:>13} |'.format('rung')
          + ''.join('{:>14}'.format('N=' + str(n)) for n in NS))
    for rung in RUNGS:
        row = []
        for n in NS:
            r = cells.get((rung, n)) or (
                cells.get((rung, 1)) if rung == 'painter' else None)
            if r is None:
                row.append('{:>14}'.format('--'))
                continue
            c = comb(r)
            tag = 'C' if c > 10000 else ('D' if c > 100 else ' ')
            row.append('{:>9.0f}{} [{:>2}]'.format(
                c, tag, r.get('probes_spent', '-')))
            if rung == 'painter':
                row += ['{:>14}'.format('(same)')] * (len(NS) - 1)
                break
        print('{:>13} |'.format(rung) + ''.join(row))
    print('{:>13} |'.format('steady ms')
          + ''.join('{:>14}'.format('') for n in NS))
    for rung in RUNGS:
        row = []
        for n in NS:
            r = cells.get((rung, n)) or (
                cells.get((rung, 1)) if rung == 'painter' else None)
            if r is None:
                row.append('{:>14}'.format('--'))
                continue
            row.append('{:>14.1f}'.format(r['avg_lat']))
            if rung == 'painter':
                row += ['{:>14}'.format('')] * (len(NS) - 1)
                break
        print('{:>13} |'.format(rung) + ''.join(row))
    print()
    return True


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
    for s in seeds:
        seed_table(s)


if __name__ == '__main__':
    main()
