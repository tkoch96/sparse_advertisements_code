"""Render figures/dashboards/depcache/depcache_bench.png from cache/depcache_bench.json
(load time vs n_pops, CSV vs shards). Run by the dash refresh loop."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helpers.figpaths import fig_path  # -> figures/dashboards/<dashboard>/

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    fn = os.path.join(_REPO_ROOT, 'cache', 'depcache_bench.json')
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    if os.path.exists(fn):
        rows = sorted(json.load(open(fn)), key=lambda r: r['n_pops'])
        ns = [r['n_pops'] for r in rows]
        ax.plot(ns, [r['csv_s'] for r in rows], 'o-', color='#c02f4e',
                lw=1.8, label='CSV loop (before)')
        ax.plot(ns, [r['shard_s'] for r in rows], 'o-', color='#2f9e6e',
                lw=1.8, label='per-pop shards (after)')
        for r in rows:
            ax.annotate('{:.0f}x'.format(r['csv_s'] / max(r['shard_s'], 1e-9)),
                        (r['n_pops'], r['shard_s']),
                        textcoords='offset points', xytext=(0, -14),
                        fontsize=8, ha='center', color='#2f9e6e')
    else:
        ax.text(0.5, 0.5, 'bench pending', ha='center')
    ax.set_xlabel('number of pops loaded')
    ax.set_ylabel('latency-data load time (s)')
    ax.set_title('deployment-creation: measurement load, before/after '
                 '(head, in parallel with campaign)', fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = fig_path('depcache_bench.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
