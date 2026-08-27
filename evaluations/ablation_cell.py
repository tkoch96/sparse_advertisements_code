"""Reusable per-cell (per-sim) eval wrapper for ablation experiments.

One cell = one subprocess speaking run_fork_ladder's CLI + result-JSON
convention, run inside a slot workspace. The wrapper guarantees the disk
footprint stays flat no matter how the cell ends (success, nonzero rc,
timeout, exception): it harvests the ~1MB that matters (result JSON is
written by the cell itself into the store; convergence/model-error PDFs
are copied to the artifacts dir under a canonical name) and then deletes
the fluff -- the run dir (state pickles included) and the slot's private
ray tmp sessions.

Any ablation driver (grid queue, one-off ladders, future fleet work)
should run cells through run_cell() instead of hand-rolling
subprocess + cleanup; that is the code-reuse contract (Tom 2026-08-27).
"""
import glob
import os
import re
import shutil
import subprocess


def harvest_figs(ws, figs_dir, label, dpsize):
    """Copy this cell's convergence/model-error PDFs from ws/runs/* into
    figs_dir under '<label>_<rundir-suffix>.pdf'. Each slot runs ONE cell
    at a time, so any dir in this slot's runs/ belongs to the cell that
    just finished (timestamped actual-N run dirs escape labeled globs --
    found 2026-08-18)."""
    if not figs_dir:
        return
    os.makedirs(figs_dir, exist_ok=True)
    for d in glob.glob(os.path.join(ws, 'runs', '*')):
        suffix = os.path.basename(d).replace(
            'ablation-{}-'.format(dpsize), '', 1)
        src = os.path.join(d, 'convergence_over_iterations.pdf')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(
                figs_dir, '{}_{}.pdf'.format(label, suffix)))
        me = os.path.join(d, 'model_error_over_iterations.pdf')
        if os.path.exists(me):
            shutil.copy(me, os.path.join(
                figs_dir, 'ME_{}_{}.pdf'.format(label, suffix)))
        # final state pickle rides along (Tom 2026-08-27): highest-N
        # state-*.pkl is the hot-start / post-mortem artifact -- ~1MB,
        # part of the keep set, everything else in the run dir is fluff.
        states = []
        for f in glob.glob(os.path.join(d, 'state-*.pkl')):
            m = re.search(r'state-(\d+)\.pkl$', f)
            if m:
                states.append((int(m.group(1)), f))
        if states:
            n, f = max(states)
            shutil.copy(f, os.path.join(
                figs_dir, '{}_{}_state-{}.pkl'.format(label, suffix, n)))


def clean_cell(ws, ray_tmp=None):
    """Delete the cell's disk fluff: every dir under ws/runs (run state
    pickles ride inside) and the slot's ray tmp sessions. Runs on EVERY
    exit path -- the old inline harvest only fired on rc==0, so failed
    cells leaked their run dirs, and ray_q_S* grew one dead session per
    cell until the box hit ENOSPC (2026-08-27)."""
    for d in glob.glob(os.path.join(ws, 'runs', '*')):
        shutil.rmtree(d, ignore_errors=True)
    if ray_tmp and os.path.isdir(ray_tmp):
        # the cell's driver has exited, so its raylet tree is dead; only
        # session dirs live here and the next cell creates a fresh one.
        for d in glob.glob(os.path.join(ray_tmp, 'session_*')):
            if os.path.islink(d):
                try:
                    os.remove(d)
                except OSError:
                    pass
            else:
                shutil.rmtree(d, ignore_errors=True)


def run_cell(cmd, ws, env, log_path, timeout_s,
             figs_dir=None, label=None, dpsize=None):
    """Run one ablation cell to completion and ALWAYS leave the slot
    clean. Returns the subprocess rc (-99 on timeout, matching the queue
    convention). Figures are harvested before cleanup regardless of rc so
    a failed cell still leaves its convergence evidence."""
    try:
        with open(log_path, 'w') as lf:
            try:
                rc = subprocess.call(cmd, cwd=ws, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT,
                                     timeout=timeout_s)
            except subprocess.TimeoutExpired:
                rc = -99
    finally:
        try:
            harvest_figs(ws, figs_dir, label, dpsize)
        finally:
            clean_cell(ws, env.get('RAY_TMPDIR'))
    return rc
