"""Unit tests for the worker-actor death recovery (worker_comms_ray) and the
SCULPTOR_MIN_ITER stopping-condition floor (sparse_advertisements_v3).

These exercise the control flow with stubs/monkeypatch so no real Ray cluster
or deployment is needed -- the real cluster behaviour (spot reclaim -> rebuild)
is validated separately on AWS.

Run: ~/Documents/venv312/bin/python -m pytest tests/test_actor_recovery.py -q
"""
import os
import types

import pytest

# Importing worker_comms_ray triggers a local ray.init (idempotent). We never
# create real actors here; ray.kill / ray.cluster_resources are monkeypatched.
import worker_comms_ray as wc
from worker_comms_ray import Worker_Manager, _ACTOR_DEATH_EXC
from ray.exceptions import RayActorError


def _bare_wm():
    """A Worker_Manager with __init__ bypassed, populated with just the
    attributes the recovery methods touch."""
    wm = object.__new__(Worker_Manager)
    wm.worker_sockets = {}
    wm._target = 4
    wm._start_calls = 0
    # Stub the two collaborators recovery relies on.
    wm._target_n_workers = lambda: wm._target

    def _fake_start_workers(n_workers_override=None):
        wm._start_calls += 1
        # Simulate a freshly-spawned pool of the target size.
        wm.worker_sockets = {i: types.SimpleNamespace(actor=object())
                             for i in range(wm._target)}
    wm.start_workers = _fake_start_workers
    return wm


# --------------------------------------------------------------------------- #
# _with_recovery
# --------------------------------------------------------------------------- #
def test_retry_on_actor_death(monkeypatch):
    monkeypatch.setattr(wc.ray, 'kill', lambda *a, **k: None)
    monkeypatch.setattr(wc.ray, 'cluster_resources', lambda: {'CPU': 64})
    wm = _bare_wm()

    calls = {'n': 0}

    def fn():
        calls['n'] += 1
        if calls['n'] == 1:
            raise RayActorError()
        return 'ok'

    assert wm._with_recovery(fn) == 'ok'
    assert calls['n'] == 2            # failed once, retried once
    assert wm._start_calls == 1       # pool was rebuilt exactly once


def test_no_retry_on_normal_exception(monkeypatch):
    monkeypatch.setattr(wc.ray, 'kill', lambda *a, **k: None)
    monkeypatch.setattr(wc.ray, 'cluster_resources', lambda: {'CPU': 64})
    wm = _bare_wm()

    def fn():
        raise ValueError('not an actor death')

    with pytest.raises(ValueError):
        wm._with_recovery(fn)
    assert wm._start_calls == 0       # no rebuild for a normal exception


def test_second_failure_propagates(monkeypatch):
    monkeypatch.setattr(wc.ray, 'kill', lambda *a, **k: None)
    monkeypatch.setattr(wc.ray, 'cluster_resources', lambda: {'CPU': 64})
    wm = _bare_wm()

    def fn():
        raise RayActorError()

    with pytest.raises(RayActorError):
        wm._with_recovery(fn)
    assert wm._start_calls == 1       # rebuilt once, then re-raised


# --------------------------------------------------------------------------- #
# _rebuild_worker_pool
# --------------------------------------------------------------------------- #
def test_rebuild_waits_for_cpus_then_spawns(monkeypatch):
    monkeypatch.setattr(wc.ray, 'kill', lambda *a, **k: None)
    monkeypatch.setattr(wc.time, 'sleep', lambda s: None)   # don't actually wait
    seq = iter([{'CPU': 0}, {'CPU': 8}, {'CPU': 64}])       # node boots on 3rd poll
    monkeypatch.setattr(wc.ray, 'cluster_resources', lambda: next(seq))
    wm = _bare_wm()
    wm._rebuild_worker_pool()
    assert wm._start_calls == 1
    assert len(wm.worker_sockets) == wm._target


def test_rebuild_times_out(monkeypatch):
    monkeypatch.setattr(wc.ray, 'kill', lambda *a, **k: None)
    monkeypatch.setattr(wc.time, 'sleep', lambda s: None)
    monkeypatch.setattr(wc.ray, 'cluster_resources', lambda: {'CPU': 0})  # never enough
    monkeypatch.setenv('SCULPTOR_RECOVER_NODE_TIMEOUT_S', '0')            # immediate deadline
    wm = _bare_wm()
    with pytest.raises(RuntimeError):
        wm._rebuild_worker_pool()
    assert wm._start_calls == 0


def test_actor_death_tuple_excludes_task_errors():
    # Guard against accidentally widening the catch to all exceptions, which
    # would swallow real bugs (RayTaskError is NOT actor death).
    assert RayActorError in _ACTOR_DEATH_EXC
    assert Exception not in _ACTOR_DEATH_EXC
    assert ValueError not in _ACTOR_DEATH_EXC


# --------------------------------------------------------------------------- #
# SCULPTOR_MIN_ITER floor on the stopping condition.
# Mirrors the predicate built in Sparse_Advertisement_Wrapper.__init__ (the
# `else` branch). Kept in lockstep with that source: convergence cannot fire
# below the floor; the hard max cap is independent of the floor.
# --------------------------------------------------------------------------- #
def _stopping_condition(min_n_iter, max_n_iter, epsilon=0.005, rolling_adv_eps=0.01):
    return lambda el: el[0] > max_n_iter or (
        el[0] >= min_n_iter
        and el[3] < rolling_adv_eps and el[1] < epsilon and abs(el[2]) < epsilon)


def test_floor_blocks_converged_early_stop():
    sc = _stopping_condition(min_n_iter=200, max_n_iter=200)
    # Fully converged (all deltas ~0) but below the floor -> must NOT stop.
    converged = [80, 0.0, 0.0, 0.0]
    assert sc(converged) is False
    # At the floor, the same converged state -> stop.
    converged_at_floor = [200, 0.0, 0.0, 0.0]
    assert sc(converged_at_floor) is True


def test_floor_zero_is_original_behaviour():
    sc = _stopping_condition(min_n_iter=0, max_n_iter=200)
    # Converged at iter 5 with no floor -> stop immediately (original).
    assert sc([5, 0.0, 0.0, 0.0]) is True
    # Not converged -> keep going.
    assert sc([5, 1.0, 1.0, 1.0]) is False


def test_hard_max_cap_independent_of_floor():
    sc = _stopping_condition(min_n_iter=500, max_n_iter=200)
    # Past the hard cap, even below the (higher) floor and not converged -> stop.
    assert sc([201, 9.9, 9.9, 9.9]) is True
