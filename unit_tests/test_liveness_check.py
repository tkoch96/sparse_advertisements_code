"""Unit tests for ~/.sculptor_cluster_alert/liveness_check.py.

The script lives outside the repo (it's installed under ~/.sculptor_cluster_alert/
so cron can run it independent of which branch is checked out), so we load it
by absolute path via importlib. Tests run hermetically: no real SSH, no real
AWS CLI, no real SMS sends -- all the side-effecting seams are monkey-patched.

Coverage focus is the recent (session 10) sweep_pid_dead refinement: the
script now distinguishes "completed normally" / "in transition" / "just
exited" from "real crash" before firing CRIT alerts.

Run with:  pytest tests/test_liveness_check.py -v
"""
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

SCRIPT_PATH = os.path.expanduser('~/.sculptor_cluster_alert/liveness_check.py')


def _load_liveness():
    if not os.path.exists(SCRIPT_PATH):
        pytest.skip("liveness_check.py not installed at {}".format(SCRIPT_PATH))
    spec = importlib.util.spec_from_file_location('liveness_check', SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['liveness_check'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #
@pytest.fixture
def lvc(tmp_path, monkeypatch):
    """Fresh liveness_check module with all file paths redirected into a
    per-test tempdir. Also stubs out the SMS / ntfy / osascript side
    effects so nothing leaves the test process.
    """
    mod = _load_liveness()
    monkeypatch.setattr(mod, 'ALERT_DIR', str(tmp_path))
    monkeypatch.setattr(mod, 'CONFIG_PATH', str(tmp_path / 'active_cluster.json'))
    monkeypatch.setattr(mod, 'STATE_PATH', str(tmp_path / 'state.json'))
    monkeypatch.setattr(mod, 'LOG_PATH', str(tmp_path / 'alert.log'))
    # Record alerts in-memory instead of sending. Tag, severity, summary
    # are extracted from the notify() call.
    mod._test_alerts = []
    orig_notify = mod.notify
    def fake_notify(tag, severity, summary, detail=''):
        # Respect dedup logic by going through the original (which checks
        # mod.STATE_PATH and writes); we just intercept the OUTBOUND
        # channels.
        mod._test_alerts.append({'tag': tag, 'severity': severity,
                                  'summary': summary, 'detail': detail})
        return orig_notify(tag, severity, summary, detail)
    monkeypatch.setattr(mod, 'notify', fake_notify)
    # SMS / ntfy / osascript stubs (no real sends).
    monkeypatch.setattr(mod, '_send_sms', lambda body, subject=None: None)
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: None)
    # Suppress osascript invocations -- they're inside _run() calls, so the
    # per-test fake_run handles that.
    return mod


def _write_config(lvc, tmp_path, **overrides):
    """Write a valid active_cluster.json with sensible defaults; pass
    overrides via kwargs (e.g. active=False)."""
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    cfg = {
        'active': True,
        'last_updated': now_iso,
        'head': {
            'instance_id': 'i-test',
            'public_ip': '10.0.0.1',
            'ssh_key': '~/.ssh/test.pem',
            'ssh_user': 'ubuntu',
        },
        'sweep': {
            'pid_file': '/tmp/test.pid',
            'log_file': '/tmp/test.log',
            'min_log_growth_per_check_lines': 5,
        },
        'alert': {'ntfy_topic': 'test-topic', 'macos_notification': False},
    }
    for k, v in overrides.items():
        if k == 'last_updated':
            cfg['last_updated'] = v
        elif k == 'active':
            cfg['active'] = v
        else:
            cfg[k] = v
    with open(lvc.CONFIG_PATH, 'w') as f:
        json.dump(cfg, f)
    return cfg


def _make_fake_run(aws_state='running', ssh_fields=None):
    """Build a fake _run() that dispatches by command content."""
    ssh_fields = ssh_fields or {}
    def fake_run(cmd, timeout=20):
        if 'describe-instances' in cmd:
            return (0, aws_state + '\n', '')
        if cmd.startswith('/usr/bin/ssh') or '-o BatchMode=yes' in cmd:
            out = '\n'.join('{}={}'.format(k, v) for k, v in ssh_fields.items())
            return (0, out + '\n', '')
        if 'osascript' in cmd:
            return (0, '', '')
        return (0, '', '')
    return fake_run


# ---------------------------------------------------------------------- #
# AWS state checks                                                        #
# ---------------------------------------------------------------------- #
class TestAwsState:
    def test_active_and_running_no_alert(self, lvc, tmp_path, monkeypatch):
        cfg = _write_config(lvc, tmp_path)
        monkeypatch.setattr(lvc, '_run', _make_fake_run(aws_state='running'))
        lvc.check_aws_state(cfg)
        tags = [a['tag'] for a in lvc._test_alerts]
        assert 'vm_crashed' not in tags
        assert 'orphan_vm' not in tags

    def test_active_but_stopped_fires_vm_crashed(self, lvc, tmp_path, monkeypatch):
        cfg = _write_config(lvc, tmp_path, active=True)
        monkeypatch.setattr(lvc, '_run', _make_fake_run(aws_state='stopped'))
        lvc.check_aws_state(cfg)
        assert any(a['tag'] == 'vm_crashed' and a['severity'] == 'CRIT'
                   for a in lvc._test_alerts)

    def test_inactive_but_running_fires_orphan(self, lvc, tmp_path, monkeypatch):
        cfg = _write_config(lvc, tmp_path, active=False)
        monkeypatch.setattr(lvc, '_run', _make_fake_run(aws_state='running'))
        lvc.check_aws_state(cfg)
        assert any(a['tag'] == 'orphan_vm' and a['severity'] == 'WARN'
                   for a in lvc._test_alerts)

    def test_inactive_and_stopped_no_alert(self, lvc, tmp_path, monkeypatch):
        cfg = _write_config(lvc, tmp_path, active=False)
        monkeypatch.setattr(lvc, '_run', _make_fake_run(aws_state='stopped'))
        lvc.check_aws_state(cfg)
        tags = [a['tag'] for a in lvc._test_alerts]
        assert 'vm_crashed' not in tags
        assert 'orphan_vm' not in tags


# ---------------------------------------------------------------------- #
# Staleness                                                               #
# ---------------------------------------------------------------------- #
class TestStaleness:
    def test_fresh_config_no_alert(self, lvc, tmp_path):
        _write_config(lvc, tmp_path)  # last_updated=now by default
        lvc.check_staleness(json.load(open(lvc.CONFIG_PATH)))
        assert not any(a['tag'] == 'stale_config' for a in lvc._test_alerts)

    def test_48h_old_config_fires_stale(self, lvc, tmp_path):
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).replace(microsecond=0)
        _write_config(lvc, tmp_path,
                       last_updated=old.isoformat().replace('+00:00', 'Z'))
        lvc.check_staleness(json.load(open(lvc.CONFIG_PATH)))
        assert any(a['tag'] == 'stale_config' and a['severity'] == 'WARN'
                   for a in lvc._test_alerts)

    def test_inactive_config_never_stale(self, lvc, tmp_path):
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).replace(microsecond=0)
        _write_config(lvc, tmp_path, active=False,
                       last_updated=old.isoformat().replace('+00:00', 'Z'))
        lvc.check_staleness(json.load(open(lvc.CONFIG_PATH)))
        assert not any(a['tag'] == 'stale_config' for a in lvc._test_alerts)


# ---------------------------------------------------------------------- #
# Sweep PID-death discrimination -- the meat of the session-10 fix      #
# ---------------------------------------------------------------------- #
class TestSweepPidDead:
    def _setup(self, lvc, tmp_path, monkeypatch, ssh_fields):
        cfg = _write_config(lvc, tmp_path)
        monkeypatch.setattr(lvc, '_run',
                            _make_fake_run(aws_state='running',
                                           ssh_fields=ssh_fields))
        return cfg

    def test_alive_and_growing_no_alert(self, lvc, tmp_path, monkeypatch):
        cfg = self._setup(lvc, tmp_path, monkeypatch, {
            'pid_alive': 1, 'pid_age_s': 3000,
            'log_lines': 500, 'log_age_s': 5, 'log_completed': 0,
        })
        lvc.check_ssh_and_sweep(cfg, 'running')
        # Second call to simulate growth
        cfg = self._setup(lvc, tmp_path, monkeypatch, {
            'pid_alive': 1, 'pid_age_s': 3060,
            'log_lines': 600, 'log_age_s': 5, 'log_completed': 0,
        })
        lvc.check_ssh_and_sweep(cfg, 'running')
        tags = [a['tag'] for a in lvc._test_alerts]
        assert 'sweep_pid_dead' not in tags
        assert 'sweep_stalled' not in tags
        assert 'sweep_completed' not in tags

    def test_pid_dead_with_completed_marker_fires_INFO(self, lvc, tmp_path, monkeypatch):
        """The 22:20 UTC false-positive scenario: sweep finished normally
        with [sweep] ALL DONE in the log. Should fire sweep_completed INFO,
        NOT sweep_pid_dead CRIT.
        """
        cfg = self._setup(lvc, tmp_path, monkeypatch, {
            'pid_alive': 0, 'pid_age_s': 600,
            'log_lines': 5000, 'log_age_s': 120, 'log_completed': 1,
        })
        lvc.check_ssh_and_sweep(cfg, 'running')
        completed = [a for a in lvc._test_alerts if a['tag'] == 'sweep_completed']
        dead = [a for a in lvc._test_alerts if a['tag'] == 'sweep_pid_dead']
        assert len(completed) == 1
        assert completed[0]['severity'] == 'INFO'
        assert len(dead) == 0, 'Should NOT fire CRIT when ALL DONE marker present'

    def test_pid_dead_during_transition_no_alert(self, lvc, tmp_path, monkeypatch):
        """The 21:40 UTC false-positive scenario: relaunch in progress,
        pidfile was just rewritten by my launch script. pid_age_s < 180.
        """
        cfg = self._setup(lvc, tmp_path, monkeypatch, {
            'pid_alive': 0, 'pid_age_s': 30,           # ← very recent pidfile
            'log_lines': 100, 'log_age_s': 5, 'log_completed': 0,
        })
        lvc.check_ssh_and_sweep(cfg, 'running')
        tags = [a['tag'] for a in lvc._test_alerts]
        assert 'sweep_pid_dead' not in tags, 'transition window should suppress alert'

    def test_pid_dead_just_exited_no_alert(self, lvc, tmp_path, monkeypatch):
        """Process just exited (log was written < 60s ago); skip this
        tick and reassess next time.
        """
        cfg = self._setup(lvc, tmp_path, monkeypatch, {
            'pid_alive': 0, 'pid_age_s': 600,
            'log_lines': 5000, 'log_age_s': 10,         # ← log just written
            'log_completed': 0,
        })
        lvc.check_ssh_and_sweep(cfg, 'running')
        tags = [a['tag'] for a in lvc._test_alerts]
        assert 'sweep_pid_dead' not in tags

    def test_pid_dead_real_crash_fires_CRIT(self, lvc, tmp_path, monkeypatch):
        """None of the skip conditions apply: pidfile old, log old, no
        completion marker. This is a real silent crash; fire CRIT.
        """
        cfg = self._setup(lvc, tmp_path, monkeypatch, {
            'pid_alive': 0, 'pid_age_s': 600,         # not transition
            'log_lines': 5000, 'log_age_s': 600,      # not just-exited
            'log_completed': 0,                       # not completed
        })
        lvc.check_ssh_and_sweep(cfg, 'running')
        dead = [a for a in lvc._test_alerts if a['tag'] == 'sweep_pid_dead']
        assert len(dead) == 1
        assert dead[0]['severity'] == 'CRIT'
        # Detail should include pid_age_s + log_age_s for triage
        assert 'pid_age_s=600' in dead[0]['detail']
        assert 'log_age_s=600' in dead[0]['detail']


# ---------------------------------------------------------------------- #
# Stall detection -- 3 consecutive stalls before alerting                 #
# ---------------------------------------------------------------------- #
class TestStallDetection:
    def _step(self, lvc, tmp_path, monkeypatch, log_lines):
        cfg = _write_config(lvc, tmp_path)
        monkeypatch.setattr(lvc, '_run',
                            _make_fake_run(aws_state='running',
                                           ssh_fields={'pid_alive': 1, 'pid_age_s': 3000,
                                                        'log_lines': log_lines, 'log_age_s': 60,
                                                        'log_completed': 0}))
        lvc.check_ssh_and_sweep(cfg, 'running')

    def test_one_stall_no_alert(self, lvc, tmp_path, monkeypatch):
        # First call sets baseline (no prev), no growth check.
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)
        # Second call: same log_lines, +0 < 5 = stall #1
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)
        assert not any(a['tag'] == 'sweep_stalled' for a in lvc._test_alerts)

    def test_two_stalls_no_alert(self, lvc, tmp_path, monkeypatch):
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # baseline
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 1
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 2
        assert not any(a['tag'] == 'sweep_stalled' for a in lvc._test_alerts)

    def test_three_stalls_fires_WARN(self, lvc, tmp_path, monkeypatch):
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # baseline
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 1
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 2
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 3 -> alert
        stalls = [a for a in lvc._test_alerts if a['tag'] == 'sweep_stalled']
        assert len(stalls) == 1
        assert stalls[0]['severity'] == 'WARN'

    def test_growth_resets_stall_counter(self, lvc, tmp_path, monkeypatch):
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # baseline
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 1
        self._step(lvc, tmp_path, monkeypatch, log_lines=100)  # stall 2
        self._step(lvc, tmp_path, monkeypatch, log_lines=500)  # growth -> reset
        self._step(lvc, tmp_path, monkeypatch, log_lines=500)  # stall 1
        self._step(lvc, tmp_path, monkeypatch, log_lines=500)  # stall 2
        # Still only 2 consecutive stalls -- should NOT fire.
        assert not any(a['tag'] == 'sweep_stalled' for a in lvc._test_alerts)


# ---------------------------------------------------------------------- #
# Alert dedup                                                             #
# ---------------------------------------------------------------------- #
class TestDedup:
    def test_same_tag_dedupes_within_window(self, lvc, tmp_path, monkeypatch):
        """notify() should suppress a second call with the same tag inside
        DEDUP_MINUTES, regardless of how many _test_alerts were recorded
        by the spy. (Our spy records every CALL; dedup happens INSIDE
        the real notify. So we check the alert.log instead, which is
        the dedup source of truth.)
        """
        # Two real notify() calls -- the second should DEDUP-suppress.
        lvc.notify('test_dedup', 'WARN', 'first call')
        lvc.notify('test_dedup', 'WARN', 'second call')
        with open(lvc.LOG_PATH) as f:
            log = f.read()
        assert 'ALERT WARN test_dedup' in log
        assert 'DEDUP test_dedup' in log


# ---------------------------------------------------------------------- #
# Wiring sanity                                                           #
# ---------------------------------------------------------------------- #
def test_main_with_no_config_is_silent(lvc, tmp_path, monkeypatch):
    """No active_cluster.json present -> the script should no-op without
    error (the laptop without a cluster up shouldn't see alerts)."""
    # CONFIG_PATH already pointed at tmp_path but file doesn't exist yet.
    assert not os.path.exists(lvc.CONFIG_PATH)
    rc = lvc.main()
    assert rc == 0
    assert lvc._test_alerts == []
