"""Unit tests for the email-to-SMS gateway (~/Documents/budgeter/emailer.py).

This is the actual outbound path every SCULPTOR cluster alert + heartbeat
uses (liveness_check.py / heartbeat.py import send_notification from here).
The existing tests/test_liveness_check.py mocks _send_sms entirely, so the
gateway itself -- which SMTP server, which recipient, what the message looks
like -- was never exercised. These tests close that gap.

The module lives outside the repo (shared with the budgeter project), so we
load it by absolute path via importlib, and we monkeypatch smtplib so no real
mail is sent.

Run with:  pytest tests/test_emailer_sms.py -v
"""
import importlib.util
import os
import sys

import pytest

EMAILER_PATH = os.path.expanduser('~/Documents/budgeter/emailer.py')


def _load_emailer():
    if not os.path.exists(EMAILER_PATH):
        pytest.skip("emailer.py not installed at {}".format(EMAILER_PATH))
    spec = importlib.util.spec_from_file_location('emailer', EMAILER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['emailer'] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeSMTP:
    """Stand-in for smtplib.SMTP_SSL used as a context manager. Records the
    constructor args and every call so tests can assert on them."""
    last = None

    def __init__(self, host, port, *a, **k):
        _FakeSMTP.last = self
        self.host = host
        self.port = port
        self.login_args = None
        self.sent = []          # list of email.message.Message
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)


@pytest.fixture
def em(monkeypatch):
    mod = _load_emailer()
    _FakeSMTP.last = None
    monkeypatch.setattr(mod.smtplib, 'SMTP_SSL', _FakeSMTP)
    return mod


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_connects_to_gmail_ssl_465(em):
    em.send_notification("hello")
    s = _FakeSMTP.last
    assert s is not None, "SMTP_SSL was never constructed"
    assert (s.host, s.port) == ('smtp.gmail.com', 465)
    assert s.entered and s.exited, "must use the connection as a context manager"


def test_logs_in_with_configured_credentials(em):
    em.send_notification("hello")
    assert _FakeSMTP.last.login_args == (em.SENDER_EMAIL, em.APP_PASSWORD)


def test_sends_to_sms_gateway_recipient(em):
    em.send_notification("hello")
    msg = _FakeSMTP.last.sent[0]
    assert msg['To'] == em.DESTINATION
    assert msg['From'] == em.SENDER_EMAIL
    # The configured destination is a carrier SMS gateway, not a mailbox.
    assert em.DESTINATION.endswith('@vtext.com')


def test_body_and_default_subject(em):
    em.send_notification("body text here")
    msg = _FakeSMTP.last.sent[0]
    assert msg.get_payload() == "body text here"
    # Default subject is the budgeter banner (backward compat).
    assert msg['Subject'] == '💰 Copilot Budget Update'


def test_custom_subject_is_used(em):
    # This is the path the SCULPTOR alerter uses to make alerts distinguishable.
    em.send_notification("sweep dead", subject='🚨 SCULPTOR alert')
    assert _FakeSMTP.last.sent[0]['Subject'] == '🚨 SCULPTOR alert'


def test_exactly_one_message_sent_per_call(em):
    em.send_notification("one")
    assert len(_FakeSMTP.last.sent) == 1


# --------------------------------------------------------------------------- #
# Failure behaviour: the gateway SWALLOWS exceptions (prints, returns None).
# These tests document that contract so a future change that makes failures
# raise/return-False is a conscious, test-visible decision -- it matters
# because a silently-failed send means the user never learns the cluster died.
# --------------------------------------------------------------------------- #
def test_smtp_failure_is_swallowed_not_raised(em, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(em.smtplib, 'SMTP_SSL', boom)
    # Must not raise -- current contract is best-effort fire-and-forget.
    assert em.send_notification("hello") is None


def test_login_failure_is_swallowed(em, monkeypatch):
    class _LoginFails(_FakeSMTP):
        def login(self, user, password):
            raise RuntimeError("bad app password")

    monkeypatch.setattr(em.smtplib, 'SMTP_SSL', _LoginFails)
    # send_notification catches inside the `with`, so this returns None.
    assert em.send_notification("hello") is None
