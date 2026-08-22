#!/usr/bin/env python3
"""PreToolUse(Bash) hook: no stopping a VM through an unguarded door.

Wired up in `.claude/settings.json`. Reads the Claude Code hook JSON on
stdin; exit 2 blocks the tool call and shows stderr to the agent.

Why this exists
---------------
`cluster/vmctl.py stop` harvests every live run and refuses to stop when
bytes are still only on the VM. That gate is worth nothing if the agent
reaches for `aws ec2 stop-instances` instead -- which is exactly what
happened repeatedly before these tools existed, and is how several
experiments were paid for and then never read.

So: raw stop/terminate/teardown commands are blocked, and the agent is
pointed at the one door that has the gate on it. `vmctl --force` remains
available for the genuine "yes, abandon those logs" case, and says out
loud what it is abandoning.

This hook does no network I/O -- it is pure string inspection, so it adds
no latency to ordinary Bash calls.
"""

import json
import re
import sys

# (pattern, what to do instead)
BLOCKED = [
    (r'\bec2\b.*\bstop-instances\b',
     'python -m cluster.vmctl stop <ref>'),
    (r'\bec2\b.*\bterminate-instances\b',
     'python -m cluster.vmctl terminate <ref> --yes'),
    (r'\bstop_instances\s*\(',
     'python -m cluster.vmctl stop <ref>'),
    (r'\bterminate_instances\s*\(',
     'python -m cluster.vmctl terminate <ref> --yes'),
    (r'\bray\s+down\b',
     'python -m cluster.vmctl stop <ref>  (or teardown.sh once every run '
     'is harvested)'),
    (r'teardown\.sh',
     'python -m cluster.vmctl stop <ref> for each box, then teardown.sh'),
]

# Our own tools obviously must not block themselves, and neither should a
# read-only describe that merely mentions the word.
ALLOW = [
    r'cluster\.vmctl',
    r'cluster/vmctl\.py',
    r'describe-instances',
    r'describe_instances',
    # Commands that only ever MENTION the dangerous phrases -- git
    # operations (a commit message describing this very hook tripped it,
    # 2026-08-22), and read-only text tools. The guard is about EXECUTING
    # a stop, not about the words appearing in a string.
    r'^\s*git\s',
    r'^\s*(grep|rg|ag|sed|awk|cat|less|head|tail|diff)\s',
]


def main():
    try:
        payload = json.load(sys.stdin)
    except (ValueError, IOError):
        return 0
    if payload.get('tool_name') != 'Bash':
        return 0
    cmd = (payload.get('tool_input') or {}).get('command', '')
    if not cmd:
        return 0

    for pat in ALLOW:
        if re.search(pat, cmd):
            return 0

    for pat, instead in BLOCKED:
        if re.search(pat, cmd, re.I):
            sys.stderr.write(
                'BLOCKED by cluster/hooks/guard_vm_stop.py.\n\n'
                'This command stops or destroys a VM without going through '
                'the harvest gate, and unharvested logs die with the box.\n\n'
                'Use instead:\n    {}\n\n'
                'That command pulls every live run first and refuses to stop '
                'if any bytes are still only on the VM. If you genuinely '
                'mean to abandon the logs, it takes --force and will print '
                'exactly what is being thrown away.\n'.format(instead))
            return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
