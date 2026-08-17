"""Fleet registry — extends the single-head alert-JSON contract to a
fleet list (SCALE-500 P0). The file stays
~/.sculptor_cluster_alert/active_cluster.json; existing single-head
consumers (ticker, liveness cron, dash sync) keep reading `head`;
fleet-aware tools iterate `fleet`.

Contract (memory: update on EVERY lifecycle event):
  fleet: [{instance_id, public_ip, shard, state, launched_at,
           instance_type, spot}]
"""
import datetime
import json
import os

PATH = os.path.expanduser('~/.sculptor_cluster_alert/active_cluster.json')


def load():
    with open(PATH) as f:
        return json.load(f)


def save(d):
    d['last_updated'] = datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    tmp = PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, PATH)


def fleet():
    return load().get('fleet', [])


def upsert(entry):
    d = load()
    fl = d.setdefault('fleet', [])
    for i, e in enumerate(fl):
        if e.get('instance_id') == entry.get('instance_id'):
            fl[i] = {**e, **entry}
            break
    else:
        fl.append(entry)
    d['active'] = bool(d.get('active')) or any(
        e.get('state') == 'running' for e in fl)
    save(d)


def remove(instance_id):
    d = load()
    d['fleet'] = [e for e in d.get('fleet', [])
                  if e.get('instance_id') != instance_id]
    save(d)
