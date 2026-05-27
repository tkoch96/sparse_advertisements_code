"""Worker_Manager re-export.

The project has been Ray-only since the migration in mid-2026. The
historical ZMQ Worker_Manager that used to live in this file (spawning
path_distribution_computer.py subprocesses and talking to them over ZMQ
REQ/REP sockets) is gone -- it was a footgun (imports like
`from worker_comms import Worker_Manager` silently resolved to ZMQ
unless launched via the old `run_ray.py` aliasing wrapper).

Going forward, this file is just a re-export of the Ray actor pool, so
the canonical `from worker_comms import Worker_Manager` resolves to the
Ray implementation everywhere -- no sys.modules aliasing needed.

See worker_comms_ray for the actual implementation.
"""
from worker_comms_ray import *  # noqa: F401,F403
from worker_comms_ray import Worker_Manager  # noqa: F401  (explicit re-export)
