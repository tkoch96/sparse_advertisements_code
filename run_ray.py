"""
Launcher that swaps the ZMQ Worker_Manager for the Ray one and then runs an
existing driver script (eval_latency_failure.py, sparse_advertisements_v3.py,
etc.) without modifying it.

It works by:
  1. Importing `worker_comms_ray` (which initialises Ray).
  2. Aliasing `sys.modules['worker_comms'] = worker_comms_ray` so any later
	 `from worker_comms import Worker_Manager` resolves to the Ray version.
  3. Exec'ing the requested script under its own `__main__` namespace, with
	 the rest of the args forwarded so `argparse` keeps working.

Usage:
    python run_ray.py eval_latency_failure.py --port 31415 --dpsize small
    python run_ray.py eval_latency_failure --port 31415 --dpsize small   # .py optional

Env vars honoured:
    RAY_ADDRESS=auto                  # attach to an existing cluster
    RAY_ADDRESS=ray://host:10001      # ray client mode

For a quick local sanity check, just run:
    python run_ray.py eval_latency_failure --port 31415 --dpsize small
"""
import os
import sys
import runpy


def main():
	if len(sys.argv) < 2:
		print(__doc__)
		sys.exit(1)

	# Pre-import the Ray Worker_Manager and alias it as `worker_comms` so any
	# subsequent `from worker_comms import Worker_Manager` resolves to the
	# Ray actor manager.
	import worker_comms_ray as _ray_mod
	sys.modules['worker_comms'] = _ray_mod

	script = sys.argv[1]
	if not script.endswith('.py'):
		script = script + '.py'

	# Forward remaining args to the script (argparse will see them).
	sys.argv = [script] + sys.argv[2:]

	# Run the target as if invoked directly: __name__ == '__main__'.
	# run_path uses the current sys.path so imports in the script work.
	script_path = os.path.abspath(script)
	if not os.path.isfile(script_path):
		print("Script not found: {}".format(script_path))
		sys.exit(2)

	runpy.run_path(script_path, run_name='__main__')


if __name__ == '__main__':
	main()
