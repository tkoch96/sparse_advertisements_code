"""Kill all SCULPTOR processes bound to a given port.

Usage:
    python killitall.py <port>

Finds every Python process whose command line contains the port number
and SIGKILLs it. Useful when a run crashes and leaves orphan workers /
strategy subprocesses pinned to a port — without this, the next run on
the same port silently cross-talks with the orphans.

Only matches by port number so multiple concurrent simulations on
different ports don't interfere with each other.
"""
from subprocess import call, check_output,time, sys
import re
port = int(sys.argv[1])

print("KILLING ALL PROCESSES ON PORT {} --- EXIT IF YOU DONT WANT THIS".format(port))
time.sleep(5)

for pyt in ['python','Python']:
	out = check_output("ps aux | grep {}".format(pyt),shell=True).decode()
	for row in out.split('\n'):
		if ("path_distribution_computer" in row and str(port) in row) or \
			("testing_priorities" in row and str(port) in row):
			print(row)
			try:
				pnum = re.search(r"tom + (\d+) .+", row).group(1)
			except:
				try:
					pnum = re.search(r"ubuntu + (\d+) .+", row).group(1)
				except:
					pnum = re.search(r"tomkoch + (\d+) .+", row).group(1)


			print("kill -9 {}".format(pnum))
			call("kill -9 {}".format(pnum),shell=True)