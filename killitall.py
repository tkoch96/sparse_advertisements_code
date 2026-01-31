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