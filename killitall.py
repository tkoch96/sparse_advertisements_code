from subprocess import call, check_output
import re
for pyt in ['python','Python']:
	out = check_output("ps aux | grep {}".format(pyt),shell=True).decode()
	for row in out.split('\n'):
		if "path_distribution_computer" in row:
			print(row)
			pnum = re.search("tom + (\d+) .+", row).group(1)
			call("kill -9 {}".format(pnum),shell=True)