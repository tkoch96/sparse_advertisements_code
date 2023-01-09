from subprocess import call, check_output
import re

out = check_output("ps aux | grep python",shell=True).decode()
for row in out.split('\n'):
	if "path_distribution_computer" in row:
		print(row)
		pnum = re.search("tom + (\d+) .+", row).group(1)
		call("kill -9 {}".format(pnum),shell=True)