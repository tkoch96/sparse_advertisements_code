
a_lats = {}
all_a_lats = {}
for row in open('cache/vultr_anycast_latency.csv','r'):
	t,ip,lat,pop = row.strip().split(",")
	all_a_lats[ip] = None
	if lat == '-1':
		continue
	a_lats[ip] = None

print("{} in total, {} with response".format(len(all_a_lats), len(a_lats)))		