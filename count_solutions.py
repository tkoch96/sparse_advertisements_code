import pickle, os
from helpers import *
import traceback

i=0
asns = {}
for row in open('cache/studied_networks_apnic.csv','r'):
	i += 1
	if i == 1:
		continue
	asn,users,samps,pref,pct = row.strip().split(',')
	asns[asn] = None
print("{} ASNs total".format(len(asns)))
exit(0)

all_networks = {}
all_popps = {}
for i in range(32):
	# fn = 'cache/popp_failure_latency_comparison_actual-{}.pkl'.format(i+1)
	fn = "cache/popp_failure_latency_comparison_testing_feature-actual-{}.pkl".format(i+1)
	if not os.path.exists(fn): continue

	networks_this_size = {}
	metrics = pickle.load(open(fn, 'rb'))

	has_sims = {}
	for random_iter in range(1000):
		try:
			try:
				metrics['compare_rets'][random_iter]['n_advs']
				has_sims[random_iter] = None
			except TypeError:
				pass
			try:
				ug_to_ip = metrics['deployment'][random_iter]['ug_to_ip']
				popps = metrics['deployment'][random_iter]['popps']
				for ips in ug_to_ip.values():
					for ip in ips:
						ntwrk = ip32_to_24(ip)
						networks_this_size[ntwrk] = None
						all_networks[ntwrk] = None
				for popp in popps:
					all_popps[popp] = None
			except TypeError: # error this sim
				traceback.print_exc()
				pass
		except KeyError:
			pass

	print("{} /24s this deployment size".format(len(networks_this_size)))

	print("{} -- {}".format(fn,sorted(list(has_sims))))
print("{} /24s over all deployments, {} popps".format(len(all_networks), len(all_popps)))

pickle.dump(all_networks, open('cache/studied_networks.pkl','wb'))

