import numpy as np, tqdm, time
ug_lats = {}
for row in tqdm.tqdm(open('cache/no_provider_vultr_ingress_latencies_by_dst.csv'),desc="reading no provider meas"):
	t,ip,pop,peer,_,lat = row.strip().split(',')
	try:
		ug_lats[ip]
	except KeyError:
		ug_lats[ip] = {}
	ug_lats[ip][pop,peer] = np.maximum(.001,float(lat))

provider_popps = {}
for row in open('cache/vultr_provider_popps.csv','r'):
	pop,peer = row.strip().split(',')
	provider_popps[pop,peer] = None

print("Read {} clients from no provider ingress lats file".format(len(ug_lats)))
included_ips = {}
for row in tqdm.tqdm(open('cache/vultr_ingress_latencies_by_dst_bck.csv','r'), 
	desc="Reading old provider lats..."):
	t,ip,pop,peer,_,lat = row.strip().split(',')
	try:
		ug_lats[ip]
	except KeyError:
		continue
	try:
		provider_popps[pop,peer]
	except KeyError:
		continue
	ug_lats[ip][pop,peer] = np.maximum(.001,float(lat))
	included_ips[ip] = None
print("Got provider latencies for {} ips".format(len(included_ips)))

with open('cache/vultr_ingress_latencies_by_dst_filled_in.csv','w') as f:
	tnow = int(time.time())
	for ip in tqdm.tqdm(included_ips,desc='writing meas'):
		for popp,lat in ug_lats[ip].items():
			f.write("{},{},{},{},{},{}\n".format(t,ip,popp[0],popp[1],0,lat))


