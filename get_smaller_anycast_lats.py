from helpers import *
from constants import *
import tqdm,os
lat_fn = os.path.join(CACHE_DIR, 'vultr_ingress_latencies_by_dst.csv')
ug_perfs = {}
for row in tqdm.tqdm(open(lat_fn, 'r'), desc="Parsing per-ingress VULTR measurements."):
	# if np.random.random() > .9999999:break
	fields = row.strip().split(',')
	t,ip,pop,peer,_,lat = fields
	metro = 'tmp'
	asn = ip#ip32_to_24(ip)
	ug = (metro,asn)
	try:
		ug_perfs[ug]
	except KeyError:
		ug_perfs[ug] = {}
	ug_perfs[ug] = None
rows = []
for row in tqdm.tqdm(open(os.path.join(CACHE_DIR, 'vultr_anycast_latency.csv')
	,'r'),desc="Parsing VULTR anycast latencies"):
	# if np.random.random() > .999999:break
	_,ip,lat,pop = row.strip().split(',')
	if lat == '-1': continue
	metro = 'tmp'
	asn = ip#ip32_to_24(ip)
	ug = (metro,asn)
	try:
		ug_perfs[ug]
	except KeyError:
		continue
	rows.append(row)
with open(os.path.join(CACHE_DIR, 'vultr_anycast_latency_smaller.csv'),'w') as f:
	for row in rows:
		f.write(row)
