import glob, os, yamlloader,yaml, numpy as np, matplotlib.pyplot as plt, re, tqdm, time
from constants import *

def strip_digs(s):
	return re.sub(r'[0-9]', '', s)

files_dir = os.path.join(DATA_DIR, 'weathermap_samples')
loads_by_loc_over_time = {}
loads_by_router_over_time = {}
loads_by_link_over_time = {}
overall_min_t = np.inf
recognized_locations = {k:None for k in ['syd', 'bom', 'sin', 'mrs','lax','was','nyc','ymq','dfw','pao','pdx','yto','sea','sjo','mia','bhs','chi','par','waw',
	'fra','vie','mad','lon','ams','mil','bru','zrh','rbx','prg']}
for fn in tqdm.tqdm(glob.glob(os.path.join(files_dir, "*.yaml")), desc='parsing yaml data'):
	try:
		t_report = int(re.search("(\d+)\.yaml", fn).group(1))
	except AttributeError:
		print("Erroring parsing {} ".format(fn))
		continue
	obj = yaml.load(open(fn,'r'),Loader=yamlloader.ordereddict.CLoader)
	overall_min_t = np.minimum(t_report, overall_min_t)

	for router in obj:
		for links in obj[router].values():
			for link in links:
				found_loc = False
				loc = strip_digs(link['peer'].split('-')[0])
				try:
					recognized_locations[loc]
					found_loc = True
				except KeyError:
					loc = strip_digs(router.split('-')[0])
					try:
						recognized_locations[loc]
						found_loc = True
					except KeyError:
						pass
				if not found_loc:
					loc = strip_digs(link['peer'].split('-')[0])
					recognized_locations[loc] = None
					found_loc = True
				linkname = link['peer']
				for arr,k in zip([loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time], [loc,linkname,router]):
					try:
						arr[k]
					except KeyError:
						arr[k] = {}
					try:
						arr[k][t_report] += link['load']
					except KeyError:
						arr[k][t_report] = link['load']
print(recognized_locations)
for loc,vals in loads_by_loc_over_time.items():
	t = np.array(sorted(list(vals.keys()))) - overall_min_t
	loads = np.array([vals[_t] for _t in sorted(list(vals.keys()))])
	plt.plot(t,loads,label=loc)
plt.legend(fontsize=6)
plt.xlabel("Time (s)")
plt.ylabel("Load")
plt.savefig('figures/ovh_load_by_location_over_time.pdf')