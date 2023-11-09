import glob, os, yamlloader,yaml, numpy as np, matplotlib.pyplot as plt, re, tqdm, time, pickle, re, zipfile
from constants import *
from helpers import *

import multiprocessing

def strip_digs(s):
	return re.sub(r'[0-9]', '', s)

def fetch_files():
	nums = []
	base_url = "https://dataverse.uclouvain.be/api/access/datafile/"
	look_next = False
	for row in open('data/weathermap_xml.xml', 'r'):
		if base_url in row:
			data_id = re.search("datafile\/(\d+)\"", row).group(1)
			# if not os.path.exists(os.path.join('data', data_id + ".zip")):
			nums.append(data_id)
			look_next = True
			continue
		if look_next:
			if "yaml" not in row:
				del nums[-1]
			look_next = False
	tmp_cmds_fn = "tmp_fetch_cmds.sh"
	api_key = "c63e7428-43fb-4e80-af11-9d41110b8562"
	with open(tmp_cmds_fn, 'w') as f:
		for num in nums:
			f.write("curl -s -o {} \"{}{}?key={}\" & \n".format(os.path.join('data', num +".zip"), base_url, num, api_key))

def mp_parse(*args):
	fns,worker_i, = args[0]
	print("Starting up in worker {}".format(worker_i))
	recognized_locations = {k:None for k in ['syd', 'bom', 'sin', 'mrs','lax','was','nyc','ymq','dfw','pao','pdx','yto','sea','sjo','mia','bhs','chi','par','waw',
		'fra','vie','mad','lon','ams','mil','bru','zrh','rbx','prg']}
	save_fn = os.path.join('data', 'tmp_save_worker_{}.pkl'.format(worker_i))
	n_links_by_uid = {}
	loads_by_loc_over_time = {}
	loads_by_router_over_time = {}
	loads_by_link_over_time = {}
	n_total = len(fns)
	for fni, fn in enumerate(fns):
		print("{} percent done in worker {}".format(round(fni*100/n_total, 2), worker_i))
		with zipfile.ZipFile(fn) as z:
			for yaml_fn in z.namelist():
				try:
					t_report = int(re.search("(\d+)\.yaml", yaml_fn).group(1))
				except AttributeError:
					print("Erroring parsing {}->{}".format(fn,yaml_fn))
					continue
				obj = yaml.load(z.open(yaml_fn),Loader=yamlloader.ordereddict.CLoader)
				for router in obj:
					for links in obj[router].values():
						for linki,link in enumerate(links):
							found_loc = False
							link['label'] = "#{}".format(linki) 
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
							linkname = loc + "-" + router + "-" + link['peer']
							i=0
							for arr,k in zip([loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time], [loc,linkname,router]):
								try:
									arr[k]
								except KeyError:
									arr[k] = {}
								try:
									arr[k][t_report] += link['load']
									# if i==1:
									# 	print(link)
									# 	print(router)
									# 	print(k)
									# 	print(t_report)
									# 	print(fn)
									# 	exit(0)
								except KeyError:
									arr[k][t_report] = link['load']
								try:
									n_links_by_uid[k]
								except KeyError:
									n_links_by_uid[k] = {}
								n_links_by_uid[k][link['label']] = None
								i+=1
				if np.random.random() > .9999:
					pickle.dump([loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time,n_links_by_uid], open(save_fn,'wb'))

		return {
			"n_links_by_uid": n_links_by_uid,
			"loads_by_loc_over_time": loads_by_loc_over_time,
			"loads_by_router_over_time": loads_by_router_over_time,
			"loads_by_link_over_time": loads_by_link_over_time,
		}


def parse_and_plot():

	files_dir = os.path.join(DATA_DIR, 'weathermap_zips')

	
	save_fn = os.path.join('data', 'weathermap_parsed_stats.pkl')
	if True:#not os.path.exists(save_fn):
		fns = list(sorted(glob.glob(os.path.join(files_dir, "*.zip"))))
		n_workers = multiprocessing.cpu_count() // 2
		fn_chunks = split_seq(fns, n_workers)
		all_args = []
		for worker_i in range(n_workers):
			all_args.append((fn_chunks[worker_i], worker_i, ))
		ppool = multiprocessing.Pool(processes=n_workers)
		rets = ppool.map(mp_parse, all_args)
		n_links_by_uid = {}
		loads_by_loc_over_time = {}
		loads_by_router_over_time = {}
		loads_by_link_over_time = {}
		global_rets = [loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time,n_links_by_uid]
		global_keys = ["loads_by_loc_over_time", "loads_by_link_over_time", "loads_by_router_over_time", "n_links_by_uid"]
		print(loads_by_loc_over_time)
		for ret in rets:
			for k,global_ret in zip(global_keys, global_rets):
				if k == 'n_links_by_uid':
					for uid in ret[k]:
						try:
							global_ret[uid]
						except KeyError:
							global_ret[uid] = {}
						for link_lab in ret[k][uid]:
							try:
								global_ret[uid][link_lab] = None
							except KeyError:
								global_ret[uid] = {link_lab: None}
				else:
					for uid in ret[k]:
						try:
							global_ret[uid]
						except KeyError:
							global_ret[uid] = {}
						for link_time in ret[k][uid]:
							try:
								global_ret[uid][link_time] += ret[k][uid][link_time]
							except KeyError:
								global_ret[uid][link_time] = ret[k][uid][link_time]
			print(loads_by_loc_over_time)

					
		pickle.dump(global_rets, open(save_fn,'wb'))
	else:
		loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time,n_links_by_uid = pickle.load(open(save_fn,'rb'))
	overall_min_t = min([t_report for k in loads_by_loc_over_time for t_report in loads_by_loc_over_time[k]])
	n_links_by_uid = {k:len(v) for k,v in n_links_by_uid.items()}


	# for loc,vals in loads_by_loc_over_time.items():
	# 	t = np.array(sorted(list(vals.keys()))) - overall_min_t
	# 	t = t / (3600 * 24)
	# 	loads = np.array([vals[_t] for _t in sorted(list(vals.keys()))])
	# 	plt.plot(t,loads,label=loc)
	# plt.legend(fontsize=6)
	# plt.xlabel("Time (days)")
	# plt.ylabel("Load")
	# plt.savefig('figures/ovh_load_by_location_over_time.pdf')
	# plt.clf(); plt.close()

	sorted_ks = sorted(loads_by_link_over_time)
	structured_link_utilizations = {}
	daily_variance = {}
	peak_utils = {}
	days = {}
	for lnk in sorted_ks:
		vals = loads_by_link_over_time[lnk]
		t = np.array(sorted(list(vals.keys()))) - overall_min_t
		t = t / (3600 * 24)
		loads = np.array([vals[_t] / n_links_by_uid[lnk] for _t in sorted(list(vals.keys()))])
		structured_link_utilizations[lnk] = (t,loads)
		peak_utils[lnk] = {}
		daily_variance[lnk] = {}
		for _t,_l in zip(t,loads):
			day = int(_t)
			try:
				peak_utils[lnk][day] = np.maximum(_l, peak_utils[lnk][day])
			except KeyError:
				peak_utils[lnk][day] = _l
			try:
				daily_variance[lnk][day].append(_l)
			except KeyError:
				daily_variance[lnk][day] = [_l]
			days[day] = None
		for day,ls in daily_variance[lnk].items():
			if len(ls) > 1:
				daily_variance[lnk][day] = np.var(ls)
			else:
				daily_variance[lnk][day] = 0

		# plt.plot(t,loads,label=lnk)
	# plt.xlabel("Time (days)")
	# plt.ylabel("Load")
	# plt.savefig('figures/ovh_load_by_links_over_time.pdf')
	# plt.clf(); plt.close()

	print("{} links total".format(len(sorted_ks)))
	days = sorted(list(days))
	pctls = [99,99.9,99.99, 100]
	peak_util_dist_over_days = {pctl:{day:None for day in days} for pctl in pctls}
	for day in tqdm.tqdm(days, desc="Computing peak utilization percentilesg over days..."):
		these_lnk_utils = [peak_utils[lnk].get(day,0) for lnk in peak_utils]
		peak_util_dist_over_days
		for pctl in pctls:
			peak_util_dist_over_days[pctl][day] = np.percentile(these_lnk_utils, pctl)

	for pctl in pctls:
		plt.plot(days, [peak_util_dist_over_days[pctl][day] for day in days], label="{}th pctl".format(pctl))
	plt.xlabel("Day")
	plt.ylabel("Peak Utilization Across Links")
	plt.legend()
	plt.ylim([0,100])
	plt.savefig('figures/ovh_peakutil_over_time_distribution.pdf')
	plt.clf(); plt.close()

	for lnk in daily_variance:
		first_days_var = np.mean(list([daily_variance[lnk].get(day,0) for day in range(10)]))
		if first_days_var < 1e-8: first_days_var = 1
		plt.plot(days, [daily_variance[lnk].get(day,0) / first_days_var for day in days])
	plt.xlabel("Day")
	plt.ylabel("Intra-Day Variance")
	plt.savefig('figures/ovh_variance_over_time_distribution.pdf')
	plt.clf(); plt.close()

if __name__ == "__main__":
	# fetch_files()
	parse_and_plot()




