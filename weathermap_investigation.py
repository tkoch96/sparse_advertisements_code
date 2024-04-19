import glob, os, yamlloader,yaml, numpy as np, matplotlib.pyplot as plt, re, tqdm, time, pickle, re, zipfile
from constants import *
from helpers import *

import multiprocessing

def strip_digs(s):
	return re.sub(r'[0-9]', '', s)

def fetch_files():
	### Writes a command file that you'd execute to download all the data
	nums = []
	base_url = "https://dataverse.uclouvain.be/api/access/datafile/"
	look_next = False
	## you get this from the main page: export metadata as DDI
	for row in open('data/weathermap_xml.xml', 'r'):
		if base_url in row:
			data_id = re.search("datafile\/(\d+)\"", row).group(1)
			if not os.path.exists(os.path.join('data', 'weathermap_zips', data_id + ".zip")):
				nums.append(data_id)
			# nums.append(data_id)
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
			f.write("curl -s -o {} \"{}{}?key={}\" & \n".format(os.path.join('data', 'weathermap_zips', num +".zip"), base_url, num, api_key))

def mp_parse(*args):
	### Multiprocessing worker for reading data
	### Reading yaml is slow, so multiple workers helps
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
	try:
		for fni, fn in enumerate(fns):
			print("{} percent done in worker {}".format(round(fni*100/n_total, 2), worker_i))
			with zipfile.ZipFile(fn) as z:
				for yaml_fn in z.namelist():
					if ".log" in yaml_fn: continue
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
								loc = loc.lower()
								try:
									recognized_locations[loc]
									found_loc = True
								except KeyError:
									loc = strip_digs(router.split('-')[0])
									loc = loc.lower()
									try:
										recognized_locations[loc]
										found_loc = True
									except KeyError:
										pass
								if not found_loc:
									loc = strip_digs(link['peer'].split('-')[0])
									loc = loc.lower()
									print('unrecognized location {}'.format(loc))
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
	except:
		import traceback
		traceback.print_exc()
		return "womp womp"
	pickle.dump([loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time,n_links_by_uid], open(save_fn,'wb'))
	return "done"


def parse_and_plot():
	#### Parses all yaml files, caches results, generates plots
	files_dir = os.path.join(DATA_DIR, 'weathermap_zips')
	hlstats_save_fn = os.path.join('data', 'weathermap_highlevelstats.pkl')
	if not os.path.exists(hlstats_save_fn):
		parsed_yaml_save_fn = os.path.join('data', 'weathermap_parsed_stats.pkl')
		n_workers = 32#multiprocessing.cpu_count()//2
		if not os.path.exists(parsed_yaml_save_fn):
			fns = list(sorted(glob.glob(os.path.join(files_dir, "*.zip"))))
			fn_chunks = split_seq(fns, n_workers)
			all_args = []
			for worker_i in range(n_workers):
				all_args.append((fn_chunks[worker_i], worker_i, ))

			ppool = multiprocessing.Pool(processes=n_workers)
			exit_codes = ppool.map(mp_parse, all_args)
			print(exit_codes)
			

			n_links_by_uid = {}
			loads_by_loc_over_time = {}
			loads_by_router_over_time = {}
			loads_by_link_over_time = {}
			global_rets = [loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time,n_links_by_uid]
			global_keys = ["loads_by_loc_over_time", "loads_by_link_over_time", "loads_by_router_over_time", "n_links_by_uid"]
			for worker_i in range(n_workers):
				print("Parsing return values from worker {}".format(worker_i))
				_ret = pickle.load(open(os.path.join('data', 'tmp_save_worker_{}.pkl'.format(worker_i)), 'rb'))
				ret = {k:_ret[i] for i,k in enumerate(global_keys)}
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



						
			pickle.dump(global_rets, open(parsed_yaml_save_fn,'wb'))
		else:
			loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time,n_links_by_uid = pickle.load(open(parsed_yaml_save_fn,'rb'))
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
		for lnk in tqdm.tqdm(sorted_ks, desc="Sorting stats by day..."):
			vals = loads_by_link_over_time[lnk]
			loads, t = [], []
			peak_utils[lnk] = {}
			daily_variance[lnk] = {}
			for _t in sorted(list(vals.keys())):
				_l = vals[_t] / n_links_by_uid[lnk]
				loads.append(_l)
				t.append((_t - overall_min_t) / (3600 * 24))
				day = int(t[-1])
				try:
					peak_utils[lnk][day] = np.maximum(_l, peak_utils[lnk][day])
				except KeyError:
					peak_utils[lnk][day] = _l
				try:
					daily_variance[lnk][day].append(_l)
				except KeyError:
					daily_variance[lnk][day] = [_l]
				days[day] = None
			t = np.array(t)
			loads = np.array(loads)
			structured_link_utilizations[lnk] = (t,loads)
			del loads_by_link_over_time[lnk]
			
			# plt.plot(t,loads,label=lnk)
		# plt.xlabel("Time (days)")
		# plt.ylabel("Load")
		# plt.savefig('figures/ovh_load_by_links_over_time.pdf')
		# plt.clf(); plt.close()


		aggregated_daily_variance = {}
		for lnk in structured_link_utilizations:
			aggregated_daily_variance[lnk] = {}
			for day,ls in daily_variance[lnk].items():
				if len(ls) > 1:
					aggregated_daily_variance[lnk][day] = np.max(ls) - np.min(ls)
				else:
					aggregated_daily_variance[lnk][day] = 0
		del structured_link_utilizations

		print("{} links total".format(len(sorted_ks)))
		pickle.dump({
			'daily_variance': daily_variance,
			'aggregated_daily_variance': aggregated_daily_variance,
			'peak_utils': peak_utils,
			'days': days,
		}, open(hlstats_save_fn, 'wb'))
	else:
		print("Loading precomputed high-level stats")
		d = pickle.load(open(hlstats_save_fn, 'rb'))
		daily_variance, aggregated_daily_variance, peak_utils,days = d['daily_variance'],d['aggregated_daily_variance'],d['peak_utils'],d['days']
		print("Done loading precomputed high-level stats")
	
	

	days = sorted(list(days))
	pctls = [99,99.9,99.99, 100]
	peak_util_dist_over_days = {pctl:{day:None for day in days} for pctl in pctls}
	for day in tqdm.tqdm(days, desc="Computing peak utilization percentilesg over days..."):
		these_lnk_utils = [peak_utils[lnk].get(day,0) for lnk in peak_utils]
		for pctl in pctls:
			peak_util_dist_over_days[pctl][day] = np.percentile(these_lnk_utils, pctl)

	for pctl in pctls:
		plt.plot(days, [peak_util_dist_over_days[pctl][day] for day in days], label="{}th pctl".format(pctl))
	plt.xlabel("Day")
	plt.ylabel("Peak Utilization Across Links")
	plt.legend()
	# plt.ylim([0,100])
	plt.savefig('figures/ovh_peakutil_over_time_distribution.pdf')
	plt.clf(); plt.close()

	ns = [7,30,60,90,120]
	max_period_num_by_n = {n:np.max([day//n for day in days])+1 for n in ns}
	print(max_period_num_by_n)
	peak_utils_n_day_period = {n:{lnk:{i:[] for i in range(max_period_num_by_n[n])} for lnk in peak_utils} for n in ns}
	pctl_of_interest = 95
	for n in ns:
		for day in days:
			period_num = day//n
			for lnk in peak_utils:
				link_util = peak_utils[lnk].get(day,0)
				peak_utils_n_day_period[n][lnk][period_num].append(link_util)
		for lnk in peak_utils_n_day_period[n]:
			for period_num,vs in peak_utils_n_day_period[n][lnk].items():
				try:
					peak_utils_n_day_period[n][lnk][period_num] = np.percentile(vs,pctl_of_interest)
				except IndexError:
					if len(vs) == 1:
						peak_utils_n_day_period[n][lnk][period_num] = vs[0]
					else:
						peak_utils_n_day_period[n][lnk][period_num] = 0
		# if n == 120:
		# 	for lnk in peak_utils:
		# 		print(peak_utils_n_day_period[n][lnk])
		# 		if np.random.random() > .9:
		# 			exit(0)
	for n in ns:
		deltas = []
		for lnk in peak_utils_n_day_period[n]:
			max_delta = 0
			for period_num in range(1,max_period_num_by_n[n]):
				if peak_utils_n_day_period[n][lnk][period_num] == 0 or peak_utils_n_day_period[n][lnk][period_num-1] == 0:
					continue
				delta = peak_utils_n_day_period[n][lnk][period_num] - peak_utils_n_day_period[n][lnk][period_num-1]
				max_delta = np.maximum(delta,max_delta)
			deltas.append(max_delta)
		x,cdf_x = get_cdf_xy(deltas)
		plt.plot(x,cdf_x,label="Agg over {} days".format(n))
	plt.legend()
	plt.xlabel('Max Change in Peak Utilization Over Period')
	plt.ylabel("CDF of Links")
	plt.grid(True)
	plt.savefig('figures/change_in_peak_util_over_periods.pdf')
	plt.clf(); plt.close()


	aggregated_daily_variance_by_day = {day:[] for day in days}

	for lnk in aggregated_daily_variance:
		first_days_var = 1#np.mean(list([aggregated_daily_variance[lnk].get(day,0) for day in range(10)]))
		for day in days:
			aggregated_daily_variance_by_day[day].append(aggregated_daily_variance[lnk].get(day,0))
		if first_days_var < 1e-8: first_days_var = 1
		plt.plot(days, [aggregated_daily_variance[lnk].get(day,0) / first_days_var for day in days])
	plt.xlabel("Day")
	plt.ylabel("Intra-Day Variance")
	plt.savefig('figures/ovh_variance_over_time_distribution.pdf')
	plt.clf(); plt.close()

	pctls = [50,75,90,99]
	variance_pctles_by_day = {pctl:[] for pctl in pctls}
	for day in days:
		for pctl in pctls:
			variance_pctles_by_day[pctl].append(np.percentile(aggregated_daily_variance_by_day[day], pctl))
	for pctl in pctls:
		plt.plot(days, variance_pctles_by_day[pctl], label="{}th pctl".format(pctl))
	plt.xlabel("Day")
	plt.ylabel("Intra-Day Swing Across Links")
	plt.legend()
	# plt.ylim([0,100])
	plt.savefig('figures/intra_day_variance_over_time_pctl_links.pdf')
	plt.clf(); plt.close()


if __name__ == "__main__":
	# fetch_files()
	parse_and_plot()




