import glob, os, yamlloader,yaml, numpy as np,  re, tqdm, time, pickle, re, zipfile
from constants import *
from helpers import *
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from paper_plotting_functions import *
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
						pickle.dump([loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time, n_links_by_uid], open(save_fn,'wb'))
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
			loads_by_loc_over_time, loads_by_link_over_time, loads_by_router_over_time, n_links_by_uid = pickle.load(open(parsed_yaml_save_fn,'rb'))
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

	ns = [7,30,60,90,120,240]
	max_period_num_by_n = {n:np.max([day//n for day in days])+1 for n in ns}
	print(max_period_num_by_n)
	peak_utils_n_day_period = {n:{lnk:{i:[] for i in range(max_period_num_by_n[n])} for lnk in peak_utils} for n in ns}
	avg_utils_n_day_period = {n:{lnk:{i:[] for i in range(max_period_num_by_n[n])} for lnk in peak_utils} for n in ns}
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
					avg_utils_n_day_period[n][lnk][period_num] = np.mean(vs)
				except IndexError:
					if len(vs) == 1:
						peak_utils_n_day_period[n][lnk][period_num] = vs[0]
						avg_utils_n_day_period[n][lnk][period_num] = vs[0]
					else:
						peak_utils_n_day_period[n][lnk][period_num] = 0
						avg_utils_n_day_period[n][lnk][period_num] = 0
	pickle.dump([avg_utils_n_day_period, peak_utils_n_day_period], open(os.path.join('cache','ovh_cache','peak_utils_n_day_period.pkl'),'wb'))


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



def paper_plots():
	avg_utils_n_day_period, peak_utils_n_day_period = pickle.load(open(os.path.join('cache','ovh_cache','peak_utils_n_day_period.pkl'), 'rb'))
	periods_to_plot = [60,120, 240]
	f,ax = get_figure(h=2)
	for ni,n in enumerate(sorted(peak_utils_n_day_period, reverse=True)):
		if n not in periods_to_plot: continue
		deltas = []
		for lnk in peak_utils_n_day_period[n]:
			max_delta = 0
			for period_num in sorted(peak_utils_n_day_period[n][lnk]):
				if period_num == 0: continue
				if peak_utils_n_day_period[n][lnk][period_num] == 0 or peak_utils_n_day_period[n][lnk][period_num-1] == 0:
					continue
				delta = peak_utils_n_day_period[n][lnk][period_num] - peak_utils_n_day_period[n][lnk][period_num-1]
				max_delta = np.maximum(delta,max_delta)
			deltas.append(max_delta)
		x,cdf_x = get_cdf_xy(deltas,n_points=30)
		ax.plot(x,cdf_x,label="{} Day Period".format(n),marker=markers[ni])
	ax.legend()
	ax.set_xlabel('Change in 95th Pctl. Link Utilization Over N-Day Period')
	ax.set_ylabel("CDF of \nOVH Cloud Links")
	ax.grid(True)
	save_figure('change_in_peak_util_over_periods.pdf')

	over_provision_ratio = .5
	f,ax = get_figure(h=2)
	for ni,n in enumerate(sorted(peak_utils_n_day_period, reverse=True)):
		if n not in periods_to_plot: continue
		utilizations = []
		for lnk in peak_utils_n_day_period[n]:
			for period_num in sorted(peak_utils_n_day_period[n][lnk]):
				if period_num == 0: continue
				if peak_utils_n_day_period[n][lnk][period_num] == 0 or peak_utils_n_day_period[n][lnk][period_num-1] == 0:
					continue
				# over_provision_ratio = (peak_utils_n_day_period[n][lnk][period_num] - peak_utils_n_day_period[n][lnk][period_num-1]) / peak_utils_n_day_period[n][lnk][period_num-1]
				new_cap = peak_utils_n_day_period[n][lnk][period_num-1] * (1 + over_provision_ratio)
				avg_next = avg_utils_n_day_period[n][lnk][period_num]
				utilization_next = avg_next/new_cap
				if utilization_next > 1: continue ## numerical precision problems
				utilizations.append(utilization_next)

		x,cdf_x = get_cdf_xy(utilizations,n_points=50)
		ax.plot(x,cdf_x,label="{} Day Period".format(n),marker=markers[ni])
	ax.legend()
	ax.set_xlim([0,1])
	ax.set_xlabel('Utilization in Next N-Days With N-Day Planning Period')
	ax.set_ylabel("CDF of OVH Cloud\n Links,Periods")
	ax.grid(True)
	save_figure('utilization_over_periods.pdf')



	periods_to_plot = [120]
	over_provision_ratios = np.linspace(.1,.7,num=20)
	median_utilizations = {}
	congestive_events = {}
	for over_provision_ratio in over_provision_ratios:
		for ni,n in enumerate(sorted(peak_utils_n_day_period, reverse=True)):
			if n not in periods_to_plot: continue
			utilizations = []
			congestive_periods = 0
			for lnk in peak_utils_n_day_period[n]:
				for period_num in sorted(peak_utils_n_day_period[n][lnk]):
					if period_num == 0: continue
					if peak_utils_n_day_period[n][lnk][period_num] == 0 or peak_utils_n_day_period[n][lnk][period_num-1] == 0:
						continue
					# over_provision_ratio = (peak_utils_n_day_period[n][lnk][period_num] - peak_utils_n_day_period[n][lnk][period_num-1]) / peak_utils_n_day_period[n][lnk][period_num-1]
					new_cap = peak_utils_n_day_period[n][lnk][period_num-1] * (1 + over_provision_ratio)
					avg_next = avg_utils_n_day_period[n][lnk][period_num]
					if peak_utils_n_day_period[n][lnk][period_num] / new_cap > 1:
						congestive_periods += 1
					utilization_next = avg_next/new_cap
					if utilization_next > 1: continue ## numerical precision problems
					utilizations.append(utilization_next)

			try:
				median_utilizations[n].append(np.median(utilizations))
				congestive_events[n].append(congestive_periods)
			except KeyError:
				median_utilizations[n] = [np.median(utilizations)]
				congestive_events[n] = [congestive_periods]
	
	f,ax = get_figure(h=2)
	for ni,n in enumerate(sorted(median_utilizations)):
		ax.plot(100*over_provision_ratios,median_utilizations[n],linestyle='dotted', label="{} Day Planning Period".format(n),marker=markers[ni])
	ax.annotate("Link Utilization", (40,.6))

	ax2 = ax.twinx()
	for ni,n in enumerate(sorted(median_utilizations)):
		ax2.plot(100*over_provision_ratios,congestive_events[n],marker=markers[ni])
	ax2.set_ylabel("Congestion Events")
	ax2.set_ylim([0,7000])
	ax2.annotate("Congestion", (20,2100))

	ax.set_xlabel("Link Over Provisioning Factor (Pct)")
	ax.set_ylabel("Future Median \nLink Utilization")
	# ax.legend(fontsize=12)
	ax.set_ylim([0,1.0])
	ax.grid(True)
	
	save_figure('median_utilization_over_overprovisioning.pdf')

def find_flash_crowd(data, threshold_mt, location):
	crowds_arr_x = []
	crowds_arr_y = []
	sorted_data = {k: v for k,v in sorted(data.items())}
	sorted_data = list(sorted(data.items(), key = lambda el : el[0]))
	initial_time = sorted_data[0][0]

	crowd_start = 0
	crowd_end = 0
	crowd_started = False

	times = list([el[0] for el in sorted_data])
	loads = list([el[1] for el in sorted_data])

	crowds = []
	critical_n_ended = 3 ## minimum number to end a flash crowd
	critical_n_started = 3 ## minimum number to start a flash crowd
	n_ended = 0
	n_started = 0
	print("total number of times {}".format(len(times)))
	n_avg_over = 100 ## ~5N minutes
	for i in range(len(times)):
		avg_start,avg_end = i-n_avg_over,i+n_avg_over
		if i < n_avg_over:
			avg_start = 0
		elif i + n_avg_over > len(loads) - 1:
			avg_end = len(loads) - 1
		recent_threshold = np.mean(loads[avg_start:avg_end]) * threshold_mt
		if recent_threshold < 500: continue ## trivial
		if loads[i] > recent_threshold:
			n_started += 1
			if n_started >= critical_n_started:
				if not crowd_started:
					crowd_start = i - critical_n_started
					crowd_started = True
					flash_intensity = loads[i] / np.mean(loads[avg_start:avg_end])
			n_ended = 0
		else:
			n_started = 0
			if crowd_started:
				n_ended += 1
			if n_ended >= critical_n_ended:
				crowd_end = i-critical_n_ended
				crowd_started = False
				print("Crowd Detected for around ", (crowd_end - crowd_start)/12, " hours")

				print("{} {} {}".format(crowd_start, crowd_end, flash_intensity))
				print(loads[crowd_start-5:crowd_end+5])

				# plt.plot(loads[crowd_start-n_avg_over:crowd_end+n_avg_over])
				# plt.savefig('example_loads.pdf')
				# plt.clf(); plt.close()


				crowds.append((crowd_start, crowd_end, flash_intensity))
				crowd_start = 0
				crowd_end = 0   
				n_ended = 0
				flash_intensity = 0

	return times, loads, crowds      

def smooth_ewma(arr, alpha=.1):
	return arr
	new_arr = [arr[0]]
	for i in range(1,len(arr)):
		new_arr.append(alpha * new_arr[i-1] + (1 - alpha) * arr[i])
	new_arr.append(new_arr[-1])
	return np.array(new_arr) 

def ilgar_flashcrowd_analysis():

	# #### FOR SPLITTING DATA INTO PER LOCATION, EASIER TO DEBUG
	# all_data = pickle.load(open(os.path.join(DATA_DIR ,'weathermap_parsed_stats.pkl'),'rb'))
	# link_to_load = all_data[1]
	# links = list(all_data[1])
	# locations = {}
	# all_times = {}
	# for link in tqdm.tqdm(links,desc="aggregating all times"):
	# 	location = link.split('-')[0]
	# 	locations[location] = None
	# 	for t in link_to_load[link]:
	# 		all_times[t] = None
	# new_link_by_location = {}
	# all_times = sorted(list(all_times))
	# for link in tqdm.tqdm(links,desc="Filling in missing link time series data"):
	# 	location = link.split('-')[0]
	# 	try:
	# 		new_link_by_location[location]
	# 	except KeyError:
	# 		new_link_by_location[location] = {}
	# 	prev_v = None
	# 	for t in all_times:
	# 		try:
	# 			this_v = link_to_load[link][t]
	# 		except KeyError:
	# 			if prev_v is None: continue
	# 			this_v = prev_v
	# 		try:
	# 			new_link_by_location[location][t] += this_v
	# 		except KeyError:
	# 			new_link_by_location[location][t] = this_v
	# 		prev_v = this_v

	# for location in locations:
	# 	fn = os.path.join(CACHE_DIR, 'ovh_cache', 'by_location', '{}_data.pkl'.format(location))
	# 	pickle.dump(new_link_by_location[location], open(fn,'wb'))

	cache_fn = os.path.join(CACHE_DIR, 'ovh_cache', 'location_crowds_data.pkl')
	#### FOR FINDING FLASH CROWDS	
	if False:
		flash_crowd_data = pickle.load(open(cache_fn,'rb'))
		locations = sorted(list(flash_crowd_data))

		flash_crowd_data = {}
		for location in tqdm.tqdm(locations, desc="Finding flash crowds in each location..."):
			print(location)
			fn = os.path.join(CACHE_DIR, 'ovh_cache', 'by_location', '{}_data.pkl'.format(location))
			data = pickle.load(open(fn,'rb'))
			threshold = 1.5
			flash_crowd_data[location] = find_flash_crowd(data, threshold, location)
		pickle.dump(flash_crowd_data, open(cache_fn, "wb")) #Format: dictionary, keys are location names, values are arrays of [x_arr, y_arr and [(crowd_start, crowd_end)]]
	
	

	flash_crowd_data = pickle.load(open(cache_fn,'rb'))
	locations = sorted(list(flash_crowd_data))
	locations_of_interest = ['nyc', 'chi', 'zrh', 'syd', 'mil', 'mia','par','pao','sea','sin','vie', 'fra', 'nwk', 'lax', 'mad', 'yto', 'ldn', 'sgp', 'mrs', 'ymq']
	# ['ams', 'ash', 'atl', 'bhs', 'bom', 'bru', 'chi', 'dfw', 'equinix', 'eur', 'europe', 'fra', 'france', 'google', 
	# 'gra', 'gsw', 'jastel', 'las', 'lax', 'ldn', 'lej', 'lon', 'mad', 'mia', 'mil', 'mrs', 'north america', 'nwk', 
	# 'nyc', 'online', 'orange', 'oti', 'pao', 'par', 'pdx', 'prg', 'rbx', 'sbg', 'sea', 'seabone', 'sfr', 'sgp', 'sin', 
	# 'sjc', 'sjo', 'sxb', 'syd', 'telefonica', 'th', 'var', 'verizon', 'vie', 'vodafone', 'was', 'waw', 'ymq', 'yto', 'zrh']
	if False:
		all_times = {}
		for location in tqdm.tqdm(locations, desc="Finding overlapping flash crowds"):
			if location not in locations_of_interest: continue
			time_arr = flash_crowd_data[location][0]
			load_arr = flash_crowd_data[location][1]
			crowds_arr = flash_crowd_data[location][2]
		
			# print("Location: {}, crowds: {}".format(location, len(crowds_arr)))
			sorted_crowds = list(sorted(crowds_arr, key = lambda el : el[0]))
			for i,(crowd_start,crowd_stop,_) in enumerate(sorted_crowds):
				for t in range(crowd_start,crowd_stop):
					## t is an index
					actual_time = time_arr[t]
					try:
						all_times[actual_time].append((location, i))
					except KeyError:
						all_times[actual_time] = [(location, i)]
		print("Finding overlaps...")
		ranked_times = list(sorted(all_times.items(), key = lambda el : -1 * len(el[1])))

		
		used_fc_ids = {}
		all_n_events = []
		for plti,(t,events) in tqdm.tqdm(enumerate(ranked_times), desc="Plotting each FC event.."):
			used = False
			for k in events:
				try:
					used_fc_ids[k]
					used = True
					break
				except KeyError:
					pass
			if used:
				continue
			for loci,(location, _id) in enumerate(events):
				used_fc_ids[location,_id] = None
			all_n_events.append(len(events))
			
		x,cdf_x = get_cdf_xy(all_n_events)
		plt.plot(x,cdf_x)
		plt.xlabel("Number of Sites a Flash Crowd Affects")
		plt.ylabel("CDF of Flash Crowds")
		plt.grid(True)
		plt.savefig('figures/number_sites_per_flash_crowd.pdf')


		n_to_plot,n_plotted = 50,0
		f,ax = plt.subplots(n_to_plot,1)
		f.set_size_inches(7,4*n_to_plot)
		colors = ['red','blue','black','salmon','maroon', 'tan', 'coral','lawngreen', 'grey', 'peru','darkgoldenrod','violet','crimson']
		used_fc_ids = {}
		for plti,(t,events) in tqdm.tqdm(enumerate(ranked_times), desc="Plotting each FC event.."):
			used = False
			for k in events:
				try:
					used_fc_ids[k]
					used = True
					break
				except KeyError:
					pass
			if used:
				continue

			max_val = 0
			for loci,(location, _id) in enumerate(events):
				time_arr = np.array(flash_crowd_data[location][0])
				load_arr = np.array(flash_crowd_data[location][1])
				index_of_interest = np.argmin(np.abs(time_arr-t))
				indices_of_interest = np.array(list(range(index_of_interest-100,index_of_interest+100)))
				load_arr = smooth_ewma(load_arr)
				max_val = np.maximum(max_val, np.max(load_arr[indices_of_interest]))
			for loci,(location, _id) in enumerate(events):
				time_arr = np.array(flash_crowd_data[location][0])
				load_arr = np.array(flash_crowd_data[location][1])

				index_of_interest = np.argmin(np.abs(time_arr-t))
				print("{} {} {}".format(location, index_of_interest, time_arr[index_of_interest]))
				indices_of_interest = np.array(list(range(index_of_interest-100,index_of_interest+100)))

				load_arr = smooth_ewma(load_arr)
				tplot = (time_arr[indices_of_interest]-np.min(time_arr[indices_of_interest])) / 3600

				lplot = load_arr[indices_of_interest] / max_val
				ax[n_plotted].plot(tplot, lplot, label=location, color=colors[loci%len(colors)])
				ax[n_plotted].axvline((time_arr[index_of_interest]-np.min(time_arr[indices_of_interest])) / 3600,0,1.0,color=colors[loci%len(colors)])

				used_fc_ids[location,_id] = None
			
			ax[n_plotted].set_xlabel("Time (h)")
			ax[n_plotted].set_ylabel("Normalized Load")
			ax[n_plotted].legend()

			n_plotted += 1
			if n_plotted == n_to_plot: 
				break
		plt.savefig('figures/flashcrowd_investigation.pdf')




	crowds_intensities = []
	for location in tqdm.tqdm(locations, desc="Finding intense flash crowds"):
		if location not in locations_of_interest: continue
		time_arr = flash_crowd_data[location][0]
		load_arr = flash_crowd_data[location][1]
		crowds_arr = flash_crowd_data[location][2]
	
		# print("Location: {}, crowds: {}".format(location, len(crowds_arr)))
		sorted_crowds = list(sorted(crowds_arr, key = lambda el : el[0]))
		for i,(crowd_start,crowd_stop,intensity) in enumerate(sorted_crowds):
			## t is an index
			crowds_intensities.append((intensity,location,i))
	ranked_fcs = sorted(crowds_intensities, key = lambda el : -1 * el[0])
	
	n_to_plot,n_plotted = 50,0
	f,ax = plt.subplots(n_to_plot,1)
	f.set_size_inches(7,4*n_to_plot)
	colors = ['red','blue','black','salmon','maroon', 'tan', 'coral','lawngreen', 'grey', 'peru','darkgoldenrod','violet','crimson']
	for plti,(intensity,location,i) in enumerate(ranked_fcs):
		crowd_start,crowd_stop,_ = flash_crowd_data[location][2][i]
		time_arr = np.array(flash_crowd_data[location][0])
		load_arr = np.array(flash_crowd_data[location][1])

		index_of_interest_start = crowd_start
		index_of_interest_stop = crowd_stop
		indices_of_interest = np.array(list(range(index_of_interest_start-100,index_of_interest_stop+100)))

		load_arr = smooth_ewma(load_arr)
		tplot = (time_arr[indices_of_interest]-np.min(time_arr[indices_of_interest])) / 3600
		lplot = load_arr[indices_of_interest] / np.max(load_arr[indices_of_interest])

		ax[n_plotted].plot(tplot, lplot, label=location, color=colors[0])
		ax[n_plotted].axvline((time_arr[index_of_interest_start]-np.min(time_arr[indices_of_interest])) / 3600,0,1.0,color=colors[0])

		ax[n_plotted].set_xlabel("Time (h)")
		ax[n_plotted].set_ylabel("Intensity: {} Normalized Load".format(round(intensity,2)))
		ax[n_plotted].legend()

		n_plotted += 1
		if n_plotted == n_to_plot: 
			break
	plt.savefig('figures/flashcrowd_investigation_by_intensity.pdf')



if __name__ == "__main__":
	# fetch_files()
	# parse_and_plot()
	# paper_plots()
	ilgar_flashcrowd_analysis()




