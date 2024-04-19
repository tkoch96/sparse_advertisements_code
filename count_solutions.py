import pickle, os

for i in range(32):
	fn = 'cache/popp_failure_latency_comparison_actual-{}.pkl'.format(i+1)
	if not os.path.exists(fn): continue

	metrics = pickle.load(open(fn, 'rb'))

	has_sims = {}
	for random_iter in range(1000):
		try:
			try:
				metrics['compare_rets'][random_iter]['n_advs']
				has_sims[random_iter] = None
			except TypeError:
				pass
		except KeyError:
			pass


	print("{} -- {}".format(fn,sorted(list(has_sims))))