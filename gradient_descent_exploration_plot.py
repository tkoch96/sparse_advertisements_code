import pickle, os, argparse, re, glob, numpy as np

from helpers import *

def interesting_gradients_over_iterations(run_dir):
	all_states = glob.glob(os.path.join(run_dir, 'state-*.pkl'))
	biggest_state_num = np.max(list([int(re.search('state\-(\d+)\.pkl', state_file).group(1)) for state_file in all_states]))

	biggest_state = pickle.load(open(os.path.join(run_dir, 'state-{}.pkl'.format(biggest_state_num)), 'rb'))
	metrics = biggest_state['metrics']

	all_as = np.array(metrics['advertisements'][0:])

	f, ax = plt.subplots(1)

	colors = ['orange','brown','aqua','deeppink','peru','grey','k','tan']
	for pref_i in range(all_as.shape[2]):
		for popp_i in range(all_as.shape[1]):
			this_adv_deltas = np.diff(all_as[:, popp_i, pref_i].flatten())
			if np.max(np.abs(this_adv_deltas)) > .03:
				ax.plot(all_as[:, popp_i, pref_i], c=colors[popp_i%len(colors)])
	ax.set_xlabel("Gradient Step")
	ax.set_ylabel("Advertisement Value")
	ax.set_ylim([0,1.0])

	save_fig('sample_advertisements_over_gradients.pdf')

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_run_dir", default=None, required=True)
	args = parser.parse_args()

	run_dir = os.path.join(RUN_DIR, args.save_run_dir)
	interesting_gradients_over_iterations(run_dir)


