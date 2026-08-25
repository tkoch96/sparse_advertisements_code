"""Paper-quality plot styling primitives.

Sets up matplotlib rcParams for paper-grade PDFs (TrueType fonts, proper
sizes, etc.), and provides helpers used across the various
`make_*_plots.py` files. Imported, not run directly.
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os

def get_figure(l=7,h=3):
	plt.clf()
	plt.close()

	font = {'size'   : 14}
	matplotlib.rc('font', **font)
	f,ax = plt.subplots(1)
	f.set_size_inches(l,h)
	return f,ax

def save_figure(fn):
	if 'penalty' not in fn: ## Spammy, not using them
		# SCULPTOR_FIG_SUBDIR namespaces a run's figures (the dpsweep
		# driver sets it from --figures-subdir); this helper ignoring it
		# meant make_paper_plots always wrote figures/paper/ and the
		# dash's per-run copies never updated (Tom 2026-08-25)
		_sub = os.environ.get('SCULPTOR_FIG_SUBDIR')
		_dir = os.path.join('figures', _sub) if _sub else \
			os.path.join('figures', 'paper')
		os.makedirs(_dir, exist_ok=True)
		plt.savefig(os.path.join(_dir, fn), bbox_inches='tight')
	plt.clf()
	plt.close()

solution_to_plot_label = {
	'sparse': 'SCULPTOR',
	'painter': 'PAINTER',
	'anyopt': 'AnyOpt',
	'anycast': 'Anycast',
	'one_per_pop': 'Unicast',
	'one_per_peering': 'One per Peering'
}

solution_to_line_color = {
	'sparse': 'magenta',
	'painter': 'black',
	'anyopt': 'orange',
	'anycast': 'midnightblue',
	'one_per_pop': 'red', 
	'one_per_peering': 'lawngreen',
}

some_colors = sorted(list(solution_to_line_color.values()))

solution_to_marker = {
	'sparse': '*',
	'painter': 'o',
	'anyopt': '>',
	'anycast': 'D',
	'one_per_pop': '+',
	'one_per_peering': '_',
}

markers = sorted(list(solution_to_marker.values()))