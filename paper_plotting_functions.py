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
	if 'lagrange' not in fn and 'penalty' not in fn: ## Spammy, not using them
		plt.savefig(os.path.join('figures', 'paper', fn), bbox_inches='tight')
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
solution_to_marker = {
	'sparse': '*',
	'painter': 'o',
	'anyopt': '>',
	'anycast': 'D',
	'one_per_pop': '+',
	'one_per_peering': '_',
}

markers = sorted(list(solution_to_marker.values()))