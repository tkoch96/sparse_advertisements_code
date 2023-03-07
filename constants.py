CACHE_DIR = "cache"
GRAPH_DIR = "graphs"
KM_TO_MS = .01
cols = ['firebrick','salmon','orangered','lightsalmon','sienna','lawngreen','darkseagreen','palegoldenrod',
	'darkslategray','deeppink','crimson','mediumpurple','khaki','dodgerblue','lime','black','midnightblue',
	'lightsteelblue']

MIN_LATENCY = 1
MAX_LATENCY = 40
NO_ROUTE_LATENCY = MAX_LATENCY
NO_ROUTE_BENEFIT = -1 * NO_ROUTE_LATENCY
LBX_DENSITY = 100 # number of points to discretize benefit into

BASE_SOCKET = 31415

ADVERTISEMENT_THRESHOLD = .5
import numpy as np
def threshold_a(a):
	return (a > ADVERTISEMENT_THRESHOLD).astype(np.float32)

DPSIZE = 'small'
PRINT_FREQUENCY = {
	'really_friggin_small': 50,
	'small': 5,
	'decent': 1,
	'med': 1,
	'large': 1
}[DPSIZE]

N_WORKERS = {
	'really_friggin_small': 1,
	'decent': 4,
	'med': 1,
	'small': 4,
}.get(DPSIZE, 8)



DEFAULT_EXPLORE = 'bimodality'