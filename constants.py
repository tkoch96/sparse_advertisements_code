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

ADVERTISEMENT_THRESHOLD = .5
import numpy as np
def threshold_a(a):
	return (a > ADVERTISEMENT_THRESHOLD).astype(np.float32)