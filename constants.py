CACHE_DIR = "cache"
DATA_DIR = "data"
GRAPH_DIR = "graphs"
KM_TO_MS = .01
cols = ['firebrick','salmon','orangered','lightsalmon','sienna','lawngreen','darkseagreen','palegoldenrod',
	'darkslategray','deeppink','crimson','mediumpurple','khaki','dodgerblue','lime','black','midnightblue',
	'lightsteelblue']


LIMITED_CAP_LATENCY_MULTIPLIER = 1.5
GRAD_CLIP_VAL = 10

LBX_DENSITY = 50

BASE_SOCKET = 31427

ADVERTISEMENT_THRESHOLD = .5
import numpy as np
def threshold_a(a):
	return (a > ADVERTISEMENT_THRESHOLD).astype(np.float32)

DPSIZE = 'actual'
PRINT_FREQUENCY = {
	'really_friggin_small': 20,
	'actual': 12,
	'small': 5,
	'decent': 1,
	'med': 1,
	'large': 1
}[DPSIZE]

if DPSIZE == "really_friggin_small":
	MIN_LATENCY = 1
	MAX_LATENCY = 20
else:
	MIN_LATENCY = 1
	MAX_LATENCY = 300
NO_ROUTE_LATENCY = 1.5*MAX_LATENCY
NO_ROUTE_BENEFIT = -1 * NO_ROUTE_LATENCY

N_WORKERS = {
	'really_friggin_small': 1,
	'actual': 2,
	'small': 2,
	'decent': 8,
	'med': 1,
}.get(DPSIZE, 8)
RESILIENCE_DIFFICULTY = 'hard'



DEFAULT_EXPLORE = 'entropy'


POP_TO_LOC = {
	'peering':{
		'amsterdam01': (52.359,4.933),
	}, 'vultr': {
		'amsterdam': (52.359,4.933),
		'atlanta': (33.749, -84.388),
		'bangalore': (12.940, 77.782),
		'chicago': (41.803,-87.710),
		'dallas': (32.831,-96.641),
		'delhi': (28.674,77.099),
		'frankfurt': (50.074, 8.643),
		'johannesburg': (-26.181, 27.993),
		'london' : (51.452,-.110),
		'losangelas': (34.165,-118.489),
		'madrid': (40.396,-3.678),
		'melbourne': (-37.858, 145.028),
		'mexico': (19.388, -99.138),
		'miami' : (25.786, -80.229),
		'mumbai' : (19.101, 72.869),
		'newyork': (40.802,-73.970),
		'paris': (48.836,2.308),
		'saopaulo' : (-23.561, -46.532),
		'seattle': (47.577, -122.373),
		'seoul': (37.683,126.942),
		'silicon': (37.312,-121.816),
		'singapore': (1.322,103.962),
		'stockholm': (59.365,17.943),
		'sydney': (-33.858,151.068),
		'tokyo': (35.650,139.619),
		'toronto': (43.679, -79.305),
		'warsaw': (52.248,21.027),
	},
}