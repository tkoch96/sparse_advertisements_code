CACHE_DIR = "cache"
DATA_DIR = "data"
GRAPH_DIR = "graphs"
FIG_DIR = 'figures'
RUN_DIR = 'runs'
KM_TO_MS = .01
LOG_DIR = 'logs'
cols = ['firebrick','salmon','orangered','lightsalmon','sienna','lawngreen','darkseagreen','palegoldenrod',
	'darkslategray','deeppink','crimson','mediumpurple','khaki','dodgerblue','lime','black','midnightblue',
	'lightsteelblue']


LIMITED_CAP_LATENCY_MULTIPLIER = 1.5
GRAD_CLIP_VAL = 10

LBX_DENSITY = 50

N_PREFIXES = 10

ADVERTISEMENT_THRESHOLD = .5
import numpy as np
def threshold_a(a):
	return (a > ADVERTISEMENT_THRESHOLD).astype(np.float32)



UGS_OF_INTEREST = []
# UGS_TO_DELETE = [('vtrnewyork', 459)]
UGS_TO_DELETE = []

DEFAULT_PORT = 31600


## to identify when we're looking at an actual deployment
ACTUAL_DEPLOYMENT_SIZES = ["actual_first_prototype", "actual_second_prototype", "actual_third_prototype"]
DEBUG_CLIENT_INFO_ADDING = False


CONSIDERING_POPS_ACTUAL_DEPLOYMENT = {
	"actual_first_prototype": ['vtrnewyork', 'vtratlanta', 'vtrmiami'],
	"actual_second_prototype": ['vtrnewyork', 'vtratlanta', 'vtrmiami', 'vtrparis', 'vtrlondon'],
	"actual_third_prototype": ['vtrnewyork', 'vtratlanta', 'vtrmiami', 'vtrparis', 'vtrlondon', 'vtrsaopaulo', 'vtrchicago', 'vtrdallas', 'vtrmadrid', 'vtrstockholm']
}

NON_SIMULATED_LINK_CAPACITY = 100000

N_POPS_ACTUAL_DEPLOYMENT = 10

MIN_LATENCY = 1
MAX_LATENCY = 300
NO_ROUTE_LATENCY = 1.5*MAX_LATENCY
NO_ROUTE_BENEFIT = -1 * NO_ROUTE_LATENCY

import re
def n_pops_from_dpsize(deployment_size):
	if deployment_size in ACTUAL_DEPLOYMENT_SIZES:
		return len(CONSIDERING_POPS_ACTUAL_DEPLOYMENT[deployment_size])
	elif 'actual' in deployment_size:
		return int(re.search('actual\-(.+)',deployment_size).group(1))
	elif deployment_size == 'small':
		return 2

def PRINT_FREQUENCY(dpsize):
	### How often we make plots, often slow to create
	if dpsize in ACTUAL_DEPLOYMENT_SIZES:
		return 2
	dpsize_pops = n_pops_from_dpsize(dpsize)
	if dpsize == 'small':
		return 3#10
	elif dpsize_pops <= 5:
		return 50
	elif dpsize_pops <= 15:
		return 30
	else:
		return 15


def get_n_workers(deployment_size):
	n_workers = {
		'really_friggin_small': 1,
		'actual': 4,
		'actual-small': 4,
		'actual-large': 28,
		'actual_first_prototype': 2,
		'actual_second_prototype': 2,
		'actual_third_prototype': 2,
		'small': 4,
		'decent': 8,
		'med': 1,
	}.get(deployment_size)
	if n_workers is None:
		n_pops = n_pops_from_dpsize(deployment_size)
		if n_pops <= 5:
			n_workers = 3
		elif n_pops < 15:
			n_workers = 8
		elif n_pops < 20:
			n_workers = 16
		else:
			n_workers = 24

	return n_workers

RESILIENCE_DIFFICULTY = 'hard'


#### SOLVER SETTINGS
## number of threads to allocate to one Gurobi solver
N_WORKERS_GENERIC = 1 
## minimuze MLU + ALPHA * LATENCY ;; so alpha is a tradeoff between congestion and latency (roughly)
ALPHA = .1
## minimize average latency + ALPHA_BULK * bulk_traffic_overuse
ALPHA_BULK = 100.0
## multiplier over regular volume for bulk volume
BULK_MULTIPLIER = 2.0
BULK_CAP_LIMIT = 10.0


DEFAULT_EXPLORE = 'entropy'


POP_TO_LOC = {
	'peering':{
		'amsterdam01': (52.359,4.933),
	}, 'vultr': {
		'vtramsterdam': (52.359,4.933),
		'vtratlanta': (33.749, -84.388),
		'vtrbangalore': (12.940, 77.782),
		'vtrchicago': (41.803,-87.710),
		'vtrdallas': (32.831,-96.641),
		'vtrdelhi': (28.674,77.099),
		'vtrfrankfurt': (50.074, 8.643),
		'vtrhonolulu': (21.354, -157.854),
		'vtrjohannesburg': (-26.181, 27.993),
		'vtrlondon' : (51.452,-.110),
		'vtrlosangelas': (34.165,-118.489),
		'vtrmadrid': (40.396,-3.678),
	 	'vtrmanchester': (53.48,-2.265),
		'vtrmelbourne': (-37.858, 145.028),
		'vtrmexico': (19.388, -99.138),
		'vtrmiami' : (25.786, -80.229),
		'vtrmumbai' : (19.101, 72.869),
		'vtrnewyork': (40.802,-73.970),
	 	'vtrosaka': (34.677,135.48),
	 	'vtrsantiago': (-33.487, -70.683),
		'vtrparis': (48.836,2.308),
		'vtrsaopaulo' : (-23.561, -46.532),
		'vtrseattle': (47.577, -122.373),
		'vtrseoul': (37.683,126.942),
		'vtrsilicon': (37.312,-121.816),
		'vtrsingapore': (1.322,103.962),
		'vtrstockholm': (59.365,17.943),
		'vtrsydney': (-33.858,151.068),
	 	'vtrtelaviv': (32.086,34.782),
		'vtrtokyo': (35.650,139.619),
		'vtrtoronto': (43.679, -79.305),
		'vtrwarsaw': (52.248,21.027),
	},
}

POP2TIMEZONE = {  # GMT
	'vtramsterdam': 2,
	'vtratlanta': -4,
	'vtrbangalore': 5.5,
	'vtrchicago': -5,
	'vtrdallas': -5,
	'vtrdelhi': 5.5,
	'vtrfrankfurt': 2,
	'vtrjohannesburg': 2,
	'vtrlondon': 1,
	'vtrlosangelas': -7,
	'vtrmadrid': 2,
	'vtrmelbourne': 10,
	'vtrmexico': -6,
	'vtrmiami': -4,
	'vtrmumbai': 5.5,
	'vtrnewyork': -4,
	'vtrparis': 2,
	'vtrsaopaulo': -3,
	'vtrseattle': -7,
	'vtrseoul': 9,
	'vtrsilicon': -7,
	'vtrsingapore': 8,
	'vtrstockholm': 2,
	'vtrsydney': 10,
	'vtrtokyo': 9,
	'vtrtoronto': -4,
	'vtrwarsaw': 2,
    'vtrosaka': 9
}

