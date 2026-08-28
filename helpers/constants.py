"""Project-wide constants and the dpsize -> n_pops mapping.

Imported with `from constants import *` in most modules — anything added
here is global. Keep additions to truly constant config; per-experiment
knobs belong in env vars (see README.md "Environment variables").

Notable values:
  - NO_ROUTE_LATENCY        sentinel latency for users with no feasible
                            path. Also used as a numeric value in LP
                            objective contributions (so changes here
                            affect both the sentinel-check semantics
                            AND avg-latency arithmetic).
  - NON_SIMULATED_LINK_CAPACITY  capacity used for actual-deployment
                                  popps where we don't have a measured
                                  cap
  - `dpsize_to_n_pops`       mapping from named dpsize strings ('small',
                             'decent', 'med', 'large',
                             'really_friggin_small') to PoP counts
"""
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

# whether to cluster UGs based on their latency to PoPPs
DO_UG_CLUSTERING = False
# whether or not to use APNIC volume (false means use simulated volume)
APNIC_VOLUME = False

LIMITED_CAP_LATENCY_MULTIPLIER = 1.5
GRAD_CLIP_VAL = 10

LBX_DENSITY = 50

N_PREFIXES = 10

ADVERTISEMENT_THRESHOLD = .5
import numpy as np
import os as _os
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
# SCULPTOR_NO_ROUTE_LATENCY overrides the no-route sentinel (default
# 100*MAX_LATENCY = 30000ms). The hard 30000ms charge gives failure-scenario
# LPs a ~1000x scale mismatch vs steady latencies, which destabilizes
# resilience-gradient training at gamma >~ 0.3; a softer value (e.g. 1000ms
# -- still >3x any real path) tames the gradient. Set it for TRAINING
# processes only: trusted rescoring/eval must keep the canonical default so
# reported numbers stay comparable.
NO_ROUTE_LATENCY = int(float(_os.environ.get('SCULPTOR_NO_ROUTE_LATENCY', 100*MAX_LATENCY)))
# Penalty PRICES vs the sentinel MARKER (Tom 2026-08-28): NO_ROUTE_LATENCY
# stays a huge value used only to MARK unroutable entries in lats arrays
# (failure/flash evals detect on it). Objective SCALARS and gradient-LP
# path prices must never use the marker -- they price bad volume at these
# bounded, gradient-friendly latencies instead: a stranded user costs
# NO_ROUTE_PENALTY_MS, a congested user about half that. High enough to
# dominate any real path (~10-250ms), low enough to keep gradients on the
# same scale as latency (the 30000 marker made every infeasible/stranded
# state a flat cliff -- zero gradient signal, see the maxhard joint
# stuck-at-iter-2 forensics 2026-08-28).
NO_ROUTE_PENALTY_MS = float(_os.environ.get('SCULPTOR_NO_ROUTE_PENALTY_MS', '800'))
CONGESTED_PENALTY_MS = float(_os.environ.get('SCULPTOR_CONGESTED_PENALTY_MS', '350'))
NO_ROUTE_BENEFIT = -1 * NO_ROUTE_LATENCY

import re

# Maps synthetic (non-actual) deployment-size names to their n_pop counts.
# Kept in sync with deployment_setup.problem_params. ACTUAL_DEPLOYMENT_SIZES
# and `actual-N` forms are still handled separately in n_pops_from_dpsize.
_SYNTHETIC_DPSIZE_N_POPS = {
	'really_friggin_small': 2,
	'small': 3,
	'decent': 10,
	'med': 30,
	'large': 100,
}

def n_pops_from_dpsize(deployment_size):
	if deployment_size in ACTUAL_DEPLOYMENT_SIZES:
		return len(CONSIDERING_POPS_ACTUAL_DEPLOYMENT[deployment_size])
	elif 'actual' in deployment_size:
		return int(re.search('actual\-(.+)',deployment_size).group(1))
	elif deployment_size == 'small':
		# Preserve original return value (2) even though small actually has
		# 3 pops; downstream callers depend on this.
		return 2
	# Fall-through covers really_friggin_small / decent / med / large.
	return _SYNTHETIC_DPSIZE_N_POPS.get(deployment_size)

def PRINT_FREQUENCY(dpsize):
	### How often we make plots, often slow to create
	if dpsize in ACTUAL_DEPLOYMENT_SIZES:
		return 2
	dpsize_pops = n_pops_from_dpsize(dpsize)
	if dpsize == 'small':
		return 3
	# Defensive: an unknown size returns None from n_pops_from_dpsize. Fall
	# through to the common case (10) instead of crashing on None <= 5.
	if dpsize_pops is None:
		return 10
	elif dpsize_pops <= 5:
		return 10
	elif dpsize_pops <= 15:
		return 10
	else:
		return 10


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
			n_workers = 5
		else:
			n_workers = 1000 # cpu count

	return n_workers

RESILIENCE_DIFFICULTY = 'hard'


#### SOLVER SETTINGS
## number of threads to allocate to one Gurobi solver
N_WORKERS_GENERIC = 1 
## minimize MLU + ALPHA * LATENCY + DEFAULT_SITE_COST * site cost
DEFAULT_SITE_COST = 100.0
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

# ---------------------------------------------------------------------------
# Monte-Carlo draws of the joint routing distribution, per latency-benefit
# evaluation (path_distribution_computer._sample_scenario_realizations runs
# MC_NUM x solve_generic_lp_persistent). THIS IS THE ONLY DEFAULT -- before
# 2026-08-21 the value 5 was written in five places, two of which were
# unreachable and one of which (_solve_max_information) would broadcast its
# own default back onto the workers and silently undo the others.
#
# 1 is a single-draw noisy estimator. It is ~5x cheaper per job and NOT a
# cheaper way to compute the same number; see experiments/mc_ab/ for the
# paired A/B measuring what the noise costs.
DEFAULT_MC_NUM = 1
# The max-information (explore) phase deliberately uses MORE draws: it is
# choosing what to measure from the belief DISTRIBUTION, where single-draw
# noise is most damaging. Restored to DEFAULT_MC_NUM afterwards.
DEFAULT_MC_NUM_EXPLORE = 5

# ---------------------------------------------------------------------------
# WHEN-probing: measure-XOR-step under a TOTAL measurement budget.
# Merged from experiments/ablation/sculptor_fork.py (slotted/scheduled
# 2026-08-17; gated/smart 2026-08-21). Probing is grounding at the CURRENT
# advertisement; budget exhaustion stops MEASURING, never TRAINING.
#
# THE measurement budget for a solve() run. Every mode caps total
# path_measures growth at this. Override per-run with SCULPTOR_PROBE_N.
DEFAULT_PROBE_N = 10
# Assumed convergence horizon the budget is spread over (slot tiling and
# the smart gate's spacing targets both derive from it). Falls back to the
# run's max_n_iter when that is known.
DEFAULT_PROBE_TCONV = 100
# post_step = stock (measure after every step that moved the advertisement,
# no budget). scheduled/slotted/gated/smart are budgeted.
DEFAULT_PROBE_MODE = 'smart'
# smart-gate shape (see _probe_smart_decision). Defaults reproduce the
# ablation fork's validated values.
DEFAULT_PROBE_C = 1.0            # initial (high) uncertainty threshold
DEFAULT_PROBE_FRAC = 0.75        # budget spread over this fraction of TCONV
DEFAULT_PROBE_MINGAP_FRAC = 0.7  # self-assessed criteria held below this gap
DEFAULT_SCHED_FALLBACK_MULT = 1.25   # backstop spacing multiplier
DEFAULT_SMART_STALE_FRAC = 1.0       # (b) staleness gap, x TCONV/N
DEFAULT_SMART_PLATEAU_W = 5          # (b) plateau window
DEFAULT_SMART_PLATEAU_EPS = 0.01     # (b) plateau tolerance, x belief span
DEFAULT_SMART_SIGN_W = 6             # (c) predicted-vs-realized window
DEFAULT_SMART_SIGN_RATE = 0.5        # (c) sign-disagreement rate to fire
DEFAULT_SMART_SURPRISE_REL = 0.05    # (d) surprise threshold, x belief span
DEFAULT_SMART_SURPRISE_FACTOR = 0.5  # (d) c multiplier on a surprising probe
DEFAULT_U_ENT_W = 0.0                # weight of the adjacency-entropy term in U


def resolve_probe_budget(n_prefixes=None):
    """The measurement budget in force, resolving the 'prefixes' sentinel.

    SCULPTOR_PROBE_N may be an int or the literal 'prefixes' (each
    deployment's own prefix count, which varies with size). Both SCULPTOR
    and painter must resolve it identically or a "budget-fair" comparison
    silently is not; this is the single place that does it.

    Returns None when probing is unbudgeted (PROBE_MODE=post_step).
    """
    import os as _os
    mode = _os.environ.get('SCULPTOR_PROBE_MODE',
                           _os.environ.get('SCULPTOR_ABLATION_PROBE_MODE',
                                           DEFAULT_PROBE_MODE))
    if mode == 'post_step':
        return None
    raw = str(_os.environ.get('SCULPTOR_PROBE_N',
                              _os.environ.get('SCULPTOR_ABLATION_PROBE_N',
                                              DEFAULT_PROBE_N))).strip().lower()
    if raw in ('prefixes', 'n_prefixes', 'prefix'):
        n = int(n_prefixes or 0)
        return n if n > 0 else int(DEFAULT_PROBE_N)
    try:
        return int(raw)
    except ValueError:
        return int(DEFAULT_PROBE_N)
