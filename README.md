# Sparse Advertisements Code

## Key commands

To, in principle, generate most of the plots in the paper, you would just need to do

"python evaluate_over_deployment_sizes.py --port 31415" # conduct emulated experiments

"python actual_deploymenteval_lateny_failure && python make_actual_deployment_plots.py" # conduct real-deployment experiments

In practice, each of these scripts usually needs babysitting and they also take several days to run. So, for a given change or experiment, you want to call subsets of that analysis with small deployments and then scale your way up to the largest sizes. The key tool for doing so is:

"python eval_latency_failure.py --port 31415 --dpsize small" # completely simulated deployment, fake latencies

or, if you want to use real latencies
"python eval_latency_failure.py --port 31415 --dpsize actual-10" # actual 10 means use real latencies, emulated deployment, for 10 random sites

Most of the scripts must be run on a specific port. The port is the port that the leader script communicates on to direct the worker scripts. Calling two different evaluations at the same time on the same port will confuse the workers and likely ruin all the simulations. Hence, it's very important that, if you're running many evaluations at once, you use *different* ports. 


## Setup
These instructions are a WIP

0. Get an environment. To run the largest sizes in a day, it generally helps to have 50+ CPUs. I think I used a m7g.8xlarge. Make sure to only use this size when you actively need it, as it is really expensive. (There might be even better generations of VM these days, just use the best one.) I would use a python virtual environment for python package management.
1. Install all the python packages. It would help if folks could contribute to the requirements.txt file to enable quick setup.
2. Get the data from the drive. You might want to consider getting your own data as, over time, this data will grow stale. I will try to upload everything that's needed [here](https://drive.google.com/drive/folders/1PvGOPRgkvjTaeq5m2ogyh0zSZ4r6JLcJ)
3. Get a gurobi license and configure it on your machine. I think it involves putting the license file in your home directory, but I'm not sure. It's a WSL student license: https://www.gurobi.com/academia/academic-program-and-licenses/


## Key directories

### data
Holds "data" which is anything I pull from somewhere and is generally external from the codebase. Things like AS relationships, latency measurements, lists of IP addresses.

### figures
Holds all figures for evaluation, etc

### graphs
unused

### old_scripts
unused scripts but keeping just in case

### cache
Holds data generated during 

### runs
Holds state that is meant to 
1. give useful debugging information
2. give us metrics that allow us to conduct analysis in the future without needing to rerun the training
3. restart a run from partway through the learning process if it got stopped or ran into an error
4. hold measurements from when we actually deploy things on the Internet

### logs
Holds logs from workers threads, debugging.

### .
The main directory holds all the scripts we use in no logically organized way :)

## Key types of scripts

### Solely Plotting Scripts
A small number of scripts just plot clean results for the paper, assuming all the evaluations have been done and that those results have been stored somewhere.

*make_actual_deployment_plots.py*: Makes plots for the paper related to the actual deployment on the Internet.


### Evaluations
These call the algorithms, collect metrics, call the evaluators, and plot the results. They are used to generate most/all plots in the paper. Evaluations often fall into the following run-flow

1. Collect settings pertaining to the evaluation such as the size of the deployment we want to emulate.
2. Run the advertisement allocation algorithms, possibly several times across many emulated deployments for statistical significance. Many evaluation scripts support starting from where a prior run stopped in case the process accidentally finish early.
3. Collect and cache the output from the runs (mostly the advertisement solutions).
4. Conduct the evaluations: there are usually several different evaluations across several different settings and potentially many different "runs". 
5. Plot the results, usually across runs/settings.

These evaluation scripts are of two types. There are scripts that call the evaluations, and then scripts that actually implement the evaluation. For example, I can evaluate how a solution handles flash crowds by (a) writing a script that conducts flash crowd analysis and (b) writing a script that invokes that analysis over many deployments

##### Evaluation Implementations
*wrapper_eval.py*: Implements most if not all of the actual evaluations like assessing resilience to flash crowds.


##### Evaluation Callers
*eval_latency_failure.py*: What I usually use to conduct the key evaluations. It was built to emulate a bunch of random deployments, solve each of the various advertisement solutions on these deployments, and evaluate how they did on average over all the randomly emulated deployments.

*eval_modeling_assumptions.py*: Never got it working correctly, but the intention was to assess resilience to various assumptions we make in the paper. For example, "change_model_capacities" is meant to assess how resilient we would be to changing the capacity on one or more paths. For example, if the bottleneck capacity was somewhere else along the path. Sharad's/Ethan's impression was that this analysis was not important and that we could explain-away reviewer concerns.

*just_prior.py*: calls key evaluations for everything except SCULPTOR

*specific_deployment.py*: calls key evaluations for a specific emulated deployment, rather than randomly generating a new deployment or using whatever the default deployment was

*evaluate_over_deployment_sizes.py*: Evaluates/plots how SCULPTOR compares to other methods as we vary the deployment size. Used to generate a lot of plots in the paper.

*evaluate_over_n_prefixes.py*: Evaluates/plots how SCULPTOR compares to other methods as we vary the number of prefixes we use. The key finding was that SCULPTOR does about as well as other methods until the number of prefixes exceeds the number of sites.

*testing_feature.py* (clunky) Carbon copy of evaluate_over_deployment_sizes.py except we're toggling some arbitrary feature of the way we do things, to see if SCULPTOR does any better compared to other methods. Also used for a lot of plots in the paper.

*testing_generic_objective.py*: Same as eval_latency_failure except calls with an objective other than average latency. Used more for testing implementation that feautre iirc, so not really used anymore.

*testing_priorities.py*: Tests multipriority traffic for the B4/SWAN evaluation in the paper. Could potentially be improved/explored further but reviewers didn't seem to have a problem with it.

### Algorithms
Contain the implementations of each of the key algorithms we're comparing. "sparse_advertisements_v3.py" contains SCULPTOR, "painter.py" contains unicast and PAINTER, and "anyopt.py" contains AnyOpt.

Also here is:

*optimal_adv_wrapper.py* which wraps each of the solutions and contains a lot of generic helpful scripts like learning from measurements and so on.

*path_distribution_computer.py*: In SCULPTOR, we have to evaluate the objective lots of times whether this be for exploration or gradient descent. This script/class is a worker bee for computing that objective. 

*realworld_measure_wrapper.py*: For actual deployments on the Internet, we need to conduct advertisements. This class interfaces with another package to conduct those advertisements, caches results, invokes RIPE Atlas traceroutes, parses measurements, etc. 

*solve_lp_assignment.py*: Interfaces with Gurobi to solve traffic allocations. I.e., for eah given objective specified by "generic_objective.py", this file implements a way of calling Gurobi with the appropriate constraints etc... that map user traffic onto available routes so as to optimize that objective. 



### Generic Helper/Config Scripts
This is what most of the scripts here fall under, and each has their own purpose. Highlighting the key ones here.

*constants.py*: contains things that we expect to be held constant 

*deployment_setup.py* : Emulates deployments, routing preferences, loads latencies, etc.


*helpers.py*: Miscellaneous functions. Any generic utilities that I'd use here or across other projects.

*kilitall.py*: We run evaluations over sometimes 50+ workers, each of which is an OS process. If something bad happens, sometimes there can be tons of processes which are annoying to kill. Sometimes you may have multiple simulations running, so you don't just want to kill absolutely all the processes. This script kills the processes solely associated with one simulation, UIDed by the port that the main process is listening on.

*test_polyphase.py* (tech debt -- need to remove) Used to use this to compute the objective but have since moved to monte-carlo methods.

*worker_comms.py*: Starts up / tears down sockets/processes related to communicating with worker bees.


### Specific/one-off analysis for the paper

*count_solutions.py*: counts the representativeness of the emulations in terms of # of IPs, /24s,/ ASes 

*get_smaller_anycast_lats.py*: helps get a small subset of the latency measurements to work with. better to work with a small set of measurements for testing new code since it will just be faster.

*graph_utils.py*: not perfectly adopted everywhere but an attempt at standardization of figure plotting settings so that we don't have to worry about dimensions, font sizes, etc.

*paper_plotting_functions.py*: Also an attempt to standardize how we plot things.

*weathermap_investigation.py*: OVH cloud motivation.



