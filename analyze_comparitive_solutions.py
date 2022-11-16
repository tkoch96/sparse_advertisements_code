import numpy as np,pickle
import matplotlib.pyplot as plt
from helpers import *
metrics = pickle.load(open('cache/method_comparison_metrics.pkl','rb'))
of_interest = 1
to_show = ['painter','anyopt','maximal']
labs = [ "PAINTER","Anyopt",'Anycast']
### Difference between objective function of sparse and the other method
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams.update({'font.size': 22}) 
for i, k in enumerate(to_show):
	v = 1
	try:
		x,cdf_x = get_cdf_xy(metrics['objective_diffs'][k])
		plt.plot(x,cdf_x,label=labs[i])
	except IndexError:
		continue
plt.ylim([0,1.0])
plt.legend()
plt.grid(True)
plt.xlabel("Objective Difference (Prior Work - New Framework)")
plt.ylabel("CDF of Random Deployments")
save_fig("for_dissertation_proposal_objective.pdf")

### Difference between objective function of sparse and the other method
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams.update({'font.size': 22}) 
for i, k in enumerate(to_show):
	v = 1
	try:
		x,cdf_x = get_cdf_xy(metrics['latency_benefit_diffs'][k])
		plt.plot(x,cdf_x,label=labs[i])
	except IndexError:
		continue
plt.ylim([0,1.0])
plt.legend()
plt.grid(True)
plt.xlabel("Mean Latency Difference (Prior Work - New Framework) (ms)")
plt.ylabel("CDF of Random Deployments")
save_fig("for_dissertation_proposal_latency.pdf")