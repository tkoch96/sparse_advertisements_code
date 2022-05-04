import numpy as np
from helpers import *
import matplotlib.pyplot as plt

for k in [1,5,10]:

	np.random.seed(31415)

	def sigmoid(arr,ax):
		return 1 / (1 + np.exp(-1 * k * (np.sum(arr,axis=ax) - .5)))


	N = 10
	errs = {'sum': [], 'prod': [], 'l1': []}
	vs = np.linspace(0,1,num=N)
	for i in range(N):
		for j in range(N):
			for l in range(N):
				for m in range(N):
					a = np.array([[vs[i],vs[j]],[vs[l],vs[m]]])
					n_peers = a.shape[0]
					n_prefs = a.shape[1]

					c_pref = 1

					c_peer = np.ones((n_peers,))
					cost_each_peer = c_peer * sigmoid(a,1)
					cost_peer = np.sum(cost_each_peer)

					cost_each_pref = c_pref * sigmoid(a,0)
					cost_pref = np.sum(cost_each_pref)


					c_sum = cost_peer + cost_pref

					# do product idea
					cost_peer = np.sum(1 - np.prod(1 - 1 / (1 + np.exp(-1 * k * (a - .5))),axis=1))
					cost_pref = np.sum(1 - np.prod(1 - 1 / (1 + np.exp(-1 * k * (a - .5))),axis=0))

					c_prod = cost_peer + cost_pref

					# True cost
					cost_peer, cost_pref = 0,0
					for i in range(n_peers):
						if np.sum(a[i,:]>= .5) >= 1:
							cost_peer += 1
					for j in range(n_prefs):
						if np.sum(a[:,j]>= .5) >= 1:
							cost_pref += 1

					c_gt = cost_peer+ cost_pref

					errs['sum'].append(np.power((c_sum - c_gt),2) )
					errs['prod'].append(np.power((c_prod - c_gt),2))
					errs['l1'].append(np.power( np.sum(np.abs(a).flatten()) - c_gt,2))



	for _k in errs:
		x,cdf_x = get_cdf_xy(errs[_k])
		plt.plot(x,cdf_x,label="{}-{}".format(_k, k))
plt.legend()
plt.xlabel("Err")
plt.show()

