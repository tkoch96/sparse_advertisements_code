import numpy as np
from sklearn.mixture import GaussianMixture

## want to know about cases where, compared to where we currently are, there's significant
## chance you could go wildly in either direction
## so kind of like bimodality about the current benefit
## more weight should be given when the peaks are further from the current benefit, indicating
## more extreme relative outcomes


## so best solution should be probability evenly split, peaks at extreme bins
## worst solution should be unimodal at the current benefit

## maybe like joint max of |negative pdf * (x_negative-x)| vs |positive pdf * (x_positive-x)|
## so could do product of the two

def bimodality_metric(benefits,probs,current_benefit):
	ex = np.average(benefits,weights=probs)
	exsq = np.average(np.power(benefits,2),weights=probs)
	var = exsq - np.power(ex,2)
	std = np.sqrt(var)
	skew = np.average(np.power((benefits - ex) / std, 3), weights = probs)
	kurt = np.average(np.power((benefits - ex) / std , 4), weights = probs)
	v = kurt - np.power(skew,2)
	# maximizing v is maximizing bimodality
	return -1 * v

def other_bimodality_metric(benefits,probs,current_benefit):
	### sure, but we dont know the current benefit, so this is bad
	negative_part = np.where(benefits <= current_benefit)
	positive_part = np.where(benefits > current_benefit)
	positive_mass = np.sum(probs[positive_part] * (benefits[positive_part] - current_benefit))
	negative_mass = np.sum(probs[negative_part] * (current_benefit - benefits[negative_part]))
	print("{} {}".format(positive_mass,negative_mass))
	v = positive_mass * negative_mass
	return v

def fitting_gmm(benefits, probs, currrent_benefit):
	### idea is maximize distance between bimodal peaks
	### we find bimodal peaks by fitting gmm
	probs = np.array(probs)
	x_samp = np.random.choice(benefits,size=(1000,1),p=probs)
	gm_means = GaussianMixture(n_components=2).fit(x_samp).means_
	return np.abs(np.diff(gm_means.flatten()))[0]


current_benefit = 33
benefits = np.linspace(0,40)
n = len(benefits)
uniform_probs = 1/n * np.ones(benefits.shape)

m1 = bimodality_metric(benefits,uniform_probs,current_benefit)
m2 = other_bimodality_metric(benefits, uniform_probs,current_benefit)
m3 = fitting_gmm(benefits, uniform_probs,current_benefit)
print("paper:{} not great:{} gmm:{}\n".format(m1,m2,m3))

unimodal_probs = np.exp(-1 * np.power(benefits-20,2))
unimodal_probs = unimodal_probs/np.sum(unimodal_probs)
m1 = bimodality_metric(benefits,unimodal_probs,current_benefit)
m2 = other_bimodality_metric(benefits, unimodal_probs,current_benefit)
m3 = fitting_gmm(benefits, unimodal_probs,current_benefit)
print("paper:{} not great:{} gmm:{}\n".format(m1,m2,m3))


bimodal_probs = np.exp(-1 * np.power(benefits-5,2)) + np.exp(-1 * np.power(benefits-35,2))
bimodal_probs = bimodal_probs/np.sum(bimodal_probs)
m1 = bimodality_metric(benefits,bimodal_probs,current_benefit)
m2 = other_bimodality_metric(benefits, bimodal_probs,current_benefit)
m3 = fitting_gmm(benefits, bimodal_probs,current_benefit)
print("paper:{} not great:{} gmm:{}\n".format(m1,m2,m3))

