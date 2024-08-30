import pickle

call_args = pickle.load(open('tmp.pkl','rb'))

d1 = call_args[0][1]
d2 = call_args[1][1]


d1_vol = d1['ug_to_vol']
d2_vol = d2['ug_to_vol']

for ug in d1_vol:
	if d1_vol[ug] != d2_vol[ug]:
		print("UG {} {} vs {}".format(ug,d1_vol[ug], d2_vol[ug]))


all_rets = pickle.load(open('all_rets.pkl','rb'))
ret0 = all_rets[0]['lats_by_ug']
ret1 = all_rets[5]['lats_by_ug']
print(ret0-ret1)


















# import pickle
# from helpers import *

# working = pickle.load(open('saving_working_sparse_deployment.pkl','rb'))
# notworking = pickle.load(open('saving_notworking_sparse_deployment.pkl','rb'))

# for k in working:
# 	eq = working[k] == notworking[k]
# 	if not eq:
# 		print("{} not equal".format(k))
# print(get_difference(working['popps'],notworking['popps']))
# print(get_difference(notworking['popps'],working['popps']))

# print(get_difference(working['ugs'],notworking['ugs']))
# print(get_difference(notworking['ugs'],working['ugs']))
# ### problem is this fucker is being removed in the working case, but not in the post-working case
# ug_of_interest = ('vtrnewyork',459)
# print(notworking['ug_perfs'][ug_of_interest])

# print(notworking['ug_to_ip'][ug_of_interest])


# pseudo = pickle.load(open('runs/1713752460-actual_second_prototype-sparse/state-0.pkl','rb'))['deployment']


# converted_ugs = list([(ug[0],int(ug[1])) for ug in pseudo['ugs']])
# og_to_pseudo = {}
# for ug in pseudo['ugs']:
# 	try:
# 		og_to_pseudo[ug[0],int(ug[1])].append(ug)
# 	except KeyError:
# 		og_to_pseudo[ug[0],int(ug[1])] = [ug]
# print(ug_of_interest in converted_ugs)
# for pseudo_ug in og_to_pseudo[ug_of_interest]:
# 	ips = pseudo['ug_to_ip'][pseudo_ug]
# 	# for ug,_ips in pseudo['ug_to_ip'].items():
# 	# 	if ug == pseudo_ug: continue
# 	# 	if len(get_intersection(_ips,ips)) > 0:
# 	# 		print("OISHDFIOHSDFIO")



# ## problem is we're removing it during training for some reason
# ## but we're not removing it while loading all the files, for some other reason
# ## I think its because I fixed siblings halfway through