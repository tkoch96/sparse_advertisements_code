import numpy as np
import gurobipy as gp
from helpers import *
import itertools


def main():

	# ## Infeasible
	constraint_specification = [
		(np.array([1., 1., 1., 1.]), [(1, 0.2265343056899113), (11, 0.6541057636797938), (18, 0.11935993063029487)]), 
		(np.array([0., 1., 1., 1.]), [                         (11, 0.6541057636797938), (18, 0.11935993063029487), (21, 0.2265343056899113)]),
		(np.array([1., 0., 1., 1.]), [(1, 0.09252043836037117),                          (18, 0.1272768311342983), (21, 0.7802027305053305)]), 
		(np.array([1., 1., 0., 1.]), [(1, 0.36974840876735166),(11, 0.6302515912326484)])
	]

	# constraint_specification = [
	#     (np.array([1., 1., 1., 1.]), [(1, 0.2265343056899113), (11, 0.6541057636797938), (18, 0.11935993063029487)]), 
	#     (np.array([0., 1., 1., 1.]), [                         (11, 0.6541057636797938), (18, 0.11935993063029487), (21, 0.2265343056899113)]),
	#     (np.array([1., 0., 1., 1.]), [(1, 0.09252043836037117),                          (18, 0.1272768311342983), (21, 0.7802027305053305)]), 
	#     (np.array([1., 1., 0., 1.]), [(1, 0.48910833939),(11, 0.6541057636797938)])
	# ]

	# constraint_specification = [(np.array([1., 1., 1.]), [(51, 0.5653871650642475), (168, 0.43461283493575265)]), 
	# 	(np.array([1., 0., 1.]), [(168, 0.6198301412234611), (45, 0.3801698587765388)]), 
	# 	(np.array([1., 1., 0.]), [(51, 0.7800589831496408), (45, 0.21994101685035916)])]

	# constraint_specification = [(np.array([1, 1]), [(421, 0.00047322598429870127), (417, 0.9995267740157013)])]

	# constraint_specification = [(np.array([1., 1., 1.]), [(11, 1.0)]), (np.array([1., 0., 1.]), [(1, 0.1975116715638386), (39, 0.8024883284361614)])]

	# constraint_specification = [(np.array([1., 1., 1.]), [(25, 1.0)]), (np.array([1., 1., 0.]), [(0, 0.8867263163608518), (1, 0.11327368363914826)])]

	constraint_specification = [(np.array([1., 1., 1.]), [(26, 0.18200428896916668), (6, 0.7384024248182287), (41, 0.07959328621260467)]), 
		(np.array([0., 1., 1.]), [(26, 0.5934701280792039), (41, 0.40652987192079615)]), (np.array([1., 0., 1.]), [(6, 1.0)]), (np.array([1., 1., 0.]), [(6, 1.0)])]

	### STill infeasible
	constraint_specification = [(np.array([1., 1., 1., 1.]), [(39, 0.4102623040530583), (63, 0.43244922967704086), (67, 0.15728846626990078)]), (np.array([1., 0., 1., 1.]), [(2, 1.0)]), 
		(np.array([1., 1., 0., 1.]), [(2, 0.8427115337300992), (67, 0.15728846626990078)]), (np.array([1., 1., 1., 0.]), [(39, 0.5675507703229591), (63, 0.43244922967704086)])]

	all_popps = sorted(set([constraint[0] for actives, constraint_set in constraint_specification for constraint in constraint_set]))
	n_popps = len(all_popps)
	n_scenarios = len(constraint_specification)
	print(all_popps)
	def solve_model(n_xi):
		print("Solving model with nxi = {}".format(n_xi))
		model = gp.Model("mip1")
		# model.Params.LogToConsole = 0
		x = model.addMVar(n_xi, name='volume_each_user', lb=0)
		model.addConstr(np.ones(n_xi) @ x == 1)
		

		all_sums = []
		all_vars = {}
		for k,(actives,constraint_set) in enumerate(constraint_specification):
			running_sum = np.zeros(n_xi)
			constraint_set = {popp:vol for popp,vol in constraint_set}
			all_vars[k] = {}
			for j,popp in enumerate(all_popps):
				vol = constraint_set.get(popp,0)
				a_jk = model.addMVar(n_xi, vtype=gp.GRB.BINARY, name="a_{}_{}".format(j,k))
				# model.addConstr((a_jk @ np.eye(n_xi) @ x) == vol)
				if vol > 0:
					obj = model.addMVar((1,), lb=-10000)
					model.addConstr(obj == ((a_jk @ np.eye(n_xi) @ x) - vol))
					tmp_obj_object = model.addMVar((1,), lb=0)
					model.addConstr(tmp_obj_object[0] == gp.norm(obj,2))
					all_sums.append(tmp_obj_object)
				else:
					model.addConstr((a_jk @ np.eye(n_xi) @ x) == vol)
				running_sum += a_jk
				all_vars[k][j] = a_jk
			model.addConstr(running_sum == np.ones(n_xi))

		## Add preference constraints
		popp_combs = list(itertools.combinations(list(range(n_popps)), 2))
		scenario_combs = list(itertools.combinations(list(range(n_scenarios)), 2))
		for popp_combi in popp_combs:
			if popp_combi[0] == popp_combi[1]: continue
			poppa,poppb = popp_combi
			for scenario_comb in scenario_combs:
				if scenario_comb[0] == scenario_comb[1]: continue
				k1,k2 = scenario_comb
				if constraint_specification[k1][0][poppa] == 1 and constraint_specification[k1][0][poppb] == 1 and \
					constraint_specification[k2][0][poppa] == 1 and constraint_specification[k2][0][poppb] == 1:
					# print("Adding constraint for k1: {} k2: {} poppa: {} ({}) poppb: {} ({})".format(k1,k2,poppa,all_popps[poppa],poppb,all_popps[poppb]))
					model.addConstr((all_vars[k1][poppa] - all_vars[k1][poppb]) * (all_vars[k2][poppa] - all_vars[k2][poppb]) >= np.zeros(n_xi))


		model.setObjective(gp.quicksum(all_sums))
		model.optimize()
		return model, all_vars, x

	n_xi = 8
	last_obj = 50000
	while True:
		model, all_vars, x = solve_model(n_xi)
		obj_value = model.getObjective().getValue()
		print("{}: Squared sum of errors: {}".format(n_xi,obj_value))
		if np.abs(obj_value - last_obj) < 0.01:
			n_xi -=1
			model, all_vars, x = solve_model(n_xi)
			break
		else:
			last_obj = obj_value
			n_xi += 1
	
	print("Squared sum of errors: {}".format(obj_value))

	routes_np = {k:np.zeros((n_popps,n_xi)) for k in range(n_scenarios)}
	try:
		print("Volumes: {}".format(x.X))
		user_vols = x.X
	except:
		pass
	for k in range(n_scenarios):
		print('\n')
		print(constraint_specification[k][0].astype(np.int32))
		print("----")
		for j in range(n_popps):
			try:
				print((all_vars[k][j].X>0).astype(np.int32))
				routes_np[k][j,:] = all_vars[k][j].X
			except:
				pass

	### Convert routes to preferences
	ui_popp_to_parents = {}
	base_actives = np.ones(n_popps) ## all on
	base_k = [k for k,(actives,constraint_set) in enumerate(constraint_specification) if np.array_equal(actives,base_actives)][0]
	base_mapping = all_vars[base_k]
	for k,(actives,constraint_set) in enumerate(constraint_specification):
		if np.array_equal(actives,base_actives): continue
		popps_on = list([all_popps[i] for i in np.where(actives)[0]])
		routes = routes_np[k]
		print(actives)
		for ui in range(n_xi):
			winning_popp = all_popps[np.where(routes[:,ui])[0][0]]
			for losing_popp in get_difference(popps_on, [winning_popp]):
				try:
					ui_popp_to_parents[ui,losing_popp].append(winning_popp)
				except KeyError:
					ui_popp_to_parents[ui,losing_popp] = [winning_popp]
				if ui == 3:
					print("{} beats {}".format(winning_popp,losing_popp))
	# Initialize the winning popp as routed popp in the base case with all popps on
	ui_to_ranked_popps = {ui:[all_popps[np.where(routes_np[base_k][:,ui])[0][0]]] for ui in range(n_xi)}
	max_ni = 10
	for ui in range(n_xi):
		print(ui)
		print({k[1]:v for k,v in ui_popp_to_parents.items() if k[0] == ui})
		i=0
		while len(ui_to_ranked_popps[ui]) < n_popps:
			i+=1
			if i >= max_ni: 
				print("SIDOFOISDF")
				exit(0)
			unassigned_popps = {popp:None for popp in get_difference(all_popps, ui_to_ranked_popps[ui])}
			print("Ranked: {}".format(ui_to_ranked_popps[ui]))
			print("Unassigned: {}".format(unassigned_popps))
			for popp in unassigned_popps:
				beaten = False ## if no parents are currently unassigned, assign this as the next-best popp
				for parent_popp in ui_popp_to_parents.get((ui, popp), []):
					try:
						unassigned_popps[parent_popp]
						beaten = True
						break
					except KeyError:
						pass
				if not beaten:
					ui_to_ranked_popps[ui].append(popp)
					break

	print(ui_to_ranked_popps)
	ui_to_ranked_popps = {ui:{popp:i for i,popp in enumerate(ui_to_ranked_popps[ui])} for ui in ui_to_ranked_popps}
	for k in range(n_scenarios):
		scenario = constraint_specification[k][0]
		popp_to_volume = {}
		for ui in ui_to_ranked_popps:
			active_popps = [popp for popp,a in zip(all_popps, scenario) if a]
			routed_popp = sorted(active_popps, key = lambda el : ui_to_ranked_popps[ui][el])[0]
			try:
				popp_to_volume[routed_popp] += user_vols[ui]
			except KeyError:
				popp_to_volume[routed_popp] = user_vols[ui]

		print("Active PoPPs: {} ".format(scenario))
		print("Popp to Volume: {}".format(popp_to_volume))
		print("Desired PoPP to Volume: {}".format({k:v for k,v in constraint_specification[k][1]}))
		print('\n')







if __name__ == "__main__":
	main()
