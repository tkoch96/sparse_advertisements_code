import pickle
import gurobipy as gp, numpy as np

n_popps, n_paths, available_latencies, cap_constraint_A, caps, volume_conservation_A, conservation_b = pickle.load(open('tmp.pkl','rb'))
### Gurobi solve
model = gp.Model()
# model.Params.LogToConsole = 0
model.Params.Threads = 1

available_latencies = available_latencies.reshape((1, n_paths))
caps = caps.reshape((n_popps,1))
conservation_b = conservation_b.reshape((volume_conservation_A.shape[0], 1))

## lagrange variable to discourage overutilization
lambduh = model.addMVar((1,n_popps), name='capacity_slacks', lb=0)
## amount of volume on each path
x = model.addMVar((n_paths,1), name='volume_each_path', lb=0)

# lambduh = np.random.random((1,n_popps))
# x = np.random.random((n_paths,1))
# print((available_latencies @ x + lambduh @ (cap_constraint_A @ x - caps)).shape)
# print((volume_conservation_A @ x == conservation_b).shape)


import scipy

model.setObjective(available_latencies @ x + lambduh @ (cap_constraint_A @ x - caps))
model.addConstr(volume_conservation_A @ x == conservation_b)
model.addConstr(lambduh >= 0)
model.addConstr(lambduh <= 10)
model.optimize()

# alpha = .1
# ## amount of volume on each path
# x = model.addMVar((n_paths,1), name='volume_each_path', lb=0)
# model.setObjective(available_latencies @ x + alpha * gp.quicksum((cap_constraint_A @ x - caps)))
# model.addConstr(volume_conservation_A @ x == conservation_b)
# model.optimize()

print(x.X)
print(lambduh.X)