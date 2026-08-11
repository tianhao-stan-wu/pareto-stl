import gurobipy as gp

# 1. Initialize the Gurobi environment with your keys
env = gp.Env()

# 2. Pass that environment to your model
m = gp.Model(env=env)

m.setParam("OutputFlag", 0)
x = m.addVars(5000, vtype=gp.GRB.BINARY)
m.optimize()
print("Full license OK — no size limit")
