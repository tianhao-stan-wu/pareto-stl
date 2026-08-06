import gurobipy as gp
m = gp.Model()
m.setParam("OutputFlag", 0)
x = m.addVars(5000, vtype=gp.GRB.BINARY)   # well past the 2000-var trial cap
m.optimize()
print("Full license OK — no size limit")