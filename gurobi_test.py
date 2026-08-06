import gurobipy as gp

# Define your license keys directly in Python
options = {
    "WLSACCESSID": "your-wls-access-id",
    "WLSSECRET": "your-wls-secret-key",
    "LICENSEID": 1234567,  # Replace with your actual License ID
}

# 1. Initialize the Gurobi environment with your keys
env = gp.Env(params=options)

# 2. Pass that environment to your model
m = gp.Model(env=env)

m.setParam("OutputFlag", 0)
x = m.addVars(5000, vtype=gp.GRB.BINARY)
m.optimize()
print("Full license OK — no size limit")
