import numpy as np
from gurobipy import *

### The main optimizer

nx1 = 2
nx2 = 2
ny1 = 2
ny2 = 2

cost_mat = np.array([[1.0, 2.0], [2.0, 0.0]])
gamma = 1.0
coupling_results = np.zeros((nx1, ny1, nx2, ny2))

px = np.zeros((nx1, nx2))
py = np.zeros((ny1, ny2))


px[0, :] = 0.1*np.array([0.8, 0.2])
px[1, :] = 0.9*np.array([0.2, 0.8])

py[0, :] = 0.5*np.array([0.9, 0.1])
py[1, :] = 0.5*np.array([0.1, 0.9])
#
# px[0, :] = 0.5*np.array([0.5, 0.5])
# px[1, :] = 0.5*np.array([0.5, 0.5])
#
# py[0, :] = 0.5*np.array([0.5, 0.5])
# py[1, :] = 0.5*np.array([0.5, 0.5])


def optim(cost, risk_aver, px, py, verbose=False):
    m = Model('Primal')
    if verbose:
        m.setParam('OutputFlag', 1)
        m.setParam('LogToConsole', 1)
    else:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)

    ## joint density of X and Y
    pi = m.addVars(nx1, ny1, nx2, ny2, lb=0.0, ub=1.0, name='transport')
    # marginal constraints
    m.addConstrs((pi.sum(i, '*', j, '*') == px[i, j] for i in range(nx1) for j in range(nx2)), name='x_marginal')
    m.addConstrs((pi.sum('*', k, '*', l) == py[k, l] for k in range(ny1) for l in range(ny2)), name='y_marginal')

    # # causal constraint
    m.addConstrs((px[i, :].sum()*pi.sum(i, j, k, '*') == pi.sum(i, j, '*', '*')*px[i, k] for i in range(nx1)
                  for j in range(ny1) for k in range(nx2)), name='causal')
    #
    # anticausal constraint
    m.addConstrs((py[j, :].sum()*pi.sum(i, j, '*', l) == pi.sum(i, j, '*', '*')*py[j, l] for i in range(nx1)
                  for j in range(ny1) for l in range(ny2)), name='anticausal')



    mean = quicksum([cost[i, j, k, l]*pi[i, j, k, l] for i in range(nx1) for j in range(ny1)
                     for k in range(nx2) for l in range(ny2)])
    second_moment = quicksum([pi[i, j, k, l]*cost[i, j, k, l]**2 for i in range(nx1) for j in range(ny1)
                     for k in range(nx2) for l in range(ny2)])
    obj = mean + risk_aver*(second_moment - mean**2)

    m.setObjective(obj, GRB.MINIMIZE)

    m.params.NonConvex = 2
    m.setParam('MIPGap', 1e-9)
    # print('Maximizing the exotic option...')

    m.optimize()
    # m.printQuality()

    names_to_retrieve = (f"transport[{i},{j},{k},{l}]" for i in range(nx1) for j in range(ny1)
                         for k in range(nx2) for l in range(ny2))
    coupling = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    # for ans in m.getVars():
    #   print('%s %g' % (ans.VarName, ans.X))

    return obj.getValue(), coupling.reshape((nx1, ny1, nx2, ny2))

two_periods_cost = np.zeros((2, 2, 2, 2))
for x1 in range(2):
    for y1 in range(2):
        for x2 in range(2):
            for y2 in range(2):
                two_periods_cost[x1, y1, x2, y2] = cost_mat[x1, y1] + cost_mat[x2, y2]


total_obj, coupling = optim(cost=two_periods_cost, risk_aver=gamma, px=px, py=py, verbose=True)




first_marginal = np.zeros((nx1, ny1))
for x1 in range(2):
    for y1 in range(2):
        first_marginal[x1, y1] = coupling[x1, y1, :, :].sum()

print('Coupling for x1 and y1', first_marginal)

for x1 in range(2):
    for y1 in range(2):
        if first_marginal[x1, y1] != 0:
            print('Conditional kernel', x1, y1, coupling[x1, y1, :, :]/first_marginal[x1, y1])

print('Objective value', total_obj)