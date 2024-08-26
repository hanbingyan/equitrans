import numpy as np
from gurobipy import *

##### The main optimizer ######
def suboptim(cost, risk_aver, px, py, verbose=False):
    m = Model('Primal')
    if verbose:
        m.setParam('OutputFlag', 1)
        m.setParam('LogToConsole', 1)
    else:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)

    nx = px.shape[0]
    ny = py.shape[0]

    ## joint density of X and Y
    pi = m.addVars(nx, ny, lb=0.0, ub=1.0, name='transport')
    # marginal constraints
    m.addConstrs((pi.sum(i, '*') == px[i] for i in range(nx)), name='x_marginal')
    m.addConstrs((pi.sum('*', j) == py[j] for j in range(ny)), name='y_marginal')

    mean = quicksum([cost[i, j]*pi[i, j] for i in range(nx) for j in range(ny)])
    second_moment = quicksum([pi[i, j]*cost[i, j]**2 for i in range(nx) for j in range(ny)])
    obj = mean + risk_aver*(second_moment - mean**2)

    m.setObjective(obj, GRB.MINIMIZE)

    m.params.NonConvex = 2
    m.setParam('MIPGap', 1e-3)

    m.optimize()
    m.printQuality()

    names_to_retrieve = (f"transport[{i},{j}]" for i in range(nx) for j in range(ny))
    coupling = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    # for ans in m.getVars():
    #   print('%s %g' % (ans.VarName, ans.X))

    coupling = coupling.reshape((nx, ny))

    return obj.getValue(), coupling


cost_mat = np.array([[1.0, 2.0], [2.0, 0.0]])
gamma = 1.0
kernels = np.zeros((2, 2, 2, 2))

value_func_time1 = np.zeros((2, 2))

for x1 in range(2):
    for y1 in range(2):
        if x1 == 0:
            px = np.array([0.8, 0.2])
        elif x1 == 1:
            px = np.array([0.2, 0.8])

        if y1 == 0:
            py = np.array([0.9, 0.1])
        elif y1 == 1:
            py = np.array([0.1, 0.9])

        # px = np.array([0.5, 0.5])
        # py = np.array([0.5, 0.5])

        val, coupling = suboptim(cost=cost_mat, risk_aver=gamma, px=px, py=py, verbose=False)
        kernels[x1, y1, :, :] = coupling
        value_func_time1[x1, y1] = val

print('Couplings at the second step')
print(kernels)


def firstoptim(cost, cond_kernel, risk_aver, px, py, verbose=False):
    m = Model('Primal')
    if verbose:
        m.setParam('OutputFlag', 1)
        m.setParam('LogToConsole', 1)
    else:
        m.setParam('OutputFlag', 0)
        m.setParam('LogToConsole', 0)

    nx1 = cond_kernel.shape[0]
    ny1 = cond_kernel.shape[1]
    nx2 = cond_kernel.shape[2]
    ny2 = cond_kernel.shape[3]

    ## joint density of X and Y
    pi = m.addVars(nx1, ny1, lb=0.0, ub=1.0, name='transport')
    # marginal constraints
    m.addConstrs((pi.sum(i, '*') == px[i] for i in range(nx1)), name='x_marginal')
    m.addConstrs((pi.sum('*', j) == py[j] for j in range(ny1)), name='y_marginal')



    mean = quicksum([cost[i, j, k, l]*pi[i, j]*cond_kernel[i, j, k, l] for i in range(nx1) for j in range(ny1)
                     for k in range(nx2) for l in range(ny2)])
    second_moment = quicksum([pi[i, j]*cond_kernel[i, j, k, l]*cost[i, j, k, l]**2 for i in range(nx1)
                              for j in range(ny1) for k in range(nx2) for l in range(ny2)])
    obj = mean + risk_aver*(second_moment - mean**2)

    m.setObjective(obj, GRB.MINIMIZE)

    m.params.NonConvex = 2
    m.setParam('MIPGap', 1e-3)


    m.optimize()
    m.printQuality()

    names_to_retrieve = (f"transport[{i},{j}]" for i in range(nx1) for j in range(ny1))
    coupling_firststep = np.array([m.getVarByName(name).X for name in names_to_retrieve])
    # for ans in m.getVars():
    #   print('%s %g' % (ans.VarName, ans.X))

    return obj.getValue(), coupling_firststep.reshape((nx1, ny1))

two_periods_cost = np.zeros((2, 2, 2, 2))
for x1 in range(2):
    for y1 in range(2):
        for x2 in range(2):
            for y2 in range(2):
                two_periods_cost[x1, y1, x2, y2] = cost_mat[x1, y1] + cost_mat[x2, y2]


total_obj, coupling_firststep = firstoptim(cost=two_periods_cost, cond_kernel=kernels,
                                           risk_aver=gamma, px=[0.1, 0.9], py=[0.5, 0.5], verbose=False)

print('Value function at time 0', total_obj)
print('Coupling for x1 and y1')
print(coupling_firststep)
