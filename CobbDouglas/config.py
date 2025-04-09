import pandas as pd
import numpy as np
import os
import seaborn as sns
import ot
import pickle
np.random.seed(12345)
pd.set_option('mode.chained_assignment', None)

# scale const for linear regression
DISCOUNT = 0.9
# scale_const = 2000
alpha_arr = np.linspace(-0.5, 0.5, 51)*10.0
# sinkhorn coef
sink_reg_coef = 0.1

time_horizon = 5
workertype_num = 5
firmtype_num = 5

cost_mat = np.zeros((firmtype_num, workertype_num))
for i in range(firmtype_num):
    for j in range(workertype_num):
        cost_mat[i, j] = -1/(i+1)*(5**0.6 - (j+1)**0.6)

worker_trans = np.array([[0.4, 0.15, 0.15, 0.15, 0.15], [0.15, 0.4, 0.15, 0.15, 0.15],
                         [0.15, 0.15, 0.4, 0.15, 0.15], [0.15, 0.15, 0.15, 0.4, 0.15],
                         [0.15, 0.15, 0.15, 0.15, 0.4]])

firm_trans = np.array([[10., 0., 0., 0., 0.], [0., 14., 6., 0., 0.],
                       [0., 6., 14., 0., 0.], [0., 0., 0., 18., 2.],
                       [0., 0., 0., 1., 9.]])

# initial distribution is uniform in this case. Main conclusions are the same. Graphs are very trivially true.
# firm_trans = np.array([[1.0, 0., 0., 0., 0.], [0., 0.7, 0.3, 0., 0.],
#                        [0., 0.3, 0.7, 0., 0.], [0., 0., 0., 0.9, 0.1],
#                        [0., 0., 0., 0.1, 0.9]])

log_dir = './CobbDouglas_logs'
cali_dir = './CobbDouglas_cali'



