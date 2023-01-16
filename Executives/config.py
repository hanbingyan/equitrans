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
alpha_arr = np.linspace(-0.5, 0.5, 51)
# sinkhorn coef
sink_reg_coef = 0.1

log_dir = './exec_logs'
cali_dir = './exec_cali'

# if not os.path.exists(cali_dir):
#     os.makedirs(cali_dir)
