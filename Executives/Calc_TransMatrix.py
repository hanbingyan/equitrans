import numpy as np
import pandas as pd
from config import *

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

fintwl = pd.read_csv('./sales_trans_12years.csv')
manager = pd.read_csv('./tdc_12years.csv')
gsector = fintwl['gsector'].unique()
gsector = np.sort(gsector)
print('gs', gsector)
years = fintwl['year'].unique()
time_horizon = len(years)

n_group = 6

for gs in gsector:
    sale_trans = np.zeros((n_group, n_group))
    gv_sec = fintwl[fintwl['gsector'] == gs]['gvkey'].unique()
    for t in np.arange(time_horizon-1):
        for gv in gv_sec:
            row_ = (fintwl['year'] == years[t]) & (fintwl['gvkey'] == gv)
            pre_scr = fintwl[row_]['firm_group'].values
            row_ = (fintwl['year'] == years[t+1]) & (fintwl['gvkey'] == gv)
            aft_scr = fintwl[row_]['firm_group'].values
    #         if aft_scr.shape[0] == 0:
    #             print(uni, t)
            if pre_scr.shape[0] == 1 and aft_scr.shape[0] == 1:
                sale_trans[int(pre_scr), int(aft_scr)] += 1
            # else:
            #     print('Something wrong', pre_scr, aft_scr)

    s_file_name = "sale_trans_gs_{}.pickle".format(int(gs))
    with open('{}/{}'.format(log_dir, s_file_name), 'wb') as fp:
        pickle.dump(sale_trans, fp)


for gs in gsector:
    wage_trans = np.zeros((n_group, n_group))
    gv_sec = manager[manager['gsector'] == gs]['gvkey'].unique()
    for t in np.arange(time_horizon-1):
        for gv in gv_sec:
            row_ = (manager['year'] == years[t]) & (manager['gvkey'] == gv)
            pre_scr = manager[row_]['wage_group'].values
            row_ = (manager['year'] == years[t+1]) & (manager['gvkey'] == gv)
            aft_scr = manager[row_]['wage_group'].values
            if pre_scr.shape[0] == 1 and aft_scr.shape[0] == 1:
                wage_trans[int(pre_scr), int(aft_scr)] += 1
            # else:
            #     print('Something wrong', pre_scr, aft_scr)

    w_file_name = "wage_trans_gs_{}.pickle".format(int(gs))
    with open('{}/{}'.format(log_dir, w_file_name), 'wb') as fp:
        pickle.dump(wage_trans, fp)