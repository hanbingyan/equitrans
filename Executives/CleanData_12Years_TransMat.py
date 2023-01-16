from config import *
import numpy as np
from scipy.stats import rankdata
from jenks import getJenksBreaks, classify, getGVF

n_group = 6
merged = pd.read_csv('{}/classified_group_{}.csv'.format(log_dir, n_group))
keep_gv = merged['gvkey'].unique()

ceo = pd.read_csv('ceo_20years.csv')
ceo.columns = ceo.columns.str.lower()
ceo = ceo[ceo['gvkey'].isin(keep_gv)]

finstat = pd.read_csv('finstat_20years.csv')
finstat = finstat[~finstat['sale'].isnull()]
finstat = finstat[finstat['gvkey'].isin(keep_gv)]
finstat = finstat.rename(columns={"fyear": "year"})


####### Clean Financial Statement ########
time_selected = np.arange(2010, 2022, 1)
ceo = ceo[ceo['year'].isin(time_selected)]
finstat = finstat[finstat['year'].isin(time_selected)]

gsector = finstat['gsector'].unique()
gsector = np.sort(gsector)

# drop 2022 and 2011 data and missing fyear data
fintwl = finstat[finstat['year'].isin(time_selected)]
fintwl = fintwl[~fintwl['year'].isnull()]
fintwl['year'] = fintwl['year'].astype(np.int64)
# drop gsector missing data
fintwl = fintwl[~fintwl['gsector'].isnull()]
fintwl = fintwl.fillna(0)
fintwl = fintwl.reset_index(drop=True)
gvkey = fintwl['gvkey'].unique()
fintwl = fintwl.loc[:, ['gvkey', 'datadate', 'year', 'tic', 'sale', 'gsector']]

for gs in gsector:
    for year in time_selected:
        sliced_idx = (fintwl['year'] == year) & (fintwl['gsector'] == gs)
        data = fintwl[sliced_idx]['sale'].values
        n_obs = data.shape[0]
        print('Year', year, 'GS', gs, 'No_of_Firms', n_obs)
        sale_list = list(rankdata(-data, method='ordinal') - 1)
        firm_breaks = getJenksBreaks(sale_list, n_group)
        firm_group = np.zeros(n_obs)
        for fidx in range(n_obs):
            firm_group[fidx] = classify(sale_list[fidx], firm_breaks)
        fintwl.loc[sliced_idx, 'firm_group'] = n_group - firm_group
        fintwl.loc[sliced_idx, 'firm_rank'] = rankdata(data, method='ordinal') - 1

fintwl.to_csv('sales_trans_12years.csv', index=False)

########## Clean wage data ###########
ceo = ceo[~ceo['tdc1'].isnull()]
ceo = ceo.reset_index(drop=True)
manager = ceo.groupby(['gvkey', 'year'])['tdc1'].mean().reset_index()
n_obs = manager.shape[0]
for idx in range(n_obs):
    gv = manager.loc[idx, 'gvkey']
    gs = np.unique(finstat[finstat['gvkey'] == gv]['gsector'].values)
    if gs.shape[0] == 1:
        manager.loc[idx, 'gsector'] = gs[0]
manager = manager[~manager['gsector'].isnull()]
for gs in gsector:
    for year in time_selected:
        sliced_idx = (manager['year'] == year) & (manager['gsector'] == gs)
        data = manager[sliced_idx]['tdc1'].values
        n_obs = data.shape[0]
        print('Year', year, 'GS', gs, 'No_of_Firms', n_obs)
        wage_list = list(rankdata(-data, method='ordinal') - 1)
        wage_breaks = getJenksBreaks(wage_list, n_group)
        wage_group = np.zeros(n_obs)
        for fidx in range(n_obs):
            wage_group[fidx] = classify(wage_list[fidx], wage_breaks)
        manager.loc[sliced_idx, 'wage_group'] = n_group - wage_group
        manager.loc[sliced_idx, 'wage_rank'] = rankdata(data, method='ordinal') - 1

manager.to_csv('tdc_12years.csv', index=False)
