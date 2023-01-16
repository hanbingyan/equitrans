import numpy as np
from config import *
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from scipy.stats import rankdata, spearmanr, kendalltau
from jenks import getJenksBreaks, classify, getGVF

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

merged = pd.read_csv('merged_y17_avgmanager.csv')
merged = merged[merged['year'].isin([2017, 2018, 2019, 2020, 2021])]
junk_grade = ['CCC+', 'CCC', 'CCC-', 'CC+', 'CC', 'CC-', np.nan, 'nan']
merged = merged[~merged['credit'].isin(junk_grade)]
merged = merged.reset_index(drop=True)

merged['tdc1'] = np.maximum(1.0, merged['tdc1'].values)/1000.0
merged['mkvalt'] = np.maximum(1.0, merged['mkvalt'].values)
merged['sale'] = np.maximum(1.0, merged['sale'].values)
gs = merged['gsector'].unique()
gs = np.sort(gs)
print('gs', gs)
year = merged['year'].unique()
time_horizon = len(year)

rating = pd.read_csv('rating_y17.csv')
rate_level = []
for k in range(len(gs)):
    rating_num = merged[merged['gsector'] == gs[k]]['credit'].nunique()
    rate_level.append(rating_num)
print('Rate level', rate_level)


wage_beta = pd.DataFrame()
beta_idx = 0

search_mode = False
even_split = True

if search_mode:
    candidates = range(2, 20, 1)
    spearman_list = []
    kendall_list = []
else:
    candidates = [6]

for n_group in candidates:
    for gs_idx in range(len(gs)):
        industry = [gs[gs_idx]]
        gvkey = merged[merged['gsector'].isin(industry)]['gvkey'].unique()
        firm_num = len(gvkey)
        # print('GICS code', industry)
        # print('No. of firms', firm_num)

        # Calculate the transition matrix
        for k in range(0, time_horizon, 1):
            # work forwardly
            time = year[k]
            df = merged[(merged['gsector'].isin(industry)) & (merged['year'] == time)]
            sel_idx = df.index
            # calculate the fair salary information
            X = df['sale'].values
            X = X.reshape(-1, 1)
            y = df['tdc1'].values
            reg = HuberRegressor().fit(np.log(X), y)

            if even_split:
                sale_list = list(rankdata(-X.reshape(-1), method='ordinal') - 1)
            else:
                sale_list = list(np.log(X).reshape(-1))
            firm_breaks = getJenksBreaks(sale_list, n_group)
            firm_len = len(sale_list)
            firm_group = np.zeros(firm_len)
            for fidx in range(firm_len):
                firm_group[fidx] = classify(sale_list[fidx], firm_breaks)
            merged.loc[sel_idx, 'firm_group'] = n_group - firm_group


            df_wage = df['tdc1'].values
            if even_split:
                wage_list = list(rankdata(-df_wage, method='ordinal') - 1)
            else:
                wage_list = list(df_wage)
            wage_breaks = getJenksBreaks(wage_list, n_group)
            wage_len = len(wage_list)
            wage_group = np.zeros(wage_len)
            for fidx in range(wage_len):
                wage_group[fidx] = classify(wage_list[fidx], wage_breaks)
            merged.loc[sel_idx, 'wage_group'] = n_group - wage_group

            # save coef of wage
            row = pd.DataFrame({'gsector': gs[gs_idx], 'n_group': n_group,
                                'year': time, 'R-squared': reg.score(np.log(X), y),
                                'Wage_GVF': getGVF(wage_list, n_group),
                                'Sale_GVF': getGVF(sale_list, n_group),
                                'wage_coef': reg.coef_}, index=[beta_idx])
            beta_idx += 1
            wage_beta = pd.concat([wage_beta, row])


    spearman_arr = np.zeros(11)
    kendall_arr = np.zeros(11)
    for k in range(11):
        spearman_arr[k] = merged[merged['gsector'] == gs[k]][['tdc1', 'sale']].corr(method='spearman').iloc[0, 1]
        kendall_arr[k] = merged[merged['gsector'] == gs[k]][['tdc1', 'sale']].corr(method='kendall').iloc[0, 1]


    diff = np.zeros(11)
    for gs_idx in range(11):
        sliced_df = merged[merged['gsector'] == gs[gs_idx]]
        diff[gs_idx] = np.abs(sliced_df['firm_group'] - sliced_df['wage_group']).mean()
        # print('GS', gs[gs_idx], gs_idx, 'Diff', diff[gs_idx])

    spearman_res = spearmanr(spearman_arr, diff)
    kendall_res = kendalltau(kendall_arr, diff)
    print('Number of group', n_group, 'correlation', spearman_res)
    # print('Number of group', n_group, 'Kendall tau', kendalltau(kendall_rho_arr, diff))

    if search_mode:
        spearman_list.append(spearman_res[0])
        kendall_list.append(kendall_res[0])

    if not search_mode:
        merged.to_csv('{}/classified_group_{}.csv'.format(log_dir, n_group), index=False)
        wage_beta.to_csv('{}/wage_beta_group_{}.csv'.format(log_dir, n_group), index=False)

if search_mode:
    with open('{}/spearman_list.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(np.array(spearman_list), fp)

    with open('{}/kendall_list.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(np.array(kendall_list), fp)
