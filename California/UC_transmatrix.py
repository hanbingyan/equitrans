from config import *
from scipy.stats import rankdata

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

merged = pd.read_csv('uc_salary.csv')
prof_title = ['Prof-Ay-B/E/E', 'Assoc Prof-Ay-B/E/E', 'Asst Prof-Ay-B/E/E', 'Postdoc-Employee'] # merged['Position'].unique()
# years = merged['Year'].unique()
years = np.arange(2013, 2022, 1)
time_horizon = years.shape[0]
universities = merged['EmployerName'].unique()
base_num = universities.shape[0]
w_split_n = merged[w_name].nunique() # base_num
s_split_n = merged[u_name].nunique() # base_num

# Calculate the transition matrix
for pos_idx in np.arange(len(prof_title)):
    position = prof_title[pos_idx]
    wage_trans = np.zeros((w_split_n, w_split_n))
    for t in np.arange(time_horizon-1):
        for uni in universities:
            row_ = (merged['Year'] == years[t]) & (merged['EmployerName'] == uni) & (merged['Position'] == position)
            pre_idx = merged[row_][w_name].values
            row_ = (merged['Year'] == years[t+1]) & (merged['EmployerName'] == uni) & (merged['Position'] == position)
            aft_idx = merged[row_][w_name].values
            if pre_idx.shape[0] and aft_idx.shape[0]:
                wage_trans[int(pre_idx), int(aft_idx)] += 1

    w_file_name = 'wage_trans_{}.pickle'.format(pos_idx)
    with open('{}/{}'.format(log_dir, w_file_name), 'wb') as fp:
        pickle.dump(wage_trans, fp)



uni_trans = np.zeros((s_split_n, s_split_n))
position = prof_title[1]
for t in np.arange(time_horizon-1):
    for uni in universities:
        row_ = (merged['Year'] == years[t]) & (merged['EmployerName'] == uni) & (merged['Position'] == position)
        pre_scr = merged[row_][u_name].values
        row_ = (merged['Year'] == years[t+1]) & (merged['EmployerName'] == uni) & (merged['Position'] == position)
        aft_scr = merged[row_][u_name].values
        if aft_scr.shape[0] == 0:
            print(uni, t)
        if pre_scr.shape[0] and aft_scr.shape[0]:
            uni_trans[int(pre_scr), int(aft_scr)] += 1

s_file_name = 'uni_trans.pickle'
with open('{}/{}'.format(log_dir, s_file_name), 'wb') as fp:
    pickle.dump(uni_trans, fp)
