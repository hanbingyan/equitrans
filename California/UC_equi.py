import os
from config import *

pd.set_option('mode.chained_assignment', None)

bench_zero = False

merged = pd.read_csv('uc_salary.csv')
prof_title = ['Prof-Ay-B/E/E', 'Assoc Prof-Ay-B/E/E', 'Asst Prof-Ay-B/E/E', 'Postdoc-Employee']
# years = merged['Year'].unique()
# years = np.arange(2013, 2022, 1)
years = np.arange(2017, 2022, 1)
time_horizon = years.shape[0]
universities = merged['EmployerName'].unique()
base_num = universities.shape[0]
w_split_n = merged[w_name].nunique() # base_num
s_split_n = merged[u_name].nunique() # base_num
uni_num = 9
# prof_idx = 2
l2_mat = np.zeros((s_split_n, w_split_n))
for i in range(s_split_n):
    for j in range(w_split_n):
        l2_mat[i, j] = 10.0*np.abs(i/s_split_n - j/w_split_n) # multiplied 10 before

for prof_idx in range(4):

    # observed path, the second dimension contains score and wage
    obs_path = np.zeros((uni_num, time_horizon*2))
    for k in range(uni_num):
        uni = universities[k]
        for t in range(time_horizon):
            row_ = (merged['Year'] == years[t]) & (merged['EmployerName'] == uni) & (merged['Position'] == prof_title[prof_idx])
            uni_group = merged[row_][u_name].values
            row_ = (merged['Year'] == years[t]) & (merged['EmployerName'] == uni) & (merged['Position'] == prof_title[prof_idx])
            wage_group = merged[row_][w_name].values
            obs_path[k, 2*t] = int(uni_group)
            obs_path[k, 2*t+1] = int(wage_group)


    # # Load the transmatrix first
    with open('{}/wage_trans_{}.pickle'.format(log_dir, prof_idx), 'rb') as fp:
        wage_trans = pickle.load(fp)

    with open('{}/uni_trans.pickle'.format(log_dir), 'rb') as fp:
        uni_trans = pickle.load(fp)


    sim_log_dir = '{}/prof_{}_zero_{}'.format(log_dir, prof_idx, bench_zero)

    if not os.path.exists(sim_log_dir):
        os.makedirs(sim_log_dir)


    sim_uni_n = 200

    for sim_idx in range(10):
        print('Simulation', sim_idx)
        # observed path, the second dimension contains score and wage
        sim_path = np.zeros((sim_uni_n, time_horizon*2))

        for k in range(0, sim_uni_n):
            draw_uni = np.random.randint(uni_num, size=1, dtype=int)
            # prob = np.random.uniform(0.0, 1.0, size=1)[0]
            if bench_zero:
                sim_path[k, :] = np.round(obs_path[draw_uni, ::2].mean())
                # sim_path[k, ::2] = np.round(obs_path[draw_uni, 1::2].mean())
            else:
                sim_path[k, :] = obs_path[draw_uni, :]

        with open('{}/sim_path_{}.pickle'.format(sim_log_dir, sim_idx), 'wb') as fp:
            pickle.dump(sim_path, fp)

        for alpha_idx in range(len(alpha_arr)):

            alpha = alpha_arr[alpha_idx]
            # all the correspondence between wage discrete level and wage number, time is forward
            value_func = np.zeros((time_horizon, s_split_n, w_split_n))
            # value_func = 10.0*np.ones((time_horizon, s_split_n, w_split_n))
            ot_pi = np.zeros((time_horizon, s_split_n, w_split_n, s_split_n, w_split_n))


            for time in range(time_horizon-2, -1, -1):
                # work backwardly

                for uni_idx in range(s_split_n):
                    for wage_idx in range(w_split_n):
                        uni_dist = uni_trans[uni_idx, :]/uni_trans[uni_idx, :].sum()
                        wage_dist = wage_trans[wage_idx, :]/wage_trans[wage_idx, :].sum()
                        dist_mat = np.zeros_like(l2_mat)
                        # dist_mat[uni_idx, wage_idx] = 30.0
                        for i in range(s_split_n):
                            for j in range(w_split_n):
                                dist_mat[i, j] = 30.0*np.exp(-np.abs(j - wage_idx)/w_split_n - np.abs(i - uni_idx)/s_split_n)
                                # dist_mat[i, j] = np.exp(-4 * np.abs((j - i)-(wage_idx - uni_idx))/ w_split_n)

                        min_obj = l2_mat - alpha*dist_mat + DISCOUNT*value_func[time+1, :, :]
                        ot_plan = ot.emd(uni_dist, wage_dist, min_obj)
                        # ot_plan = ot.sinkhorn(scr_dist, wage_dist, -surplus, reg=0.1)
                        ot_pi[time+1, uni_idx, wage_idx, :, :] = ot_plan.copy()
                        value_func[time, uni_idx, wage_idx] = np.multiply(l2_mat + DISCOUNT*value_func[time+1, :, :], ot_plan).sum()


            ########## deal with the inital step ###############
            # for the first step ot plan

            uni_dist = uni_trans.sum(axis=1) / uni_trans.sum()
            wage_dist = wage_trans.sum(axis=1) / wage_trans.sum()

            ######## build sales data for the first step ############
            min_obj = l2_mat + DISCOUNT*value_func[0, :, :]
            ot_plan = ot.emd(uni_dist, wage_dist, min_obj)
            # Since there is no previous state at time 0, ot plan is only one matrix with no previous state

            ot_init = ot_plan.copy()
            value_init = np.multiply(min_obj, ot_plan).sum()

            # sub_folder = "{}_{}_{}_{}".format('sector', gs[gs_idx], 'alpha', alpha)
            sub_folder = "sim_{}_alpha_{}".format(sim_idx, alpha_idx)
            sub_dir = '{}/{}'.format(sim_log_dir, sub_folder)

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            # Save params configuration
            with open('{}/params.txt'.format(sub_dir), 'w') as fp:
                fp.write('sim uni num {} \n'.format(sim_uni_n))
                fp.write('w_split_n {} \n'.format(w_split_n))
                fp.write('s_split_n {} \n'.format(s_split_n))
                # fp.write('scale const {} \n'.format(scale_const))

            with open('{}/value_func.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(value_func, fp)

            with open('{}/ot_pi.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(ot_pi, fp)

            with open('{}/value_init.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(value_init, fp)

            with open('{}/ot_init.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(ot_init, fp)
