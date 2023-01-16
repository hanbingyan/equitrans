from config import *
pd.set_option('mode.chained_assignment', None)

n_group = 6

bench_zero = False
SIMDATA_PROP = 0.0
merged = pd.read_csv('{}/classified_group_{}.csv'.format(log_dir, n_group))
gs = merged['gsector'].unique()
gs = np.sort(gs)
print('gs', gs)
year = merged['year'].unique()
time_horizon = len(year)

s_split_n = n_group
w_split_n = n_group

l2_mat = np.zeros((s_split_n, w_split_n))
for i in range(s_split_n):
    for j in range(w_split_n):
        l2_mat[i, j] = 10.0*np.abs(i/s_split_n - j/w_split_n)

for gs_idx in range(11):

    # Load the transition matrix
    w_file_name = "wage_trans_gs_{}.pickle".format(int(gs[gs_idx]))
    with open('{}/{}'.format(log_dir, w_file_name), 'rb') as fp:
        wage_trans = pickle.load(fp)

    s_file_name = "sale_trans_gs_{}.pickle".format(int(gs[gs_idx]))
    with open('{}/{}'.format(log_dir, s_file_name), 'rb') as fp:
        sale_trans = pickle.load(fp)

    industry = [gs[gs_idx]]
    gvkey = merged[merged['gsector'].isin(industry)]['gvkey'].unique()
    firm_num = len(gvkey)

    print('GICS code', industry)
    print('No. of firms', firm_num)

    # observed path, the second dimension contains score and wage
    obs_path = np.zeros((firm_num, time_horizon*2))
    sliced_df = merged[merged['gsector'].isin(industry)]

    gslog_dir = '{}/gs_{}_group_{}_zero_{}_simprop_{}'.format(log_dir, int(gs[gs_idx]), n_group, bench_zero, SIMDATA_PROP)

    if not os.path.exists(gslog_dir):
        os.makedirs(gslog_dir)

    # condi_dist = np.zeros((s_split_n, w_split_n))
    for k in range(firm_num):
        firm = gvkey[k]
        for t in range(time_horizon):
            firm_group = sliced_df[(sliced_df['gvkey'] == firm) & (sliced_df['year'] == year[t])]['firm_group'].values[0]
            wage_group = sliced_df[(sliced_df['gvkey'] == firm) & (sliced_df['year'] == year[t])]['wage_group'].values[0]
            obs_path[k, 2*t] = int(firm_group)
            obs_path[k, 2*t+1] = int(wage_group)
            # condi_dist[int(firm_group), int(wage_group)] += 1

    with open('{}/obs_path.pickle'.format(gslog_dir), 'wb') as fp:
        pickle.dump(obs_path, fp)

    sim_firm_n = 500

    for sim_idx in range(10):
        print('Simulation', sim_idx)
        # observed path, the second dimension contains score and wage
        sim_path = np.zeros((sim_firm_n, time_horizon*2))

        for k in range(0, sim_firm_n):
            prob = np.random.uniform(0.0, 1.0, size=1)[0]
            draw_firm = np.random.randint(firm_num, size=1, dtype=int)
            if bench_zero:
                sim_path[k, ::2] = np.round(obs_path[draw_firm, ::2].mean())
                sim_path[k, 1::2] = sim_path[k, ::2]
            else:
                if prob > SIMDATA_PROP:
                    sim_path[k, :] = obs_path[draw_firm, :]
                else:
                    sim_path[k, ::2] = np.round(obs_path[draw_firm, ::2].mean())
                    sim_path[k, 1::2] = np.round(obs_path[draw_firm, 1::2].mean())

        with open('{}/sim_path_{}.pickle'.format(gslog_dir, sim_idx), 'wb') as fp:
            pickle.dump(sim_path, fp)
        print('Simulated diff', np.abs(sim_path[:, ::2] - sim_path[:, 1::2]).mean())
        for alpha_idx in range(len(alpha_arr)):

            alpha = alpha_arr[alpha_idx]
            # all the correspondence between wage discrete level and wage number, time is forward
            value_func = np.zeros((time_horizon, s_split_n, w_split_n))
            # value_func = 10.0*np.ones((time_horizon, s_split_n, w_split_n))
            # ot_pi[time, state, :, :] is the plan under time and state
            # ot_pi[0] is not used, use ot_init instead
            ot_pi = np.zeros((time_horizon, s_split_n, w_split_n, s_split_n, w_split_n))

            for time in range(time_horizon-2, -1, -1):
                for firm_idx in range(s_split_n):
                    for wage_idx in range(w_split_n):
                        firm_dist = sale_trans[firm_idx, :]/sale_trans[firm_idx, :].sum()
                        wage_dist = wage_trans[wage_idx, :]/wage_trans[wage_idx, :].sum()
                        dist_mat = np.zeros_like(l2_mat)
                        # dist_mat = np.ones_like(l2_mat)*10.0
                        dist_mat[firm_idx, wage_idx] = 30.0
                        # for i in range(s_split_n):
                        #     for j in range(w_split_n):
                        #         dist_mat[i, j] = 20.0*np.exp(-np.abs(j-wage_idx)/w_split_n -np.abs(i-firm_idx)/w_split_n)
                            # dist_mat[i, j] = np.exp(-4 * np.abs((j - i)-(wage_idx - firm_idx))/ w_split_n)

                        min_obj = l2_mat - alpha*dist_mat + DISCOUNT*value_func[time+1, :, :]
                        ot_plan = ot.emd(firm_dist, wage_dist, min_obj)
                        # ot_plan = ot.sinkhorn(scr_dist, wage_dist, -surplus, reg=0.1)
                        ot_pi[time+1, firm_idx, wage_idx, :, :] = ot_plan.copy()
                        value_func[time, firm_idx, wage_idx] = np.multiply(l2_mat + DISCOUNT*value_func[time+1, :, :], ot_plan).sum()


            ########## deal with the inital step ###############
            # for the first step ot plan
            # firm_dist = np.ones(s_split_n)/s_split_n
            # wage_dist = np.ones(w_split_n)/w_split_n
            firm_dist = sale_trans.sum(axis=1) / sale_trans.sum()
            wage_dist = wage_trans.sum(axis=1) / wage_trans.sum()

            ######## build sales data for the first step ############
            min_obj = l2_mat + DISCOUNT*value_func[0, :, :]
            ot_plan = ot.emd(firm_dist, wage_dist, min_obj)
            # Since there is no previous state at time 0, ot plan is only one matrix with no previous state
            ot_init = ot_plan.copy()
            value_init = np.multiply(min_obj, ot_plan).sum()


            #### Save logs ####
            sub_folder = "sim_{}_alpha_{}".format(sim_idx, alpha_idx)
            sub_dir = '{}/{}'.format(gslog_dir, sub_folder)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            # Save params configuration
            with open('{}/params.txt'.format(sub_dir), 'w') as fp:
                fp.write('sim firm num {} \n'.format(sim_firm_n))
                fp.write('w_split_n {} \n'.format(w_split_n))
                fp.write('s_split_n {} \n'.format(s_split_n))

            with open('{}/value_func.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(value_func, fp)

            with open('{}/ot_pi.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(ot_pi, fp)

            with open('{}/value_init.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(value_init, fp)

            with open('{}/ot_init.pickle'.format(sub_dir), 'wb') as fp:
                pickle.dump(ot_init, fp)
