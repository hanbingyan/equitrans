from config import *
from scipy.stats import kendalltau, spearmanr

corr_log_dir = './corr_logs'

if not os.path.exists(corr_log_dir):
    os.makedirs(corr_log_dir)

# kendall tau correlation log
corr_hist = np.zeros_like(alpha_arr)

for alpha_idx in range(len(alpha_arr)):

    alpha = alpha_arr[alpha_idx]
    # time is forward
    value_func = np.zeros((time_horizon, firmtype_num, workertype_num))

    ot_pi = np.zeros((time_horizon, firmtype_num, workertype_num, firmtype_num, workertype_num))

    wage_hist = np.zeros((firmtype_num, workertype_num, workertype_num))

    for time in range(time_horizon-2, -1, -1):
        # solve backwardly
        for firm_idx in range(firmtype_num):
            for worker_idx in range(workertype_num):
                firm_dist = firm_trans[firm_idx, :]/firm_trans[firm_idx, :].sum()
                worker_dist = worker_trans[worker_idx, :]/worker_trans[worker_idx, :].sum()
                dist_mat = np.zeros_like(cost_mat)

                for i in range(firmtype_num):
                    for j in range(workertype_num):
                        dist_mat[i, j] = np.exp(-(i - firm_idx)**2/2 - (j - worker_idx)**2/2)

                min_obj = DISCOUNT*cost_mat - alpha*dist_mat + DISCOUNT*value_func[time+1, :, :]
                res = ot.emd(firm_dist, worker_dist, min_obj, log=True)

                if time == 0:
                    wage_hist[firm_idx, worker_idx, :] = -res[1]['v']

                ot_plan = res[0]
                # ot_plan = ot.sinkhorn(scr_dist, wage_dist, -surplus, reg=0.1)
                ot_pi[time+1, firm_idx, worker_idx, :, :] = ot_plan.copy()
                value_func[time, firm_idx, worker_idx] = np.multiply(DISCOUNT*cost_mat + DISCOUNT*value_func[time+1, :, :],
                                                                     ot_plan).sum()


    ########## the initial step at 0 ###############
    # for the first step ot plan
    firm_dist = firm_trans.sum(axis=1)/firm_trans.sum()
    worker_dist = worker_trans.sum(axis=1)/worker_trans.sum()

    min_obj = DISCOUNT*cost_mat + DISCOUNT*value_func[0, :, :]

    res_init = ot.emd(firm_dist, worker_dist, min_obj, log=True)
    ot_plan = res_init[0]
    # dual is negative of wage. Besides, normalize it to min = 0.0
    wage_init = -res_init[1]['v']
    wage_init = wage_init - np.min(wage_init)
    ot_init = ot_plan.copy()
    value_init = np.multiply(min_obj, ot_plan).sum()

    ### Calculate statistics
    num_cell = 100
    firm_paired = np.zeros(num_cell)
    worker_paired = np.zeros(num_cell)
    salary_paired = np.zeros(num_cell)
    prob_paired = np.zeros(num_cell)

    tmp_id = 0
    for firm_idx in range(firmtype_num):
        for w_idx in range(workertype_num):
            if ot_init[firm_idx, w_idx] > 0:
                len_ = int(np.round(ot_init[firm_idx, w_idx] * num_cell))
                firm_paired[tmp_id:tmp_id + len_] = firm_idx
                worker_paired[tmp_id:tmp_id + len_] = w_idx
                salary_paired[tmp_id:tmp_id + len_] = wage_init[w_idx]
                prob_paired[tmp_id:tmp_id + len_] = ot_init[firm_idx, w_idx]
                tmp_id = tmp_id + len_

    if firm_paired[-1] == 0:
        # print('Unfilled array')
        firm_paired[tmp_id:] = firm_paired[tmp_id-1]
        worker_paired[tmp_id:] = worker_paired[tmp_id-1]
        salary_paired[tmp_id:] = salary_paired[tmp_id-1]
        prob_paired[tmp_id:] = prob_paired[tmp_id-1]

    print('alpha', alpha)
    print('correlation between firm/worker paired', kendalltau(firm_paired, worker_paired))
    corr_hist[alpha_idx] = kendalltau(firm_paired, worker_paired)[0]
    # print('corr', kendalltau(firm_paired, salary_paired))

    sub_folder = "alpha_{}".format(alpha_idx)
    sub_dir = '{}/{}'.format(corr_log_dir, sub_folder)

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Save params configuration
    with open('{}/params.txt'.format(sub_dir), 'w') as fp:
        fp.write('workertype_num {} \n'.format(workertype_num))
        fp.write('firmtype_num {} \n'.format(firmtype_num))

    with open('{}/value_func.pickle'.format(sub_dir), 'wb') as fp:
        pickle.dump(value_func, fp)

    with open('{}/ot_pi.pickle'.format(sub_dir), 'wb') as fp:
        pickle.dump(ot_pi, fp)

    with open('{}/value_init.pickle'.format(sub_dir), 'wb') as fp:
        pickle.dump(value_init, fp)

    with open('{}/ot_init.pickle'.format(sub_dir), 'wb') as fp:
        pickle.dump(ot_init, fp)

    with open('{}/wage_t1.pickle'.format(sub_dir), 'wb') as fp:
        pickle.dump(wage_hist, fp)

    with open('{}/wage_init.pickle'.format(sub_dir), 'wb') as fp:
        pickle.dump(wage_init, fp)

with open('{}/corr_hist.pickle'.format(corr_log_dir), 'wb') as fp:
    pickle.dump(corr_hist, fp)


