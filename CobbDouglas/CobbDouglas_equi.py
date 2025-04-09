from config import *

bench_zero = True
obs_path = np.array([[0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0., 2., 0., 3.],
                     [1., 0., 1., 1., 1., 0., 1., 1., 1., 2.],
                     [1., 3., 2., 3., 1., 3., 1., 2., 2., 3.],
                     [2., 0., 2., 2., 2., 2., 2., 1., 2., 1.],
                     [2., 3., 1., 2., 2., 1., 2., 0., 1., 1.],
                     [3., 2., 3., 0., 3., 3., 3., 3., 3., 0.],
                     [3., 4., 3., 4., 4., 4., 4., 3., 4., 4.],
                     [4., 2., 4., 3., 4., 2., 3., 4., 4., 2.]])

sample_num = obs_path.shape[0]

sim_log_dir = '{}/zero_{}'.format(log_dir, bench_zero)

if not os.path.exists(sim_log_dir):
    os.makedirs(sim_log_dir)

sim_firm_n = 200

for sim_idx in range(10):
    print('Simulation', sim_idx)
    # observed path, the second dimension contains worker types
    sim_path = np.zeros((sim_firm_n, time_horizon*2))

    for k in range(0, sim_firm_n):
        draw_firm = np.random.randint(sample_num, size=1, dtype=int)
        if bench_zero:
            sim_path[k, :] = np.round(obs_path[draw_firm, ::2].mean())
        else:
            sim_path[k, :] = obs_path[draw_firm, :]

    with open('{}/sim_path_{}.pickle'.format(sim_log_dir, sim_idx), 'wb') as fp:
        pickle.dump(sim_path, fp)

    for alpha_idx in range(len(alpha_arr)):

        alpha = alpha_arr[alpha_idx]
        value_func = np.zeros((time_horizon, firmtype_num, workertype_num))

        ot_pi = np.zeros((time_horizon, firmtype_num, workertype_num, firmtype_num, workertype_num))

        # wage_hist = np.zeros((firmtype_num, workertype_num, workertype_num))

        for time in range(time_horizon-2, -1, -1):
            # solve backwardly
            for firm_idx in range(firmtype_num):
                for worker_idx in range(workertype_num):
                    firm_dist = firm_trans[firm_idx, :]/firm_trans[firm_idx, :].sum()
                    worker_dist = worker_trans[worker_idx, :]/worker_trans[worker_idx, :].sum()
                    dist_mat = np.zeros_like(cost_mat)

                    for i in range(firmtype_num):
                        for j in range(workertype_num):
                            dist_mat[i, j] = np.exp(-(i - firm_idx)**2/2 -(j - worker_idx)**2/2)

                    min_obj = DISCOUNT*cost_mat - alpha*dist_mat + DISCOUNT*value_func[time+1, :, :]
                    res = ot.emd(firm_dist, worker_dist, min_obj, log=True)

                    # if time == 0:
                    #     wage_hist[firm_idx, worker_idx, :] = -res[1]['v']

                    ot_plan = res[0]
                    ot_pi[time+1, firm_idx, worker_idx, :, :] = ot_plan.copy()
                    value_func[time, firm_idx, worker_idx] = np.multiply(DISCOUNT*cost_mat + DISCOUNT*value_func[time+1, :, :],
                                                                         ot_plan).sum()

        ########## deal with the initial step ###############
        firm_dist = firm_trans.sum(axis=1)/firm_trans.sum()
        worker_dist = worker_trans.sum(axis=1)/worker_trans.sum()

        min_obj = DISCOUNT*cost_mat + DISCOUNT*value_func[0, :, :]
        res_init = ot.emd(firm_dist, worker_dist, min_obj, log=True)
        ot_plan = res_init[0]
        # print('Initial wage', -res_init[1]['v'])
        wage_init = -res_init[1]['v']
        ot_init = ot_plan.copy()
        value_init = np.multiply(min_obj, ot_plan).sum()

        sub_folder = "sim_{}_alpha_{}".format(sim_idx, alpha_idx)
        sub_dir = '{}/{}'.format(sim_log_dir, sub_folder)

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # Save params configuration
        with open('{}/params.txt'.format(sub_dir), 'w') as fp:
            fp.write('sim firm num {} \n'.format(sim_firm_n))
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

        # with open('{}/wage_t1.pickle'.format(sub_dir), 'wb') as fp:
        #     pickle.dump(wage_hist, fp)

        with open('{}/wage_init.pickle'.format(sub_dir), 'wb') as fp:
            pickle.dump(wage_init, fp)
