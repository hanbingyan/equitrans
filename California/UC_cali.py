from config import *
import matplotlib.pyplot as plt

bench_zero = False
merged = pd.read_csv('uc_salary.csv')
prof_title = ['Prof-Ay-B/E/E', 'Assoc Prof-Ay-B/E/E', 'Asst Prof-Ay-B/E/E', 'Postdoc-Employee']
years = np.arange(2017, 2022, 1)
time_horizon = years.shape[0]
universities = merged['EmployerName'].unique()
base_num = universities.shape[0]
w_split_n = merged[w_name].nunique() # base_num
s_split_n = merged[u_name].nunique() # base_num
uni_num = 9


for prof_idx in range(4):
    for sim_idx in range(10):
        print('Calibrating simulation', sim_idx)
        sink_arr = np.zeros_like(alpha_arr)
        sub_folder = 'prof_{}_sim_{}_zero_{}'.format(prof_idx, sim_idx, bench_zero)
        sub_cali_dir = '{}/{}'.format(cali_dir, sub_folder)

        if not os.path.exists(sub_cali_dir):
            os.makedirs(sub_cali_dir)

        with open('{}/prof_{}_zero_{}/sim_path_{}.pickle'.format(log_dir, prof_idx, bench_zero, sim_idx), 'rb') as fp:
            sim_path = pickle.load(fp)

        # real_path = np.tile(real_path, (10, 1))
        likelihood = np.zeros((alpha_arr.shape[0], sim_path.shape[0]))


        for alph_idx in range(alpha_arr.shape[0]):
            alpha = alpha_arr[alph_idx]
            sub_f = "sim_{}_alpha_{}".format(sim_idx, alph_idx)
            sub_dir = '{}/prof_{}_zero_{}/{}'.format(log_dir, prof_idx, bench_zero, sub_f)

            with open('{}/ot_pi.pickle'.format(sub_dir), 'rb') as fp:
                pi = pickle.load(fp)

            with open('{}/ot_init.pickle'.format(sub_dir), 'rb') as fp:
                ot_init = pickle.load(fp)

            ######## only keep the top k non-zero probability in each matrix ############
            # the initial matrix
            kept_ = 10
            reg_ot_init = np.zeros_like(ot_init)
            ind = np.unravel_index(np.argsort(ot_init, axis=None), ot_init.shape)
            reg_ot_init[ind[0][-kept_:], ind[1][-kept_:]] = ot_init[ind[0][-kept_:], ind[1][-kept_:]]/ot_init[ind[0][-kept_:], ind[1][-kept_:]].sum()

            # the 1 to time_n matrix, time zero matrix not used
            reg_pi = np.zeros_like(pi)
            for t in range(1, time_horizon):
                for s in range(s_split_n):
                    for w in range(w_split_n):
                        x = pi[t, s, w, :, :]
                        if np.max(x) > 0:
                            ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
                            reg_pi[t, s, w, ind[0][-kept_:], ind[1][-kept_:]] = x[ind[0][-kept_:], ind[1][-kept_:]]/x[ind[0][-kept_:], ind[1][-kept_:]].sum()

            # need path like (scr, wage, scr, wage, ...)
            scr_list = []
            wage_list = []
            conn_list = []
            s, w = np.nonzero(reg_ot_init)
            scr_list.append(s)
            wage_list.append(w)

            for t in range(1, time_horizon):
                c, s, w = np.nonzero(reg_pi[t, scr_list[-1], wage_list[-1]])
                conn_list.append(c.copy())
                scr_list.append(s.copy())
                wage_list.append(w.copy())

            pa_num = len(wage_list[-1])

            prob_list = []
            sample_list = []

            k = 0

            for l in range(pa_num):
                # find path l
                a = np.zeros(time_horizon * 2)
                a[2*time_horizon-1] = wage_list[time_horizon-1][l]
                a[2*time_horizon-2] = scr_list[time_horizon-1][l]
                pre_ = conn_list[time_horizon-1-1][l]
                for t in range(time_horizon-2, -1, -1):
                    a[2*t+1] = wage_list[t][pre_]
                    a[2*t] = scr_list[t][pre_]
                    if t > 0:
                        pre_ = conn_list[t - 1][pre_]

                # calculate prob for path l and see if it's small
                a = a.astype(int)
                prob = reg_ot_init[a[0], a[1]]
                for t in range(1, time_horizon):
                    prob *= reg_pi[t, a[2*(t-1)], a[2*(t-1)+1], a[2*(t-1)+2], a[2*(t-1)+3]]

                # attach path if prob is large enough， 5e-7
                if prob > 1e-7:
                    prob_list.append(prob)
                    sample_list.append(a)

                k += 1
                if k % 1000000 == 0:
                    print(k, 'out of', pa_num, 'is done')

            prob_arr = np.array(prob_list)
            sample = np.array(sample_list)
            # print('prob sum after removing paths <=1e-7', np.sum(prob_arr))
            prob_arr = prob_arr/prob_arr.sum()


            sorted_idx = np.argsort(prob_arr)[::-1]
            # # the threshold index that the sum of prob > 0.3
            # thres_idx = np.argmax(np.cumsum(prob_arr[sorted_idx]) > 0.9)
            keep_idx = sorted_idx[:] # sim_path.shape[0]]
            # # keep_idx = np.argsort(prob_arr)
            # print('prob sum for top paths', np.sum(prob_arr[keep_idx]), len(keep_idx))
            prob_arr = prob_arr[keep_idx]
            sample = sample[keep_idx, :]
            prob_arr = prob_arr/prob_arr.sum()

            if alph_idx in [0, 15, 25, 35, 50]:
                with open('{}/ot_path_{}.pickle'.format(sub_cali_dir, alph_idx), 'wb') as fp:
                    pickle.dump(sample, fp)
                with open('{}/path_prob_{}.pickle'.format(sub_cali_dir, alph_idx), 'wb') as fp:
                    pickle.dump(prob_arr, fp)


            ### calculate values of score and wage in the path
            r_value = np.zeros((sim_path.shape[0], time_horizon*2))
            ot_value = np.zeros((sample.shape[0], time_horizon*2))
            for p in range(sim_path.shape[0]):
                for t in range(time_horizon):
                    r_value[p, 2*t] = sim_path[p, 2*t]/s_split_n
                    r_value[p, 2*t+1] = sim_path[p, 2*t+1]/w_split_n


            for p in range(sim_path.shape[0]):
                # calculate prob for path l and see if it's small
                a = sim_path[p, :].astype(int)
                prob = reg_ot_init[a[0], a[1]]
                for t in range(1, time_horizon):
                    prob *= reg_pi[t, a[2*(t-1)], a[2*(t-1)+1], a[2*(t-1)+2], a[2*(t-1)+3]]
                likelihood[alph_idx, p] = prob

            for p in range(sample.shape[0]):
                for t in range(time_horizon):
                    # it's score
                    ot_value[p, 2*t] = sample[p, 2*t]/s_split_n
                    ot_value[p, 2*t+1] = sample[p, 2*t+1]/w_split_n



            # cost matrix
            real_dim = sim_path.shape[0]
            tr_dim = sample.shape[0]

            # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
            # array along, (you can achieve the same effect by indexing with None, see below)
            # https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times
            RE = np.repeat(r_value[:, np.newaxis, :], tr_dim, axis=1)
            OT = np.repeat(ot_value[np.newaxis, :, :], real_dim, axis=0)

            M = np.sum(np.minimum(np.abs(RE - OT), 5.0), axis = 2)/time_horizon

            real_dist = np.ones(real_dim)/real_dim
            tr_dist = prob_arr

            sink_dist = ot.sinkhorn2(real_dist, tr_dist, M, sink_reg_coef, numItermax=50000)
            # emd_dist = ot.emd2(real_dist, tr_dist, M, numItermax=100000)
            if alph_idx%25 == 0:
                print('Alpha', alpha)
                print('Sinkhorn distance', sink_dist)
                print('Likelihood sum', likelihood[alph_idx, :].sum())
            sink_arr[alph_idx] = sink_dist
            # emd_arr[alph_idx] = emd_dist

        # Save params configuration
        with open('{}/params.txt'.format(sub_cali_dir), 'w') as fp:
            fp.write('sinkhorn coef {} \n'.format(sink_reg_coef))
            fp.write('kept cond prob {} \n'.format(kept_))

        with open('{}/sink_arr.pickle'.format(sub_cali_dir), 'wb') as fp:
            pickle.dump(sink_arr, fp)

        with open('{}/likelihood.pickle'.format(sub_cali_dir), 'wb') as fp:
            pickle.dump(likelihood, fp)
