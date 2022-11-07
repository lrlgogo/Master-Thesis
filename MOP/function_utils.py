import numpy as np


def func_generate_x(x, n_exp, n_r, sigma_list):
    n_x, x_dim = np.shape(x)
    n_s = len(sigma_list)
    len_1 = n_x
    len_2 = n_x * (n_exp - 1) * n_s
    len_3 = n_s * (n_x * n_exp + n_r)
    n1 = len_1
    n2 = n1 + len_2
    n3 = n2 + len_3

    y = np.zeros((n3, x_dim))
    y[: n1] = x

    for p in range(n_s):
        for q in range(n_x):
            num_sta = n1 + p * n_x * (n_exp - 1) + q * (n_exp - 1)
            num_end = num_sta + n_exp - 1
            y[num_sta: num_end, :] = y[q, :] + np.random.normal(0, sigma_list[p], (n_exp - 1, x_dim))
            # y[num_sta: num_end, :] = y[p, :] + np.random.normal(0, sigma_list[q], 1)
        num_sta = n1 + p * n_x * (n_exp - 1)
        num_end = num_sta + n_x * (n_exp - 1)
        num_sta1 = n2 + p * (n_x * n_exp + n_r)
        num_end1 = num_sta1 + n_x * n_exp + n_r
        y[num_sta1: num_sta1 + n_x, :] = y[: n1, :]
        y[num_sta1 + n_x: num_sta1 + n_x * n_exp, :] = y[num_sta: num_end, :]
        y[num_sta1 + n_x * n_exp: num_end1, :] = np.random.uniform(0, 1, (1, x_dim))

        for index_dim in range(x_dim):
            np.random.shuffle(y[num_sta1: num_end1, index_dim])

    return y
