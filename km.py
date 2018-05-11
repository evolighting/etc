import numpy as np
"""
use distense matrix to perform kmean
A friendly implementation
"""


def nk(m, k, max_iterations=99, init_points_index=None):
    n = m.shape[0]
    if init_points_index is None:
        k_ind = np.random.choice(np.arange(n), k)
    else:
        # copy instead of borrow ref
        k_ind = np.copy(init_points_index)
    print(k_ind)
    assignment = np.zeros(n, dtype='int')
    k_ind0 = np.zeros(k)
    for t in range(max_iterations):
        if np.array_equal(k_ind0, k_ind):
            print("centroids not change")
            break

        dist = m[k_ind, :]
        # dist => k,n
        assignment = np.argmin(dist, axis=0)
        k_ind0 = np.copy(k_ind)
        # ass => n,
        for i in np.arange(k):
            nk_m = assignment == i
            nk = m[nk_m, :][:, nk_m]
            min_nk = np.argmin(nk.sum(axis=0))
            k_ind[i] = np.arange(n)[nk_m][min_nk]

    return (k_ind, assignment)


def nk_w(m, k, max_iterations=99, init_points_index=None):
    n = m.shape[0]
    if init_points_index is None:
        print("no init point provided, random pick")
        k_ind = np.random.choice(np.arange(n), k)
    else:
        k_ind = np.copy(init_points_index)
    print(k_ind)
    assignment = np.zeros(n, dtype='int')
    k_ind0 = np.zeros(k)
    for t in range(max_iterations):
        if np.array_equal(k_ind0, k_ind):
            print("centroids not change")
            break

        dist = m[k_ind, :]
        # dist => k,n
        assignment = np.argmin(dist, axis=0)
        k_ind0 = np.copy(k_ind)
        # ass => n,
        for i in np.arange(k):
            nk_m = assignment == i
            # square the distence so that outlier get larger distence
            nk = (m[nk_m, :][:, nk_m])**2
            min_nk = np.argmin(nk.sum(axis=0))
            k_ind[i] = np.arange(n)[nk_m][min_nk]

    return (k_ind, assignment)
