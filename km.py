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
    # weighted version
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


def vnk(m, k, max_iterations=999, init_points_index=None):
    # use vector instead of loop
    n = m.shape[0]
    if init_points_index is None:
        k_ind = np.random.choice(np.arange(n), k)
    else:
        k_ind = np.copy(init_points_index)
    print(k_ind)
    stackm = m[:, np.newaxis, :].repeat(k, 1)
    assignment = np.zeros(n, dtype='int')
    k_ind0 = np.zeros(k)
    for _ in range(max_iterations):
        if np.array_equal(k_ind0, k_ind):
            print("centroids not change")
            break
        assignment = np.argmin(m[k_ind, :], axis=0)
        # ass => n,
        k_ind0 = np.copy(k_ind)

        kd = np.zeros((n, k, n))
        kd[np.arange(n), assignment, :] = stackm[np.arange(n), assignment, :]
        km = kd.sum(axis=0)
        k_ind = np.apply_along_axis(np.argmin, 1, km)
    return (k_ind, assignment)


def vnk_w(m, k, w, max_iterations=999, init_points_index=None):
    # w: the step for distance
    # use vector instead of loop
    n = m.shape[0]
    if init_points_index is None:
        k_ind = np.random.choice(np.arange(n), k)
    else:
        k_ind = np.copy(init_points_index)
    print(k_ind)
    stackm = m[:, np.newaxis, :].repeat(k, 1)
    assignment = np.zeros(n, dtype='int')
    k_ind0 = np.zeros(k)
    for _ in range(max_iterations):
        if np.array_equal(k_ind0, k_ind):
            print("centroids not change")
            break
        assignment = np.argmin(m[k_ind, :], axis=0)
        # ass => n,
        k_ind0 = np.copy(k_ind)

        kd = np.zeros((n, k, n))
        kd[np.arange(n), assignment, :] = stackm[np.arange(n), assignment, :]
        km = (kd**w).sum(axis=0)
        k_ind = np.apply_along_axis(np.argmin, 1, km)
    return (k_ind, assignment)
