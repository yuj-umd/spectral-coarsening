import random
import numpy as np
import numpy.linalg as LA
import networkx as nx
import util
import measure
import laplacian
from sklearn.cluster import KMeans

def multilevel_graph_coarsening(G, n):
    N = G.shape[0]
    Gc = G
    cur_size = N
    Q = np.eye(N)
    while cur_size > n:
        stop_flag = 0
        max_dist = 10000
        max_dist_a = -1
        max_dist_b = -1
        for i in range(cur_size):
            for j in range(i+1, cur_size):
                dist = measure.normalized_L1(Gc[i], Gc[j])
                if dist < max_dist:
                    max_dist = dist
                    max_dist_a = i
                    max_dist_b = j
                if dist < 0.001:
                    stop_flag = 1
                    break
            if stop_flag == 1:
                break
        if max_dist_a == -1:
            max_dist_a, max_dist_b = util.random_two_nodes(cur_size)
        cur_Q = util.merge_two_nodes(cur_size, max_dist_a, max_dist_b)
        Q = np.dot(Q, cur_Q)
        Gc = util.multiply_Q(Gc, cur_Q)
        cur_size = cur_size - 1
    idx = util.Q2idx(Q)
    return Gc, Q, idx

def regular_partition(N, n):
    if N%n == 0:
        block_size = N//n + 1
    else:
        block_size = N//(n-1) + 1
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        idx[i] = i//block_size
    return idx


def spectral_graph_coarsening(G, n):
    N = G.shape[0]
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n+1

    for k in range(0, n):
        if e1[k] <= 1:
            if k+1 < n and e2[k+1] < 1:
                continue
            if k+1 < n and e2[k+1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k+1)], v2[:, (k+1):n]), axis = 1)
            elif k == n-1:
                v_all = v1[:, 0:n]

            kmeans = KMeans(n_clusters=n).fit(v_all)
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = util.idx2Q(idx, n)
            Gc = util.multiply_Q(G, Q)
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q
    return Gc_min, Q_min, idx_min


def spectral_graph_coarsening_lambda(G, n):
    N = G.shape[0]
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n+1

    for k in range(0, n):
        if e1[k] <= 1:
            if k+1 < n and e2[k+1] < 1:
                continue
            if k+1 < n and e2[k+1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k+1)], v2[:, (k+1):n]), axis = 1)
            elif k == n-1:
                v_all = v1[:, 0:n]

            kmeans = KMeans(n_clusters=n).fit(v_all)
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = util.idx2Q(idx, n)
            Gc = util.multiply_Q(G, Q)
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q
    return Gc_min, Q_min, idx_min

def spectral_clustering(G, n):
    N = G.shape[0]
    e1, v1 = laplacian.spectraLaplacian_top_n(G, n)
    v_all = v1[:, 0:n]
    kmeans = KMeans(n_clusters=n).fit(v_all)
    idx = kmeans.labels_
    sumd = kmeans.inertia_
    Q = util.idx2Q(idx, n)
    Gc = util.multiply_Q(G, Q)
    return Gc, Q, idx

def get_random_partition(N, n):
    for i in range(500):
        flag = True
        a = np.zeros(N, dtype=np.int64)
        cnt = np.zeros(n, dtype=np.int64)
        for j in range(N):
            a[j] = random.randint(0, n-1)
            cnt[a[j]] += 1
        for j in range(n):
            if cnt[j] == 0:
                flag = False
                break
        if flag == False:
            continue
        else:
            break
    return a
