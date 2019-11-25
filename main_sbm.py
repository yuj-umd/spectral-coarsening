import random_graph
import laplacian
import argparse
import numpy as np
import measure
import coarsening
import util
import networkx as nx
import netlsd
import parse
import classification
import os
import sys



def main():
    parser = argparse.ArgumentParser(description='Experiment for block recovery of stochastic block model')
    parser.add_argument('--sbm_type', type=str, default="associative",
                            help='type of stochastic block model (default: associative)')
    parser.add_argument('--method', type=str, default="mgc",
                            help='name of the coarsening method')
    parser.add_argument('--N', type=int, default=200,
                        help='node size of original graphs')
    parser.add_argument('--n', type=int, default=10,
                        help='node size of coarse graphs')
    parser.add_argument('--p', type=float, default=0.5,
                        help='edge probability between nodes within the same blocks')
    parser.add_argument('--q', type=float, default=0.1,
                        help='edge probability between nodes in different blocks')
    parser.add_argument('--max_trials', type=int, default=10,
                        help='number of repeated trials')
    args = parser.parse_args()
    if args.sbm_type not in ['associative', 'dissociative', 'mixed']:
        print("Incorrect input stochastic method")
        sys.exit()
    if args.method not in ['mgc', 'sgc']:
        print("Incorrect input coarsening method")
        sys.exit()
    if args.N < 0 or args.n < 0:
        print("Incorrect node size")
        sys.exit()
    if args.p < 0 or args.p > 1:
        print("Incorrect edge probability")
        sys.exit()
    if args.q < 0 or args.q > 1:
        print("Incorrect edge probability")
        sys.exit()

    ground_truth = coarsening.regular_partition(args.N, args.n)
    if args.sbm_type == "associative":
        sbm_type = random_graph.sbm_pq
    elif args.sbm_type == "dissociative":
        sbm_type = random_graph.sbm_qp
    else:
        sbm_type = random_graph.sbm_pq_mixed
    if args.method == "mgc":
        coarse_method = coarsening.multilevel_graph_coarsening
    else:
        coarse_method = coarsening.spectral_graph_coarsening
    nmi_result = np.zeros(args.max_trials)
    for i in range(args.max_trials):
        G = sbm_type(args.N, args.n, args.p, args.q)
        Gc, Q, idx = coarse_method(G, args.n)
        nmi_result[i] = measure.NMI(idx, ground_truth, args.n)
    print("Average NMI result is %.4f"%(np.mean(nmi_result)))





if __name__ == '__main__':
    main()
