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
    parser = argparse.ArgumentParser(description='Experiment for graph classification with coarse graphs')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                            help='name of dataset (default: MUTAG)')
    parser.add_argument('--method', type=str, default="mgc",
                            help='name of the coarsening method')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='the ratio between coarse and original graphs n/N')
    args = parser.parse_args()
    if args.dataset not in ["MUTAG", "ENZYMES", "NCI1", "NCI109", "PROTEINS", "PTC"]:
        print("Incorrect input dataset")
        sys.exit()
    if args.method not in ['mgc', 'sgc']:
        print("Incorrect input coarsening method")
        sys.exit()
    if args.ratio < 0 or args.ratio > 1:
        print("Incorrect input ratio")
        sys.exit()
    dir = 'dataset'
    am, labels = parse.parse_dataset(dir, args.dataset)
    num_samples = len(am)
    X = np.zeros((num_samples, 250))
    Y = labels
    for i in range(num_samples):
        N = am[i].shape[0]
        n = int(np.ceil(args.ratio*N))
        if args.method == "mgc":
            coarse_method = coarsening.multilevel_graph_coarsening
        else:
            coarse_method = coarsening.spectral_graph_coarsening
        if n > 1:
            Gc, Q, idx = coarse_method(am[i], n)
        else:
            Gc, Q, idx = coarse_method(am[i], 1)
        G = nx.from_numpy_matrix(Gc)
        X[i] = netlsd.heat(G)
    acc, std = classification.KNN_classifier_kfold(X, Y, 10)



if __name__ == '__main__':
    main()
