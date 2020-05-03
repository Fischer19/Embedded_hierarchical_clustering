from model import *
import numpy as np
from util import *

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--n_class', required=False, type=int, default='8', help="number of clusters")
    parser.add_argument('-m', '--margin', required=False, type=float , default='4', help="margin of HGMM")
    parser.add_argument('-v', '--variance', required=False, type=float , default='1', help="variance of HGMM")
    parser.add_argument('-d', '--dim', required=False, type=int, default='100', help="dimension of HGMM")
    args = parse.parse_args()
    
    
    N_CLASS = args.n_class
    MARGIN = args.margin
    VAR = args.variance
    DIM = args.dim

    train_loaderm, synthetic_data, cla = create_data_loader(400, N_CLASS,MARGIN,VAR,DIM,2000)