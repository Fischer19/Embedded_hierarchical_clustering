from model import *
import numpy as np
from util import *

import argparse

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--n_class', required=False, type=int, default='8', help="number of clusters")
    parser.add_argument('-m', '--margin', required=False, type=float , default='4', help="margin of HGMM")
    parser.add_argument('-v', '--variance', required=False, type=float , default='1', help="variance of HGMM")
    parser.add_argument('-d', '--dim', required=False, type=int, default='100', help="dimension of HGMM")
    parser.add_argument('-hd', '--hidden_dim', required=False, type=int, default=10, help="hidden dimension size for VaDE model")
    parser.add_argument('-s', '--subsampling', required=False, type=int, default=100)
    parser.add_argument('-l', '--linkage_method', required=False, type=str, default="ward")
    args = parser.parse_args()
    
    
    N_CLASS = args.n_class
    MARGIN = args.margin
    VAR = args.variance
    DIM = args.dim
    HID_DIM = args.hidden_dim
    SUBSAMPLE_SIZE = args.subsampling
    N = 2000 # num per class


    #generate synthetic data
    train_loader, synthetic_data, cla = create_data_loader(400, N_CLASS,MARGIN,VAR,DIM,N)
    # train VaDE
    model = VaDE(N_CLASS, HID_DIM, DIM)
    model.pre_train(train_loader,pre_epoch=50)
    train(model, train_loader, 80)
    torch.save(model.state_dict(), "VaDE_parameters_C{}_M{}.pth".format(args.n_class, args.margin))
    #model.load_state_dict(torch.load("VaDE_parameters.pth"))
    # begin evaluation 
    subsample_index = np.arange(100)
    for i in range(1, N_CLASS):
        subsample_index = np.concatenate([subsample_index, i * N + np.arange(SUBSAMPLE_SIZE)])

    mean, _ = model.encoder(torch.from_numpy(synthetic_data).float())
    pca = PCA(n_components = 10)
    projection = pca.fit_transform(synthetic_data)
    print("VaDE:", compute_purity_average(mean.detach().numpy(), cla, 8, 1024, 100, method = args.linkage_method))
    print("PCA:", compute_purity_average(projection, cla, 8, 1024, 100, method = args.linkage_method))
    print("Origin:", compute_purity_average(synthetic_data, cla, 8, 1024, 100, method = args.linkage_method))
    
    print(compute_MW_objective_average(model, mean.detach().numpy(), cla, 1024, 100, method = args.linkage_method))
    print(compute_MW_objective_average(model, projection, cla, 1024, 100, method = args.linkage_method))
    print(compute_MW_objective_average(model, synthetic_data, cla, 1024, 100, method = args.linkage_method))