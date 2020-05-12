from model import *
import numpy as np
from util import *

import argparse

import torch
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from torchvision import datasets, transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--method', required=False, type=str, default='ward', help="number of clusters")
    parser.add_argument('-s', '--subsampling', required=False, type=int, default=100)
    args = parser.parse_args()
    
    SUBSAMPLE_SIZE = args.subsampling
    dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor())
    mnist_data, cla = dataset.data.numpy().reshape(-1, 784), dataset.targets.numpy()

    #generate synthetic data
    model = VaDE()
    model.load_state_dict(torch.load("pretrained_parameters/parameters_vade_linear_10classes_mnist.pth", map_location=torch.device('cpu')))
    # begin evaluation 
    print("VaDE transformed:", compute_MW_objective_average(model, mnist_data, cla, 10, 1024, 100, eval = "VaDE",transform=True, VERBOSE = True))
    
    print("VaDE MW:", compute_MW_objective_average(model, mnist_data, cla, 1024, 100, eval = "VaDE", VERBOSE = True))
    print("PCA MW:", compute_MW_objective_average(model, mnist_data, cla, 1024, 100, eval = "PCA", VERBOSE = True))
    print("Origin MW:", compute_MW_objective_average(model, mnist_data, cla, 1024, 100, eval = "Origin", VERBOSE = True))

    print("VaDE transformed DP:", compute_purity_average(model, mnist_data, cla, 10, 1024, 100, eval = "VaDE", transform=True, VERBOSE = True))
    print("VaDE DP:", compute_purity_average(model, mnist_data, cla, 10, 1024, 100, eval = "VaDE", VERBOSE = True))
    print("PCA DP:", compute_purity_average(model, mnist_data, cla, 10, 1024, 100, eval = "PCA", VERBOSE = True))
    print("Origin DP:", compute_purity_average(model, mnist_data, cla, 10, 1024, 100, eval = "Origin", VERBOSE = True))
    
    