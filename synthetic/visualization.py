from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

def HGMM(n_class, dim, margin, shift = False):
    margin = margin
    mean = np.zeros((n_class , dim))
    #mean[:(n_class // 2), 0] = margin
    #mean[(n_class // 2):, 0] = -margin
    ratio = n_class // 2
    index = 0
    while ratio != 0:
        for i in range(int(n_class // ratio)):
            mean[i*ratio:(i+1)*ratio, index] = (-1) ** i * margin / (2**index)
        #for i in range(8):
            #mean[i*1:(i+1)*1, 2] = (-1) ** i * margin / 4
        ratio = ratio // 2
        index += 1
    if shift:
        '''
        for i in range(n_class):
            mean[i,i * 10:(i*10 + index)] = mean[i,:index]
            if i != 0:
                mean[i,:index] = 0
        '''
        mean[:n_class//2,index:index + index] = mean[:n_class//2,:index]
        mean[:n_class//2,:index] = 0
    return mean

def gen_synthetic(dim, margin, n_class, var, num =100, shift = False):
    mean = HGMM(n_class, dim, margin, shift)
    data = np.random.multivariate_normal(mean[0], var * np.identity(dim), num)
    cla = np.zeros(num)
    for i in range(1, n_class):
        cla = np.concatenate([cla, i*np.ones(num)])
        data = np.concatenate([data, np.random.multivariate_normal(mean[i], var * np.identity(dim), num)])
    print(data.shape)
    return data, cla
        
viz_synthetic_data2, viz_cla2 = gen_synthetic(100, 8, 8, 1, 100)
z2 = TSNE(n_components=3).fit_transform(viz_synthetic_data2)

mean = []
for i in range(8):
    mean.append([float(sum(l))/len(l) for l in zip(*z2[i*100:(i+1)*100])])
    #mean.append(np.mean(viz_synthetic_data[i*100:(i+1)*100]))
mean = np.array(mean)
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(mean[:, 0], mean[:, 1], mean[:,2], c = np.arange(8), s = np.ones(8) * 200, cmap = "tab20b", alpha = 1)
ax.scatter(z2[:, 0], z2[:, 1], z2[:,2], c = viz_cla2, cmap ="tab20b", alpha = 0.2, marker = "^")
ax.set_xlim3d(-50,50)
ax.set_ylim3d(-50,50)
ax.set_zlim3d(-50,50)
plt.show()