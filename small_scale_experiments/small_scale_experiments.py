import numpy as np
import scipy
import numpy as np
import torch

from model import *
from util import *
from sklearn import datasets
import os
from sklearn.decomposition import PCA


device = "cuda"

def transformation(model, data, rate = 2, cla = None):
    mean, _ = model.encoder(torch.from_numpy(data).float())
    pred = model.predict(torch.from_numpy(data).float())
    cluster_means = model.mu_c[pred]
    scaled_cluster_means = cluster_means * rate
    scaled_mean = (mean - cluster_means) + scaled_cluster_means
    return scaled_mean.detach()


print("###################EXPERIMENTS ON DIGIT###########################")

data, target = datasets.load_digits(n_class=10, return_X_y=True)
train_loader = []
cla = target
for i in range(data.shape[0]):    
    train_loader.append(torch.from_numpy(data[i:(i+1)]).float())

model = VaDE(10, 5, 64)
pca = PCA(n_components = 5)


if  os.path.exists('./pretrained_parameters/parameters_digits.pth'):
    model.load_state_dict(torch.load('./pretrained_parameters/parameters_digits.pth'))
else:

    model.pre_train(train_loader, 50)
    train(model, train_loader, 200)
    torch.save(model.state_dict(), "pretrained_parameters/parameters_digits.pth")

mean, _ = model.encoder(torch.from_numpy(data).float())
scaled_mean = transformation(model, data, 3)
projection = pca.fit_transform(data)
methods_list = ["average", "centroid", "complete", "single", "ward"]

"""
for method in methods_list:
    Z_vade = linkage(mean.detach().numpy()[:1000], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla[:1000], 10))
    
    Z_pca = linkage(projection[:1000], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("PCA Dendrogram Purity " + method + ":", compute_purity(Z_pca, cla[:1000], 10))
    
    Z_pca = linkage(scaled_mean[:1000], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("Trans Dendrogram Purity " + method + ":", compute_purity(Z_pca, cla[:1000], 10))
"""
for method in methods_list:
    print(method)
    print("Transform:", compute_purity_average(scaled_mean.detach().numpy(), cla, 10, 1000, 50, method = method))
    print("VaDE:", compute_purity_average(mean.detach().numpy(), cla, 10, 1000, 50, method = method))
    print("PCA:", compute_purity_average(projection, cla, 10, 1000, 50, method = method))
    print("Origin:", compute_purity_average(data, cla, 10, 1000, 50, method = method))
    
    print(compute_MW_objective_average(model, scaled_mean.detach().numpy(), cla, 1000, 50, method = method))
    print(compute_MW_objective_average(model, mean.detach().numpy(), cla, 1000, 50, method = method))
    print(compute_MW_objective_average(model, projection, cla, 1000, 50, method = method))
    print(compute_MW_objective_average(model, data, cla, 1000, 50, method = method))

'''
print("###################EXPERIMENTS ON GLASS##########################")
with open("data/glass.data") as f:
    raw_data = f.readlines()
data = []
target = []
for i in range(len(raw_data)):
    line = raw_data[i].split(',')
    data.append(line[1:-1])
    target.append(int(line[-1]))
data = np.array(data).astype(np.float)
target = np.array(target)
cla = target
model = VaDE(6, 3, 9)
pca = PCA(n_components = 3)

if  os.path.exists('./pretrained_parameters/parameters_glass.pth'):
    model.load_state_dict(torch.load('./pretrained_parameters/parameters_glass.pth'))
else:

    train_loader = []
    for i in range(data.shape[0]):    
        train_loader.append(torch.from_numpy(data[i:i+1]).float())
    model.pre_train(train_loader, 10)
    train(model, train_loader, 100)
    torch.save(model.state_dict(), "pretrained_parameters/parameters_glass.pth")

methods_list = ["average", "centroid", "complete", "single", "ward"]
mean, _ = model.encoder(torch.from_numpy(data).float())
scaled_mean = transformation(model, data, 3)
projection = pca.fit_transform(data)
for method in methods_list:
    print(method)
    
    Z_vade = linkage(mean.detach().numpy(), method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))
    
    Z_vade = linkage(projection, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("PCA Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))
    
    Z_vade = linkage(scaled_mean, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("Trans Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))

'''
'''
print("###################EXPERIMENTS ON COVERTYPE#########################")
cover_data, cover_targets = datasets.fetch_covtype(data_home=None, download_if_missing=True, random_state=None, shuffle=False, return_X_y=True)
print(cover_data.shape, cover_targets.shape)
train_loader = []
cla = cover_targets
for i in range(cover_data.shape[0]//100):    
    train_loader.append(torch.from_numpy(cover_data[i*100:(i+1)*100]).float().to(device))

model = VaDE(7, 10, 54).to(device)
model.pre_train(train_loader, 50)
train(model, train_loader, 100)
torch.save(model.state_dict(), "covertype_VaDE_parameters.pth")
'''