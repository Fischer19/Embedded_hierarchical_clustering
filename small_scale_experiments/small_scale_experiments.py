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

with open("data/optdigits.tes") as f:
    raw_data = f.readlines()
data = []
cla = []
for i in range(len(raw_data)):
    line = raw_data[i].split(',')
    data.append(line[:-1])
    cla.append(int(line[-1]))
with open("data/optdigits.tra") as f:
    raw_data = f.readlines()
for i in range(len(raw_data)):
    line = raw_data[i].split(',')
    data.append(line[:-1])
    cla.append(int(line[-1]))    
data = np.array(data).astype(np.float)

train_loader = []
for i in range(data.shape[0]//500):  
    train_loader.append(torch.from_numpy(data[i*500:(i+1)*500]).float())


cla = np.array(cla)

model = VaDE(10, 10, 64, [50,50,100])
pca = PCA(n_components = 10)

vae = VAE(64,10)
#train_vae(vae, train_loader, 500, lr = 5e-4)
#torch.save(vae.state_dict(), "pretrained_parameters/VAE_parameters.pth")
vae.load_state_dict(torch.load("pretrained_parameters/VAE_parameters.pth"))

if  os.path.exists('./pretrained_parameters/parameters_digits.pth'):
    model.load_state_dict(torch.load('./pretrained_parameters/parameters_digits.pth'))
else:
    
    model.pre_train(train_loader, 50)
    train(model, train_loader, 200)
    torch.save(model.state_dict(), "pretrained_parameters/parameters_digits.pth")

#train(model, train_loader, 100, lr = 5e-4)    

mean, _ = model.encoder(torch.from_numpy(data).float())
_, vae_mean, _ = vae(torch.from_numpy(data).float())
scaled_mean = transformation(model, data, 3)
projection = pca.fit_transform(data)
methods_list = ["average", "centroid", "complete", "single", "ward"]
index = np.arange(200)
'''
for i in range(25):
    index = np.arange(i * 200, (i+1) * 200)
    
    Z_vade = linkage(mean.detach().numpy()[index], "ward")
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("VaDE Dendrogram Purity :", compute_purity(Z_vade, cla[index], 10))
    
    Z_pca = linkage(projection[index], "ward")
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("PCA Dendrogram Purity :", compute_purity(Z_pca, cla[index], 10))
'''

for method in methods_list:
    
    Z_vae = linkage(vae_mean.detach().numpy()[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vae, rd=True)
    print("VAE Dendrogram Purity " + method + ":", compute_purity(Z_vae, cla[index], 10))
    
    Z_vade = linkage(mean.detach().numpy()[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("VaDE Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla[index], 10))
    
    Z_pca = linkage(projection[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("PCA Dendrogram Purity " + method + ":", compute_purity(Z_pca, cla[index], 10))
    
    Z_pca = linkage(scaled_mean[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("Trans Dendrogram Purity " + method + ":", compute_purity(Z_pca, cla[index], 10))

    Z = linkage(cla[index].reshape(-1,1), method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    max = compute_objective_gt(200, rootnode, cla[index])
    
    Z = linkage(vae_mean.detach().numpy()[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("VAE MW:", compute_objective_gt(200, rootnode, cla[index]) / max)
    
    Z = linkage(mean.detach().numpy()[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("VaDE MW:", compute_objective_gt(200, rootnode, cla[index]) / max)
    
    Z = linkage(projection[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("PCA MW:", compute_objective_gt(200, rootnode, cla[index]) / max)
    
    Z = linkage(scaled_mean[index], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("Trans MW:", compute_objective_gt(200, rootnode, cla[index]) / max)


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
    
    Z_vade = linkage(data, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("Origin Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))
    
    Z_vade = linkage(mean.detach().numpy(), method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("VaDE Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))
    
    Z_vade = linkage(projection, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("PCA Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))
    
    Z_vade = linkage(scaled_mean, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("Trans Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla, 6))
    
    Z = linkage(cla.reshape(-1,1), method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    max = compute_objective_gt(216, rootnode, cla)
    Z = linkage(data, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("Origin MW:", compute_objective_gt(216, rootnode, cla) / max)
    
    Z = linkage(mean.detach().numpy(), method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("VaDE MW:", compute_objective_gt(216, rootnode, cla) / max)
    
    Z = linkage(projection, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("PCA MW:", compute_objective_gt(216, rootnode, cla) / max)
    
    Z = linkage(scaled_mean, method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("Trans MW:", compute_objective_gt(216, rootnode, cla) / max)

"""
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
"""