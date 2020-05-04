import numpy as np
import scipy
import numpy as np
import torch

from model import *
from util import *
from sklearn import datasets


device = "cuda"
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

print("###################EXPERIMENTS ON DIGIT###########################")

data, target = datasets.load_digits(n_class=10, return_X_y=True)
train_loader = []
cla = target
for i in range(data.shape[0]//10):    
    train_loader.append(torch.from_numpy(data[i*10:(i+1)*10]).float())

model = VaDE(10, 3, 64)
model.pre_train(train_loader, 50)
train(model, train_loader, 100)

Z_vade = linkage(mean.detach().numpy()[:200], "average")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
print("Dendrogram Purity:", compute_purity(Z_vade, cla[:200], 10))


print("###################EXPERIMENTS ON GLASS##########################")