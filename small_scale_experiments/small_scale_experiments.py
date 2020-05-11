import numpy as np
import scipy
import numpy as np
import torch

from model import *
from util import *
from sklearn import datasets
import os


device = "cuda"

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
if  os.path.exists('./pretrained_parameters/parameters_glass.pth'):
    model.load_state_dict(torch.load('./pretrained_parameters/parameters_glass.pth'))
else:
    train_loader = []
    for i in range(data.shape[0]):    
        train_loader.append(torch.from_numpy(data[i:i+1]).float())
    model.pre_train(train_loader, 10)
    train(model, train_loader, 100)
    torch.save(model.state_dict(), "pretrained_parameters/parameters_glass.pth")

mean, _ = model.encoder(torch.from_numpy(data).float())
Z_vade = linkage(mean.detach().numpy(), "average")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
print("Dendrogram Purity:", compute_purity(Z_vade, cla, 6))


print("###################EXPERIMENTS ON DIGIT###########################")

data, target = datasets.load_digits(n_class=10, return_X_y=True)
train_loader = []
cla = target
for i in range(data.shape[0]//10):    
    train_loader.append(torch.from_numpy(data[i*10:(i+1)*10]).float())

model = VaDE(10, 10, 64)

if  os.path.exists('./pretrained_parameters/parameters_digits.pth'):
    model.load_state_dict(torch.load('./pretrained_parameters/parameters_digits.pth'))
else:
    model.pre_train(train_loader, 50)
    train(model, train_loader, 100)
    torch.save(model.state_dict(), "pretrained_parameters/parameters_digits.pth")

mean, _ = model.encoder(torch.from_numpy(data).float())
Z_vade = linkage(mean.detach().numpy()[:200], "average")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
print("Dendrogram Purity:", compute_purity(Z_vade, cla[:200], 10))


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
