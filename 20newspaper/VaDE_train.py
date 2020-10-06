import os, time
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
#from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os
from model import *


def get_20newsgroup(data_dir, batch_size=128, device = "cuda"):
    with open(data_dir, "rb") as f:
    	dic = pickle.load(f)
    	X = dic["X"]
    	y = dic["y"]
    train_loader = []
    X.to(device)
    y.to(device)
    for i in range(len(X) // batch_size):
    	train_loader.append([X[i*batch_size: (i+1) * batch_size].float(), y[i*batch_size:(i+1)*batch_size]])
    return train_loader, batch_size

if __name__ == "__main__":
    device = "cuda"
    nClusters = 20
    hid_dim = 10
    DL,_=get_20newsgroup("tfidf_embedding.pk",batch_size = 128)

    vade=VaDE(nClusters,hid_dim,2000).to(device)
    
    vade.pre_train(DL,pre_epoch=50)
    # Re-initialize the weights (NaN occurs in loss otherwise)
    torch.nn.init.zeros_(vade.encoder.log_sigma2_l.weight)
    opti=Adam(vade.parameters(),lr=2e-3)
    lr_s=StepLR(opti,step_size=10,gamma=0.95)


    epoch_bar=tqdm(range(300))
    tsne=TSNE()
    

    for epoch in epoch_bar:

        lr_s.step()
        L=0
        for x,_ in DL:
            x=x.cuda()

            loss=vade.ELBO_Loss(x)

            opti.zero_grad()
            loss.backward()
            opti.step()

            L+=loss.detach().cpu().numpy()


        pre=[]
        tru=[]

        with torch.no_grad():
            for x, y in DL:
                x = x.cuda()

                tru.append(y.numpy())
                pre.append(vade.predict(x))


        tru=np.concatenate(tru,0)
        pre=np.concatenate(pre,0)
        
        ACC = cluster_acc(pre,tru)[0]*100

        epoch_bar.write('Loss={:.4f},ACC={:.4f}%,LR={:.4f}'.format(L/len(DL),ACC,lr_s.get_lr()[0]))
    torch.save(vade.module.state_dict(), "parameters/VaDE_parameters_h{}_c{}.pth".format(hid_dim, nClusters))
