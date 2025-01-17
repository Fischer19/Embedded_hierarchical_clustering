from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import itertools
from tqdm import tqdm
import os

def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2000, 500)
        self.fc11 = nn.Linear(500, 500)
        self.fc21 = nn.Linear(500, 10)
        self.fc22 = nn.Linear(500, 10)
        self.fc3 = nn.Linear(10, 500)
        self.fc31 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 2000)

    def encode(self, x):
        h1 = F.relu(self.fc11(F.relu(self.fc1(x))))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc31(F.relu(self.fc3(z))))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2000))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VaDE(nn.Module):
    def __init__(self, nClusters = 20, hid_dim = 10, input_dim = 2000, device = "cuda"):
        super(VaDE,self).__init__()
        self.encoder=Encoder(input_dim, hid_dim=hid_dim)
        self.decoder=Decoder(input_dim, hid_dim=hid_dim)
        self.nClusters = nClusters
        self.device = device

        self.pi_=nn.Parameter(torch.FloatTensor(nClusters,).fill_(1)/nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(nClusters,hid_dim).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(nClusters,hid_dim).fill_(0),requires_grad=True)


    def pre_train(self,dataloader,pre_epoch=10):

        if  not os.path.exists('./pretrain_model_20newsgroup.pk'):

            Loss=nn.MSELoss()
            opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    if self.device == "cuda":
                        x = x.cuda()
                    z,_=self.encoder(x)
                    x_=self.decoder(z)
                    loss=Loss(x,x_)

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            Z = []
            Y = []
            with torch.no_grad():
                for x, y in dataloader:
                    if self.device == "cuda":
                        x = x.cuda()
                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()

            gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

            torch.save(self.state_dict(), './pretrain_model_20newsgroup.pk')

        else:


            self.load_state_dict(torch.load('./pretrain_model_20newsgroup.pk'))




    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_ * (self.pi_ > 0) + 1e-5
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    def ELBO_Loss(self,x,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z)

            L_rec += F.mse_loss(x_pro, x)
            #L_rec+=F.binary_cross_entropy(x_pro,x)

        L_rec/=L

        Loss=L_rec*x.size(1)
        #print("Loss:", Loss)
        #self.pi_[self.pi_ <= 0] = 1e-5
        pi=self.pi_ * (self.pi_ > 0) + 1e-5
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c
        #print(pi)

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters
        #print(yita_c)
        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss






    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)




    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))


def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers


class Decoder(nn.Module):
    def __init__(self,input_dim=2000,inter_dims=[200,500,1000],hid_dim=20):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(hid_dim,inter_dims[-1]),
            *block(inter_dims[-1],inter_dims[-2]),
            *block(inter_dims[-2],inter_dims[-3]),
            nn.Linear(inter_dims[-3],input_dim),
            #nn.Sigmoid()
        )

    def forward(self, z):
        x_pro=self.decoder(z)

        return x_pro

class Encoder(nn.Module):
    def __init__(self,input_dim=2000,inter_dims=[200,500,1000],hid_dim=20):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l=nn.Linear(inter_dims[-1],hid_dim)
        self.log_sigma2_l=nn.Linear(inter_dims[-1],hid_dim)

    def forward(self, x):
        e=self.encoder(x)
        mu=self.mu_l(e)
        log_sigma2=self.log_sigma2_l(e)

        return mu,log_sigma2
