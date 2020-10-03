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

class Decoder(nn.Module):
    # initializers
    def __init__(self, d=128, hid_dim = 10):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.deconv1 = nn.ConvTranspose2d(hid_dim, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, 1, 4, 2, 1)
        #self.deconv4_bn = nn.BatchNorm2d(d)
        #self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1) # 1 4 2 1

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        print("Decoder")
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        print(x.shape)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        print(x.shape)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        print(x.shape)
        #x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #print(x.shape)
        x = torch.sigmoid(self.deconv4(x))
        print(x.shape)
        return x

class Encoder(nn.Module):
    # initializers
    def __init__(self, d=128, hid_dim = 10):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.d = d
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, d*8, 2, 1, 0) # 4 1 0
        
        self.mu_l = nn.Linear(d*8, hid_dim)
        self.log_sigma2_l = nn.Linear(d*8, hid_dim)
        #self.mu_l=nn.Linear(d*8,hid_dim)
        #self.log_sigma2_l=nn.Linear(d*8,hid_dim)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        print("Encoder:")
        x = F.leaky_relu(self.conv1(input), 0.2)
        print(x.shape)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        print(x.shape)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        print(x.shape)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        print(x.shape)
        x = F.leaky_relu(self.conv5(x), 0.2)
        print(x.shape)
        mu=self.mu_l(x.view(-1, self.d * 8))
        log_sigma2=self.log_sigma2_l(x.view(-1, self.d * 8))
        #print(mu.shape, log_sigma2.shape)

        return mu,log_sigma2


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

class VaDE(nn.Module):
    def __init__(self,nClusters, hid_dim, input_dim, use_cuda = True):
        super(VaDE,self).__init__()
        self.nClusters = nClusters
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        self.use_cuda = use_cuda
        self.encoder=Encoder(hid_dim = hid_dim)
        self.decoder=Decoder(hid_dim = hid_dim)

        self.pi_=nn.Parameter(torch.FloatTensor(self.nClusters,).fill_(1)/self.nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(self.nClusters,self.hid_dim).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(self.nClusters,self.hid_dim).fill_(0),requires_grad=True)




    def pre_train(self,dataloader,pre_epoch=10):
        if  not os.path.exists('./pretrain_model_cifar100.pk'):

            Loss=nn.MSELoss()
            opti=torch.optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    if self.use_cuda:
                        x=x.cuda()

                    z,_=self.encoder(x.view(-1, 1, self.input_dim, self.input_dim))
                    x_=self.decoder(z.view(-1, self.hid_dim, 1,1))
                    loss=Loss(x.view(-1, 64 * 64),x_.view(-1, 64 * 64))

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
                    if self.use_cuda:
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

            if self.use_cuda:
                self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
                self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
                self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
            else:
                self.pi_.data = torch.from_numpy(gmm.weights_).float()
                self.mu_c.data = torch.from_numpy(gmm.means_).float()
                self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).float())
                
            torch.save(self.state_dict(), './pretrain_model_cifar100.pk')
        else:
            self.load_state_dict(torch.load('./pretrain_model_cifar100.pk'))
            


    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    def ELBO_Loss(self,x,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.encoder(x.view(-1, 1, self.input_dim, self.input_dim))
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z.view(-1, self.hid_dim, 1,1))

            L_rec+=F.binary_cross_entropy(x_pro.view(-1, 64 * 64),x.view(-1, 64 * 64))

        L_rec/=L

        Loss=L_rec*x.size(1)

        pi=self.pi_
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

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

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
import pickle
import torchvision

def get_cifar100(data_dir='./data/mnist/',batch_size=128):
    img_size = 64
    transform = transforms.Compose([
            transforms.Scale(img_size),
            transforms.ToTensor()
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    return train_loader, img_size

if __name__ == "__main__":
    device = "cuda"
    nClusters = 10
    hid_dim = 10
    DL,_=get_cifar100(batch_size = 128)

    vade=VaDE(nClusters,hid_dim,64,True).to(device)
    
    vade.pre_train(DL,pre_epoch=50)
    # Re-initialize the weights (NaN occurs in loss otherwise)
    torch.nn.init.xavier_uniform_(vade.encoder.log_sigma2_l.weight)
    opti=Adam(vade.parameters(),lr=5e-4)
    lr_s=StepLR(opti,step_size=10,gamma=0.95)


    epoch_bar=tqdm(range(300))
    tsne=TSNE()
    
    writer = SummaryWriter()

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
        writer.add_scalar('Loss/train', L / len(DL), epoch)
        writer.add_scalar('Accuracy/train', ACC, epoch)
    torch.save(vade.module.state_dict(), "parameters/VaDE_parameters_h{}_c{}.pth".format(hid_dim, nClusters))

