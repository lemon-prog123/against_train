import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.conv_and_linear import Conv
from net.conv_and_linear import Linear
from pwla_relu.P_relu import PWLA3d
from pwla_relu.P_relu import PWLA1d
from util.utils import setparams
class ConvNet(nn.Module):
    def __init__(self, quant=False):
        super(ConvNet, self).__init__()
        if quant==True:
            self.conv1 = Conv(3, 15,3)    
            self.conv2 = Conv(15, 75,4)   
            self.conv3 = Conv(75,375,3)    
            self.fc1 = Linear(1500,400)    
            self.fc2 = Linear(400,120)     
            self.fc3 = Linear(120, 84)      
            self.fc4 = Linear(84, 10)
        else:
            self.conv1 = nn.Conv2d(3, 15,3)    
            self.conv2 = nn.Conv2d(15, 75,4) 
            self.conv3 = nn.Conv2d(75,375,3)    
            self.fc1 =nn.Linear(1500,400)    
            self.fc2 = nn.Linear(400,120)        
            self.fc3 = nn.Linear(120, 84)       
            self.fc4 = nn.Linear(84, 10)
        self.relu1=PWLA3d()
        self.relu2=PWLA3d()
        self.relu3=PWLA3d()
        self.relu4=PWLA1d()
        self.relu5=PWLA1d()
        self.relu6=PWLA1d()
        self.mode=0
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def activation(self):
        self.mode=1
        setparams(self.relu1)
        setparams(self.relu2)
        setparams(self.relu3)
        setparams(self.relu4)
        setparams(self.relu5)
        setparams(self.relu6)
    def reconstruct(self):
        self.conv1.reconstruct()
        self.conv2.reconstruct()
        self.conv3.reconstruct()
        self.fc1.reconstruct()
        self.fc2.reconstruct()
        self.fc3.reconstruct()
        self.fc4.reconstruct()
    def forward(self, x):
        """
        """
        x = F.max_pool2d(self.relu1(self.conv1(x),self.mode), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(self.relu2(self.conv2(x),self.mode), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(self.relu3(self.conv3(x),self.mode), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)
        x = self.relu4(self.fc1(x),self.mode)
        x = self.relu5(self.fc2(x),self.mode)
        x = self.relu6(self.fc3(x),self.mode)
        x = self.fc4(x)
        return x

    # TODO:
    def regularizationTerm(self,beta=1e-2,mode=0):
        loss = torch.zeros(1).to(self.device)
        if mode == 0:
            for layer in self.modules():
                if type(layer) == Linear:
                    w = layer._parameters[ 'weight' ]
                    m = w @ w.T
                    loss += torch.norm(m - torch.eye(m.shape[ 0 ]))
                if type(layer) == Conv:
                    w = layer._parameters[ 'weight' ]
                    N, C, H, W = w.shape
                    w = w.view(N * C, H, W)
                    m = torch.bmm(w, w.permute(0, 2, 1))
                    loss += torch.norm(m - torch.eye(H))
        else:
            for layer in self.modules():
                if type(layer) == Linear:
                    iteration = 1
                    w = layer._parameters[ 'weight' ]
                    v = layer.v
                    u = torch.normal(0, 1, size=(w.shape[ 0 ], 1)).to(self.device)
                    for _ in range(iteration):
                        u = torch.nn.functional.normalize(torch.mm(w.detach(), v), dim=0)
                        v = torch.nn.functional.normalize(torch.mm(w.detach().T, u), dim=0)
                    layer.v = v
                    loss += (u.T @ w @ v).view(1)
                elif type(layer) == Conv:
                    iteration = 1
                    w = layer._parameters[ 'weight' ]
                    v = layer.v
                    N, C, H, W = w.shape
                    m = w.view(N, C * W * H)
                    u = torch.normal(0, 1, size=(m.shape[ 0 ], 1)).to(self.device)
                    for _ in range(iteration):
                        u = torch.nn.functional.normalize(torch.mm(m.detach(), v), dim=0)
                        v = torch.nn.functional.normalize(torch.mm(m.detach().T, u), dim=0)
                    layer.v = v
                    loss += (u.T @ m @ v).view(1)
        return beta * loss