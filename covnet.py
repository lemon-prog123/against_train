import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_and_linear import Conv
from conv_and_linear import Linear
from P_relu import PWLA3d
from P_relu import PWLA1d
from util import setparams
class ConvNet(nn.Module):
    def __init__(self, quant=True):
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
    def activation(self):
        self.mode=1
        setparams(self.relu1)
        setparams(self.relu2)
        setparams(self.relu3)
        setparams(self.relu4)
        setparams(self.relu5)
        setparams(self.relu6)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        #x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(self.relu1(self.conv1(x),self.mode), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(self.relu2(self.conv2(x),self.mode), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(self.relu3(self.conv3(x),self.mode), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)
        x = self.relu4(self.fc1(x),self.mode)
        #x=F.relu(self.fc1(x))

        x = self.relu5(self.fc2(x),self.mode)
        #x=F.relu(self.fc2(x))
        x = self.relu6(self.fc3(x),self.mode)
        #x=F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    # TODO:
    def regularizationTerm(self, reg_type):
        """

        """
        term = 0.0
        if reg_type == "orthogonal":
            pass
        elif reg_type == "spectral":
            pass
        else:
            raise NotImplementedError
        return term