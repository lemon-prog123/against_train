import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from re_utils import Conv
from re_utils import Linear
class ConvNet(nn.Module):
    def __init__(self, **kwargs):
        super(ConvNet, self).__init__()
        self.conv1 = Conv(3, 15,3)     # 输入3通道，输�?15通道，卷积核�?3*3
        self.conv2 = Conv(15, 75,4)    # 输入15通道，输�?75通道，卷积核�?4*4
        self.conv3 = Conv(75,375,3)    # 输入75通道，输�?375通道，卷积核�?3*3
        self.fc1 = Linear(1500,400)       # 输入2000，输�?400
        self.fc2 = Linear(400,120)        # 输入400，输�?120
        self.fc3 = Linear(120, 84)        # 输入120，输�?84
        self.fc4 = Linear(84, 10)         # 输入 84，输�? 10（分10类）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def regularizationTerm(self,beta=1e-2,mode=0):
        loss=torch.zeros(1).to(self.device)
        if mode==0:
            for layer in self.modules():
                if type(layer)==Linear:
                    w=layer._parameters['weight']
                    m=w@w.T
                    loss+=torch.norm(m-torch.eye(m.shape[0]))
                if type(layer)==Conv:
                    w=layer._parameters['weight']
                    N,C,H,W=w.shape
                    w=w.view(N*C,H,W)
                    m=torch.bmm(w,w.permute(0,2,1))
                    loss+=torch.norm(m-torch.eye(H))
        else:
            for layer in self.modules():
                if type(layer)==Linear:
                    iteration=1
                    w = layer._parameters[ 'weight' ]
                    v=layer.v
                    u=torch.normal(0,1,size=(w.shape[0],1)).cuda()
                    for _ in range(iteration):
                         u=torch.nn.functional.normalize(torch.mm(w.detach(),v).cuda(),dim=0)
                         v=torch.nn.functional.normalize(torch.mm(w.detach().T,u).cuda(),dim=0)
                    layer.v= v
                    loss+=(u.T@w@v).view(1)
                elif type(layer)==Conv:
                    iteration =1
                    w = layer._parameters[ 'weight' ]
                    v=layer.v
                    N, C, H, W = w.shape
                    m=w.view(N,C*W*H)
                    u=torch.normal(0,1,size=(m.shape[0],1)).cuda()
                    for _ in range(iteration):
                         u=torch.nn.functional.normalize(torch.mm(m.detach(),v).cuda(),dim=0)
                         v=torch.nn.functional.normalize(torch.mm(m.detach().T,u).cuda(),dim=0)
                    layer.v = v
                    loss+=(u.T@m@v).view(1)
        return beta*loss



    def forward(self, x,mode=0):
        """
            x: 输入图片
            quant: 是否使用模型量化
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)  # �?375*2*2的tensor打平�?1维，1500
        x = F.relu(self.fc1(x))  # 全连接层 1500 -> 400
        x = F.relu(self.fc2(x))  # 全连接层 400 -> 120
        x = F.relu(self.fc3(x))  # 全连接层 120 -> 84
        x = self.fc4(x)  # 全连接层 84  -> 10
        return x
