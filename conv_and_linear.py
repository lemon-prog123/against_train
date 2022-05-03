import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.autograd import Variable
from torch import  Tensor

class Conv(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
    def reconstruct(self):
        self.weight.data=self.orignal_weight
    def quant2(self,x):
        x_new=torch.abs(x)
        max_v=torch.max(x_new)
        scale=max_v/torch.tensor(127)
        input=x.detach()/scale
        input=torch.round(input)
        input=torch.clip(input,-128,127)
        return input,scale
    def quant(self,x):
        input=x.data.detach().numpy().copy()
        input=np.abs(input)
        max_v = np.nanmax(input)
        scale=max_v/127
        input=x.data.detach().numpy().copy()/scale
        input=np.round(input)
        input=np.clip(input,-128,127)
        return input,scale
    def forward(self, x):
        x_new,scale1=self.quant2(x)
        #scale1=torch.from_numpy(np.array(scale1))
        #x.data=torch.from_numpy(x_new)*scale1
        x.data=x_new*scale1
        self.orignal_weight=self.weight.data
        weight_new,scale2=self.quant2(self.weight)
        #scale2=torch.from_numpy(np.array(scale2))
        #self.weight.data=torch.from_numpy(weight_new)*scale2
        self.weight.data=weight_new*scale2
        output=self._conv_forward(x, self.weight, self.bias)
        return output

class Linear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(Linear, self).__init__(*args, **kwargs)
    def reconstruct(self):
        self.weight.data=self.orignal_weight
    def quant2(self,x):
        x_new=torch.abs(x)
        max_v=torch.max(x_new)
        scale=max_v/torch.tensor(127)
        input=x.detach()/scale
        input=torch.round(input)
        input=torch.clip(input,-128,127)
        return input,scale
    def quant(self,x):
        input=x.data.detach().numpy().copy()
        input=np.abs(input)
        max_v = np.nanmax(input)
        scale=max_v/127
        input=x.data.detach().numpy().copy()/scale
        input=np.round(input)
        input=np.clip(input,-128,127)
        return input,scale
    def forward(self, x):
        x_new,scale1=self.quant2(x)
        #scale1=torch.from_numpy(np.array(scale1))
        #x.data=torch.from_numpy(x_new)*scale1
        x.data=x_new*scale1
        self.orignal_weight=self.weight.data
        weight_new,scale2=self.quant2(self.weight)
        #scale2=torch.from_numpy(np.array(scale2))
        #self.weight.data=torch.from_numpy(weight_new)*scale2
        self.weight.data=weight_new*scale2
        return F.linear(x, self.weight, self.bias)
