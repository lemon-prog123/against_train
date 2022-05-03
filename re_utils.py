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
        N,C,H,W=self.weight.shape
        self.v = torch.empty(C*H*W,1).cuda()
        nn.init.uniform_(self.v, -0.1, 0.1)
class Linear(nn.Linear):
    def __init__(self,*args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        H,W=self.weight.shape
        self.v = torch.empty(W,1).cuda()
        nn.init.uniform_(self.v, -0.1, 0.1)