import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import  log
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from covnet import ConvNet
class PGD(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # must be pytorch
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    def generate(self, x, **params):
        self.parse_params(**params)
        labels = self.y
        labels=labels.cuda()
        x=x.cuda()
        adv_x = self.attack(x, labels)
        return adv_x

    def parse_params(self, eps=8/255, iter_eps=2/255, nb_iter=4, clip_min=0.0, clip_max=1.0, C=0.0,
                     y=None, ord=np.inf, rand_init=True, flag_target=False):
        self.eps = eps#max eps
        self.iter_eps = iter_eps#step eps
        self.nb_iter = nb_iter#step nums
        self.clip_min = clip_min#clip num
        self.clip_max = clip_max
        self.y = y#labels
        self.ord = ord
        self.rand_init = rand_init
        #self.model.to(self.device)
        self.model.train()
        self.flag_target = flag_target
        self.C = C

    def sigle_step_attack(self, x, pertubation, labels):
        adv_x = x + pertubation#
        # get the gradient of x
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True#
        loss_func = nn.CrossEntropyLoss()
        preds = self.model(adv_x)
        if self.flag_target:
            loss = -loss_func(preds, labels)
        else:
            loss = loss_func(preds, labels)
            # label_mask=torch_one_hot(labels)
            #
            # correct_logit=torch.mean(torch.sum(label_mask * preds,dim=1))
            # wrong_logit = torch.mean(torch.max((1 - label_mask) * preds, dim=1)[0])
            # loss=-F.relu(correct_logit-wrong_logit+self.C)

        self.model.zero_grad()
        loss.backward()
        #self.model.reconstruct()
        grad = adv_x.grad.detach()
        # get the pertubation of an iter_eps
        #grad=grad.cpu().detach().numpy()
        pertubation = torch.tensor(self.iter_eps) * torch.sign(grad)
        adv_x_new= adv_x.detach() + pertubation
        #x = x.cpu().detach().numpy()

        pertubation = torch.clip(adv_x_new, self.clip_min, self.clip_max) - x
        #pertubation = clip_pertubation(pertubation, self.ord, self.eps)
        pertubation=torch.clip(pertubation,-self.eps,self.eps)
        return pertubation

    def attack(self, x, labels):
        if self.rand_init:
            x_tmp = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()#添加随机扰动
        else:
            x_tmp = x
        pertubation = torch.zeros(x.shape).type_as(x).to(self.device)
        for i in range(self.nb_iter):
            pertubation = self.sigle_step_attack(x_tmp, pertubation=pertubation, labels=labels)
            #pertubation = torch.Tensor(pertubation).type_as(x).to(self.device)
        #print(x,pertubation)
        adv_x = x + pertubation
        #adv_x = adv_x.cpu().detach().numpy()
        adv_x = torch.clip(adv_x, self.clip_min, self.clip_max)

        return adv_x

#net=VGG.VGG16()
#device = torch.device("cuda:0")
#net.to(device)
#dic=torch.load('./checkpoint/best.pth')
#net.load_state_dict(dic['state_dict'])
#net.to(device)
#model=PGD(ConvNet(quant=False))
#input_test = torch.FloatTensor(1,3,32,32)
#label=torch.LongTensor(1)
#label[0]=1
#adv_x=model.generate(x=input_test,y=label)
#print(adv_x.requires_grad)
#print(adv_x.shape)
#print(input_test-adv_x)
