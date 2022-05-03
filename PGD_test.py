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
import torchvision
from torch import  Tensor
from PGD import PGD
from setseed import setup_seed
from cifar10 import CIFAR10
from log import create_logger
from torch.utils.data import DataLoader
from covnet import ConvNet
import convnet_o
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    setup_seed(1)
    dataset=CIFAR10()
    logger=create_logger()
    BATCH_SIZE = 32
    test_loader = DataLoader(dataset.testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model=convnet_o.ConvNet()
    model.to(device)
    dic=torch.load('./checkpoint_regular/best.pth')
    model.load_state_dict(dic['state_dict'])
    model.mode=1
    pgd_net=PGD(model)
    total=0
    correct=0
    for data in test_loader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels=labels.cuda()
        grid_1=torchvision.utils.make_grid(inputs,padding=2)
        writer.add_image('test_o',grid_1,1)
        adv_inputs=pgd_net.generate(x=inputs,y=labels)
        grid_2=torchvision.utils.make_grid(adv_inputs,padding=2)
        writer.add_image('test_a',grid_2,1)
        model.eval()
        output = model(adv_inputs)
        #model.reconstruct()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    logger.info('accuracy:%d %%' % (100 * correct / total))
    writer.close()