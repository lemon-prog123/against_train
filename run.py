import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.setseed import *
from util.save import *
from net.covnet import *
from log.logger import *
from dataloader.cifar10 import *
from common.engine import *
if __name__ == '__main__':
    setup_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset=CIFAR10()
    logger = create_logger()
    BATCH_SIZE=32
    train_loader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset.testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model=ConvNet(quant=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(train_loader,test_loader,model,criterion,optimizer,logger)





