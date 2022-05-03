import os
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
from setseed import setup_seed
from cifar10 import CIFAR10
from save import save_checkpoint
from covnet import ConvNet
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import prepare_qat
from torch.quantization import convert

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(1)
    dataset=CIFAR10()
    logger=log.create_logger()
    BATCH_SIZE = 32
    train_loader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset.testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model=ConvNet(quant=True)
    model.to(device)
    logger.info('start training')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(20):
        running_loss = 0
        model.train()
        for i,data in enumerate(train_loader):
            inputs,labels=data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels=labels.cuda()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            model.reconstruct()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999: 
                logger.info('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        correct = 0
        total = 0
        best_acc=0
        if epoch % 5 == 4:
            model.eval()
            is_best = False
            for data in test_loader:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels=labels.cuda()
                output = model(inputs)
                model.reconstruct()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if (100 * correct / total) > best_acc:
                is_best = True
            best_acc = max(100 * correct / total, best_acc)
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},
                            is_best=is_best)
            logger.info('accuracy:%d %% best_acc:%d %%' % (100 * correct / total, best_acc))

