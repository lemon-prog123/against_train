import os
import time
import torch
import torch.optim
import torch.utils.data
from util.save import *
def train(train_loader,test_loader,model,criterion,optimizer,logger,iteration=20):
    logger.info('Start Training')
    model.train()
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    for epoch in range(iteration):
        running_loss = 0
        for i,data in enumerate(train_loader):
            inputs,labels=data
            inputs=inputs.to(device)
            labels=labels.to(device)
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
        if epoch %5==4:
            is_best = False
            acc=eval(test_loader,model)
            if acc > best_acc:
                is_best = True
                best_acc=acc
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},
                            is_best=is_best)
            logger.info('accuracy:%d %% best_acc:%d %%' % (acc, best_acc))

def eval(test_loader,model):
    total=0
    correct=0
    for data in test_loader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        output = model(inputs)
        model.reconstruct()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return (100 * correct / total)
