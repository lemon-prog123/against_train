import torch
import torch.nn as nn
class PWLA1d(nn.Module):
    def __init__(self,N=16,momentum=0.9):
        super(PWLA1d, self).__init__()
        self.N = N
        self.momentum = torch.tensor(momentum).cuda()
        self.Br = torch.nn.Parameter(torch.tensor(10.))
        self.Bl = torch.nn.Parameter(torch.tensor(-10.))
        self.Kl = torch.nn.Parameter(torch.tensor(0.))
        self.Kr = torch.nn.Parameter(torch.tensor(1.))
        self.running_mean=torch.zeros(1).cuda()
        self.running_var=torch.ones(1).cuda()
        self.Yidx = torch.nn.Parameter(nn.functional.relu(torch.linspace(self.Bl.item(),self.Br.item(),self.N+1)))
        self.reset_parameters()
    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
    def forward(self, x, mode):
        if mode == 1:
            d = (self.Br - self.Bl) / self.N  # Interval length
            '''{TODO} Refactor code'''
            #             Bidx = torch.linspace(self.Bl.item(),self.Br.item(),self.N)#LEFT Interval boundaries
            DATAind = torch.clamp(torch.floor((x - self.Bl.item()) / d), 0,
                                  self.N - 1).cuda()  # Number of corresponding interval for X
            Bdata = self.Bl + DATAind * d  # LEFT Interval boundaries
            Bdata=Bdata.cuda()
            maskBl = x < self.Bl  # Mask for LEFT boundary
            maskBr = x >= self.Br  # Mask for RIGHT boundary
            maskOther = ~(maskBl + maskBr)  # Mask for INSIDE boundaries
            maskBl=maskBl.cuda()
            maskBr=maskBr.cuda()
            maskOther=maskOther.cuda()
            Ydata = self.Yidx[ DATAind.type(torch.int64) ]  # Y-value for data
            Kdata = (self.Yidx[ (DATAind).type(torch.int64) + 1 ] - self.Yidx[
                DATAind.type(torch.int64) ]) / d  # SLOPE for data
            #Kdata=torch.ones_like(Ydata).cuda()
            return maskBl * ((x - self.Bl) * self.Kl + self.Yidx[ 0 ]) + maskBr * (
                    (x - self.Br) * self.Kr + self.Yidx[ -1 ]) + maskOther * ((x - Bdata) * Kdata + Ydata)
        else:
            mean = x.detach().mean([ 0, -1 ])  # {TODO}: Possibly split along channel axis
            var = x.detach().var([ 0, -1 ])  # {TODO}: Possibly split along channel axis
            self.running_mean = (self.momentum * self.running_mean) + (
                    1.0 - self.momentum) * mean  # .to(input.device)
            self.running_var = (self.momentum * self.running_var) + (1.0 - self.momentum) * (
                    x.shape[ 0 ] / (x.shape[ 0 ] - 1) * var)
            return nn.functional.relu(x)

class PWLA3d(torch.nn.Module):
    '''Piecewise Linear activation function for a 3 Dimensional data (B,C,H,L)
    from paper: https://arxiv.org/pdf/2104.03693.pdf
    Args:
        N = int - number of intervals contained in function
        momentum = float - strength of momentum during the statistics collection phase
        (Phase I in paper)
    '''
    def __init__(self,N=16,momentum=0.9):
        super(PWLA3d, self).__init__()
        self.N = N
        self.momentum = torch.tensor(momentum).cuda()
        self.Br = torch.nn.Parameter(torch.tensor(10.))
        self.Bl = torch.nn.Parameter(torch.tensor(-10.))
        self.Kl = torch.nn.Parameter(torch.tensor(0.))
        self.Kr = torch.nn.Parameter(torch.tensor(1.))
        self.running_mean=torch.zeros(1).cuda()
        self.running_var=torch.ones(1).cuda()
        self.Yidx = torch.nn.Parameter(nn.functional.relu(torch.linspace(self.Bl.item(),self.Br.item(),self.N+1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, mode):
        if mode==1:
            d=(self.Br-self.Bl)/self.N#Interval length
            '''{TODO} Refactor code'''
#             Bidx = torch.linspace(self.Bl.item(),self.Br.item(),self.N)#LEFT Interval boundaries
            DATAind = torch.clamp(torch.floor((x-self.Bl.item())/d),0,self.N-1).cuda()#Number of corresponding interval for X
            Bdata = self.Bl+DATAind*d#LEFT Interval boundaries
            Bdata=Bdata.cuda()
            maskBl = x<self.Bl#Mask for LEFT boundary
            maskBr = x>=self.Br#Mask for RIGHT boundary
            maskOther = ~(maskBl+maskBr)#Mask for INSIDE boundaries
            maskBl=maskBl.cuda()
            maskBr=maskBr.cuda()
            maskOther=maskOther.cuda()
            Ydata = self.Yidx[DATAind.type(torch.int64)]#Y-value for data
            Kdata = (self.Yidx[(DATAind).type(torch.int64)+1]-self.Yidx[DATAind.type(torch.int64)])/d#SLOPE for data
            #Kdata=torch.ones_like(Ydata).cuda()
            return  maskBl*((x-self.Bl)*self.Kl+self.Yidx[0]) + maskBr*((x-self.Br)*self.Kr + self.Yidx[-1]) + maskOther*((x-Bdata)*Kdata + Ydata)
        else:
            mean = x.detach().mean([0,1,2,-1]) #{TODO}: Possibly split along channel axis
            var = x.detach().var([0,1,2,-1]) #{TODO}: Possibly split along channel axis
            self.running_mean = (self.momentum * self.running_mean) + (1.0-self.momentum) * mean # .to(input.device)
            self.running_var = (self.momentum * self.running_var) + (1.0-self.momentum) * (x.shape[0]/(x.shape[0]-1)*var)
            return nn.functional.relu(x)