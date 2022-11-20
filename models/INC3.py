import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np
from scipy.stats import truncnorm  
class Conv(nn.module):
    def __init__(self,inp_channels,out_channels,kernel_size):
        super(Conv,self).__init__()
        self.conv = nn.Conv2d(inp_channels,out_channels,kernel_size=kernel_size)
        self.normalize = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.normalize(x)
        x = F.relu(x,inplace=True)
        return x 
class IncA(nn.Module):
    def __init__(self,inp_channels,pool_feats):
        super(IncA,self).__init__()
        self.branch1 = Conv(inp_channels,8,1)
        self.brancha1 = Conv(inp_channels,8,1)
        self.brancha2 = Conv(8,16,5)
        self.branchb1 = Conv(inp_channels,8,1)
        self.branchb2 = Conv(8,16,3)
        self.branchb3 = Conv(16,32,3)
        self.pool = Conv(inp_channels,pool_feats,1)

    def forward(self,x):
        branch1 = self.branch1(x)
        brancha1  = self.brancha1(x)
        brancha2 = self.brancha2(brancha1)
        branchb1 = self.branchb1(x)
        branchb2 = self.branchb2(branchb1)
        branchb3 = self.branchb3(branchb2)
        pool = F.avg_pool2d(x,kernel_szie=3)
        pool = self.pool(pool)
        return torch.cat([branch1,brancha2,branchb3,pool],1)

class IncB(nn.Module):
    def __init__(self,inp_channels):
        super(IncB,self).__init__()
        self.branch1 = Conv(inp_channels,32,3)
        self.brancha1 = Conv(inp_channels,3,1)
        self.brancha2 = Conv(32,64,3)
        self.brancha3 = Conv(64,96,3)

    def forward(self,x):
        branch1 = self.branch1(x)
        brancha1 = self.brancha1(x)
        brancha2 = self.brancha2(brancha1)
        brancha3 = self.brancha3(brancha2)

        pool = F.max_pool2d(x,3)
        return torch.cat([branch1,brancha3,pool],1)

class IncC(nn.Module):
    def __init__(self,inp_channels,channels7):
        super(IncC,self).__init__()
        self.branch1 = Conv(inp_channels,128,1)
        self.branch71 = Conv(inp_channels,channels7,1)
        self.branch72 = Conv(channels7,channels7,(1,7))
        self.branch73 = Conv(channels7,128,(7,1))
        self.branch7a1 = Conv(inp_channels,channels7,1)
        self.branch7a2 = Conv(channels7,channels7,(7,1))
        self.branch7a3 = Conv(channels7,channels7,(1,7))
        self.branch7a4 = Conv(channels7,channels7,(7,1))
        self.branch7a4 = Conv(channels7,channels7,(1,7))
        self.branch7a5 = Conv(channels7,128,1)

        self.pool = Conv(inp_channels,128,1)

    def forward(self,x):
        b1 = self.branch1(x)
        b71 = self.branch71(x)
        b72 = self.branch72(b71)
        b73 = self.branch73(b72)
        b7a1 = self.branch7a1(x)
        b7a2 = self.branch7a2(b7a1)
        b7a3 = self.branch7a3(b7a2)
        b7a4 = self.branch7a4(b7a3)
        b7a5 = self.branch7a5(b7a4)
        pool = F.avg_pool2d(x,3)
        pool = self.pool(pool)
        return torch.cat([b1,b73,b7a5,pool])

class IncD(nn.Module):
    def __init__(self,inp_channels):
        super(IncD,self).__init__()
        self.b1 = Conv(inp_channels,32,1)
        self.b2 = Conv(32,64,3)
        self.ba1 = Conv(inp_channels,32,1)
        self.ba2 = Conv(32,64,(1,7))
        self.ba3 = Conv(64,128,(7,1))
        self.ba4 = Conv(128,192,3)

    def forward(self,x):
        b1 = self.b1(x)
        b2 = self.b2(b1)
        ba1 = self.ba1(x)
        ba2 = self.ba2(ba1)
        ba3 = self.ba3(ba2)
        ba4 = self.ba4(ba3)
        pool = F.max_pool2d(x,3)
        return torch.cat([b2,ba4,pool],1)

class Inc3(nn.Module):
    def __init__(self,num_classes,transform=True):
        super(Inc3,self).__init__()
        self.transform = transform
        self.convlayer = Conv(3,32,3)
        self.inca = IncA(32,8)
        self.inca2 = IncA(64,72)
        self.incb = IncB(128)
        self.incc = IncC(256,64)
        self.incd = IncD(512)
        self.fl = nn.Linear(768,num_classes)

        for i in self.modules():
            if isinstance(i,nn.Conv2d) or isinstance(i,nn.Linear):
                x = truncnorm(-2,2,scale=0.01)
                weight = torch.as_tensor(x.rvs(i.weight.numel()),dtype=i.weight.dtype)
                weight = weight.view(i.weight.size())
                with torch.no_grad():
                    i.weight.copy_(weight)
            elif isinstance(i,nn.BatchNorm2d):
                nn.init.constant_(i.weight,1)
                nn.init.constant_(i.bias,0)
    
    def forward(self,x):
        x = self.convlayer(x)
        x = self.inca(x)
        x = self.inca2(x)
        x = self.incb(x)
        x = self.incc(x)
        x = self.incd(x)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = F.dropout(x,training = self.training)
        x = torch.flatten(x)
        x = self.fl(x)
        return x
