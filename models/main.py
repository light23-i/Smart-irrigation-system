import torch 
from torch import nn 
from torch import optim 
from torchvision import transforms
from INC3 import Inc3 
from data import getdata

model = Inc3()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(),0.001,0.9)

data = getdata()


def train():
    epochs = 1000
#just training setup ,need to add plot of losses etc.
    for i in range(epochs):
        model.train()
        for images,labels in data:
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = loss(out,labels)
            loss.backward()
            optimizer.step()
            