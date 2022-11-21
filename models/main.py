import torch 
from torch import nn 
from torch import optim 
from torchvision import transforms
from INC3 import Inc3 
from data import getdata
from eval import plot
model = Inc3()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(),0.001,0.9)

data1 = getdata('train')
data2 = getdata('val')
trainac, valac = [],[]

def train():
    epochs = 1000

#just training setup ,need to add plot of losses etc.
    for i in range(epochs):
        correct = 0
        correctv = 0
        model.train()
        for images,labels in data1:
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            losst = loss(out,labels)
            losst.backward()
            optimizer.step()

            _,predict = torch.max(out.data,1)
            correct += (predict==labels).sum().item()

        trainac.append(correct/70)
        
        with torch.no_grad():
            for images,labels in data2:
                images,labels = images.to(device),labels.to(device)
                out = model(images)
                _,predict = torch.max(out.data,1)
                correctv += (predict==labels).sum().item()
            valac.append(correctv/30)
            print(f"Epoch {i}: train_ac = {correct/70} | val_ac = {correctv/30}")
        return trainac,valac

trainac,valac = train()

plot(trainac,valac)
