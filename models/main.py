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
trainloss, valloss = [],[]

def train():
    epochs = 1000

#just training setup ,need to add plot of losses etc.
    for i in range(epochs):
        correct = 0
        trainl = 0
        vall = 0
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
            trainl = losst.item()*images.size(0)

        trainloss.append(trainl/70)
        
        with torch.no_grad():
            for images,labels in data2:
                images,labels = images.to(device),labels.to(device)
                out = model(images)
                lossv = loss(out,labels)
                _,predict = torch.max(out.data,1)
                correctv += (predict==labels).sum().item()
                vall = lossv.item()*images.size(0)
            valloss.append(vall/30)
            print(f"Epoch {i}: train_loss = {trainl/70} | val_loss = {vall/30}")
        return trainl/70,vall/30

trainl,vall = train()

plot(trainl,vall)
