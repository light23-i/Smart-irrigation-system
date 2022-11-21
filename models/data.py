import os 
from PIL import Image 
import torchvision.transforms.functional as TF 

def getdata(file:str):
    dir = os.listdir(file)
    data = []

    for i in dir:
        image = Image.open(f'{file}/{i}')
        x = TF.to_tensor(image)
        x.unsqueeze_(0)
        data.append((x,int(i)))
    return data
