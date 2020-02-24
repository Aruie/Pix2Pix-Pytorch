import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import torchsummary
from torch.utils.data import DataLoader, Dataset


from model import *




class CifarDataLoader(DataSet) : 
    def __init__(self) :
        






def getImage() :
    
    x_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Grayscale(1),
        transforms.ToTensor() ])
    
    y_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor() ])



    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, num_workers=1)

    

def train(epochs = 1) :









if __name__ == '__main__' :

