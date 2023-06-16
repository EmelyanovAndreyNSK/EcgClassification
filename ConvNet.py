import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 16, 5, stride=1, padding=0)#1 16 3
        self.conv1 = nn.Conv2d(16, 32, 5, stride=1, padding=0)#16 32 3
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0)

        #self.linear0 = nn.Linear(128*12*12, 512)
        self.linear0 = nn.Linear(128*11*11, 512)        
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 5)

       
    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.act(out)
        out = torch.flatten(out, 1)
        out = self.linear0(out)
        out = self.act(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out