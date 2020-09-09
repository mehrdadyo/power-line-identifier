import torch
import torch.nn as nn

class GubbiNet(nn.Module):
    'Implements network developed in doi.org/10.23919/MVA.2017.7986837 for classifying images that have utility lines'
    def __init__(self, conv1=16, conv2=32, conv3=64, conv4=96, fc1=256, fc2=512, dropout=0):
        super(GubbiNet, self).__init()
        # self.HoG =
        self.conv = nn.Sequential(
            nn.Conv2d(9, conv1, 5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv1),
            nn.Conv2d(conv1, conv2, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv2),
            nn.Conv2d(conv2, conv3, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv3),
            nn.Conv2d(conv3, conv4, 1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv4)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(64 * conv4, fc1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(fc1, fc2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(fc2, 3, bias=True),
            nn.Softmax()
        )
        
    def forward(self,u):
        return self.fc(self.conv(self.HoG(u)).reshape(-1,1))
            
        
class DuckNet(nn.Module):
    'Our novel neural network framework for catching ducks'
    def __init__(self, conv1=16, conv2 = 32, conv3=64, conv4=96, fc1= 512, dropout=0.25):
        super(Ducknet, self).__init()
        # self.HoG =
        self.conv = nn.Sequential(
            nn.Conv2d(9, conv1, 5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(conv1),
            nn.Conv2d(conv1, conv2, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(conv2),
            nn.Conv2d(conv2, conv3, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(conv3),
            nn.Conv2d(conv3, conv4, 1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(conv4)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(64 * conv4, fc1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(fc1, 3, bias=True),
            nn.Softmax()
        )
        
    def forward(self,u):
        return self.fc(self.conv(u).reshape(-1,1))