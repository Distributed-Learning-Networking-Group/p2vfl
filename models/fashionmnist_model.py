import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet18

class ClientNet(nn.Module):
    def __init__(self):
        super(ClientNet, self).__init__()
        self.resnet18 = ResNet18(in_channel=1,num_classes=1152)

    def forward(self,x):
        x = x.view(x.shape[0],1,28,-1)
        x = self.resnet18(x)
        return x

class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.fc1 = nn.Linear(1152, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# class ClientNet(nn.Module):
#     def __init__(self):
#         super(ClientNet, self).__init__()
#         self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

#     def forward(self,x):
#         x = self.conv(x)
#         x = torch.flatten(x, 1)
#         return x

# class ServerNet(nn.Module):
#     def __init__(self):
#         super(ServerNet, self).__init__()
#         self.resnet18 = ResNet18(in_channel=1)

#     def forward(self,x):
#         x = x.view(x.shape[0],1,28,-1)
#         x = self.resnet18(x)
#         return x