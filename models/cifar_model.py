import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet18

class ClientNet(nn.Module):
    def __init__(self):
        super(ClientNet, self).__init__()
        self.resnet18 = ResNet18(in_channel=3,num_classes=1152)

    def forward(self,x):
        x = x.view(x.shape[0],3,32,-1)
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
#     def __init__(self, n_dim):
#         super(ClientNet, self).__init__()
#         self.fc = nn.Linear(n_dim, 10)
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self,x):
#         # print('0:', x.size())
#         x = F.relu(self.conv1(x))  # Size([256, 32, 32, 10]) Size([256, 32, 32, 14]) Size([256, 32, 32, 8])
#         # print('1:', x.size())
#         x = F.max_pool2d(x, 2, 2)  # Size([256, 32, 16, 5]) Size([256, 32, 16, 7]) Size([256, 32, 16, 4])
#         # print('2:', x.size())
#         x = self.dropout(x)
#         x = F.relu(self.conv2(x))  # Size([256, 64, 16, 5]) Size([256, 64, 16, 7]) Size([256, 64, 16, 4])
#         x = F.max_pool2d(x, 2, 2)  # Size([256, 64, 8, 2]) Size([256, 64, 8, 3]) Size([256, 64, 8, 2])
#         # print('3:', x.size())
#         x = self.dropout(x)
#         x = self.conv3(x)
#         # print('4:', x.size())
#         x = self.dropout(x)
#         x = x.view(x.shape[0], -1) # Size([256, 1024]) Size([256, 1536]) Size([256, 1024])
#         # print('5:', x.size())
#         return x

# class ServerNet(nn.Module):
#     def __init__(self):
#         super(ServerNet, self).__init__()
#         self.fc1 = nn.Linear(1024, 64)
#         self.fc2 = nn.Linear(64, 10)
        
#     def forward(self,x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
