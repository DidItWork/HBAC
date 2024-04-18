import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class NaiveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1.weight, 
                               a=0, mode="fan_in", 
                               nonlinearity="relu") 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        nn.init.kaiming_uniform_(self.conv2.weight, 
                               a=0, mode="fan_in", 
                               nonlinearity="relu") 

        self.avg_pool = nn.AvgPool2d(5, 5)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=8512, out_features=512)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.fc2 = nn.Linear(in_features=512, out_features=6)

        nn.init.kaiming_uniform_(self.fc2.weight, 
                               a=0, mode="fan_in", 
                               nonlinearity="relu") 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.pool2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x