import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):

    def __init__(self, 
                 model_name:str = "r18", 
                 num_classes: int = 6,
                 in_channels: int = 1,
                 **kwargs) -> None:
        super(ResNet, self).__init__()

        if model_name == "r50":
            self.resnet = models.resnet50()
        elif model_name == "r34":
            self.resnet = models.resnet34()
        else:
            self.resnet = models.resnet18()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)

        print(self.resnet)
    
    def forward(self, x):

        x = self.resnet(x)

        x = nn.Softmax(x)

        return x

if __name__ == "__main__":
    r18 = ResNet()

