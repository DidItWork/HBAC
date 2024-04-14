import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_v2_s, convnext_tiny

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

        # self.output = nn.LogSoftmax(dim=0)

    def forward(self, data_dict):
        x = data_dict["spec"]
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.pool2(x)
        
        # print(f'Shape before pool: {x.shape}')
        x = self.avg_pool(x)
        # print(f'Shape after pool: {x.shape}')
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
    
        x = self.fc2(x)
        # x = self.output(x)
        return x

class CNNDetector(nn.Module):

    def __init__(self, config={}):

        super(CNNDetector, self).__init__()

        model_name = config.get("model_name", "efficientnet_b0")
        num_classes = config.get("num_classes", 6)

        self.backbone_1d = None

        if model_name == "convnext":

            self.backbone_spec = convnext_tiny(weights="IMAGENET1K_V1")
            self.backbone_spec.classifier[2] = nn.Linear(768, num_classes)
        
        elif model_name == "simple":

            self.backbone_spec = NaiveCNN()
 
        elif model_name == "efficientnet_v2s":

            self.backbone_spec = efficientnet_v2_s(weights="IMAGENET1K_V1")
            self.backbone_spec.classifier[1] = nn.Linear(1280, num_classes)
            
        else:

            self.backbone_spec = efficientnet_b0(weights="IMAGENET1K_V1")
            self.backbone_spec.classifier[1] = nn.Linear(1280, num_classes)
        
#         self.backbone_2d.conv1 = nn.Conv2d(config.get("in_channels",1), 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        
#         print(self.backbone_2d)

        print(self.backbone_spec)

        self.head = None

        self.output_layer = nn.Softmax(dim=1)

    def get_loss(self, pred_dict):

        pass

    def forward(self, data_dict, inference=False):

        """
        returns a dictionary pred_dict with the logits for loss calculation and gradient descent.
        """
        
#         print(torch.isnan(torch.sum(data_dict["spec"])))

        x = self.backbone_spec(data_dict["spec"])

        if inference:
            x = self.output_layer(x)

        return x