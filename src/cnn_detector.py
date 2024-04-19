import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_v2_s, convnext_tiny
from .networks import NaiveCNN
import torch.nn.functional as F

class CNNDetector(nn.Module):

    def __init__(self, config={}):

        super(CNNDetector, self).__init__()

        model_name = config.get("model_name", "efficientnet_b0")
        num_classes = config.get("num_classes", 6)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")


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
            self.backbone_spec.classifier[1] = nn.Linear(1408, num_classes)
        
#         self.backbone_2d.conv1 = nn.Conv2d(config.get("in_channels",1), 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        
#         print(self.backbone_2d)

        self.output_layer = nn.Softmax(dim=1)
        
        print(self)

    def get_loss(self, predictions, labels):

        predictions = F.log_softmax(predictions, dim=1)

        kl_loss = self.kl_loss(predictions, labels)

        return kl_loss

    def forward(self, data_dict, inference=False):

        """
        returns a dictionary pred_dict with the logits for loss calculation and gradient descent.
        """
        
#         print(torch.isnan(torch.sum(data_dict["spec"])))

        x = self.backbone_spec(data_dict["spec"])

        if inference:
            x = self.output_layer(x)

        return x