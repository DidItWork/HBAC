import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import ResNet

class CNNDetector(nn.Module):

    def __init__(self, config):

        super(CNNDetector, self).__init__()

        self.backbone_1d = None

        self.backbone_2d = ResNet(model_name="r18")

        self.head = None

    def forward(self, data_dict):

        """
        returns a dictionary pred_dict with the logits for loss calculation and gradient descent.
        """

        

        return self.backbone_2d(data_dict["spec"])