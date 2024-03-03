import torch
import torch.nn as nn
import torch.nn.functional as F
# from networks import ResNet, 1DCNN

class CNNDetector(nn.Module):

    def __init__(self, config):

        super(CNNDetector, self).__init__()

        self.backbone_1d = None

        self.backbone_2d = None

        self.head = None

    def forward(self, x):

        """
        returns a dictionary pred_dict with the logits for loss calculation and gradient descent.
        """

        pass