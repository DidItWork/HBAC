import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_v2_s, convnext_tiny
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

class ContrastiveDetector(nn.Module):

    def __init__(self, config={}):

        super(ContrastiveDetector, self).__init__()

        model_name = config.get("model_name", "efficientnet_b0")
        num_features = config.get("num_features", 512)
        num_classes = config.get("num_classes", 6)

        self.xf = torch.tensor([0.])
        self.x2f = torch.tensor([0.])

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        eeg_blocks = 3
            
        #Strided Convolution
        module_list = nn.ModuleList([
            nn.Conv1d(19, 64, stride=5, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, stride=5, kernel_size=7, bias=False),
        ])

        for i in range(eeg_blocks):

            module_list.append(nn.Sequential(
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128,128,stride=3,kernel_size=5,bias=False),
                ))

        module_list.append(nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,256,stride=3,kernel_size=5,bias=False)
        ))
        
        module_list.append(nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,num_features,stride=1,kernel_size=3,bias=False)
        ))

        self.backbone_eeg = nn.Sequential(*module_list)

        if model_name == "convnext":

            self.backbone_spec = convnext_tiny(weights="IMAGENET1K_V1")
            self.backbone_spec.classifier[2] = nn.Linear(768, num_features)
        
        elif model_name == "simple":

            self.backbone_spec = NaiveCNN()
 
        elif model_name == "efficientnet_v2s":

            self.backbone_spec = efficientnet_v2_s(weights="IMAGENET1K_V1")
            self.backbone_spec.classifier[1] = nn.Linear(1280, num_features)
            
        else:

            self.backbone_spec = efficientnet_b0(weights="IMAGENET1K_V1")
            self.backbone_spec.classifier[1] = nn.Linear(1280, num_features)
        
#         self.backbone_2d.conv1 = nn.Conv2d(config.get("in_channels",1), 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        
#         print(self.backbone_2d)

        self.norm_eeg = nn.BatchNorm1d(num_features)
        self.norm_spec = nn.BatchNorm1d(num_features)

        self.head = nn.Sequential(
            nn.Linear(num_features*2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes, bias=False),
        )


        self.output_layer = nn.Softmax(dim=1)

        print(self)

    def get_loss(self, predictions, labels):

        predictions = F.log_softmax(predictions, dim=1)

        kl_loss = self.kl_loss(predictions, labels)

        contrastive_loss = -torch.mean(nn.CosineSimilarity()(self.xf, self.x2f))

        return kl_loss, contrastive_loss

    def forward(self, data_dict, inference=False):

        """
        returns a dictionary pred_dict with the logits for loss calculation and gradient descent.
        """
        
#         print(torch.isnan(torch.sum(data_dict["spec"])))

        x = self.backbone_spec(data_dict["spec"])

        x2 = self.backbone_eeg(data_dict["eeg"])

        x2 = nn.Flatten()(x2)

        x = self.norm_spec(x)

        x2 = self.norm_eeg(x2)

        xx2 = torch.cat([x,x2],dim=1)

        if not inference:
            self.xf = x
            self.x2f = x2

        x = self.head(xx2)

        if inference:
            x = self.output_layer(x)

        return x

if __name__ == "__main__":

    model_config = {}

    model = ContrastiveDetector(model_config).to(device)

    data_dict = {
        "eeg":torch.rand(2,19,10000).to(device),
        "spec":torch.rand(2,3,400,300).to(device),
        "label": torch.rand(2,6).to(device),
    }

    pred = model(data_dict)

    model.get_loss(pred, data_dict["label"])
