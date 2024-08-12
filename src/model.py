import timm
import torch.nn as nn
from attention import CBAM, AttentionLayer
from config import *
import torch.nn.functional as F


class WSISurvivalModel(nn.Module):
    def __init__(self):
        super(WSISurvivalModel, self).__init__()
        self.patch_encoder = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True)

        # hf-hub:MahmoodLab/uni
        # hf-hub:1aurent/resnet18.tiatoolbox-kather100k
        # hf-hub:1aurent/resnet50.tcga_brca_simclr
        # hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3
        # self.patch_encoder = timm.create_model(model_name="hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3", pretrained=True, num_classes=0, dynamic_img_size=True)

        for param in list(self.patch_encoder.parameters()):
            param.requires_grad = False

        self.cbam = CBAM(1024)  # change according to encoder output dimension
        self.attention = AttentionLayer(1024)  # change according to encoder output dimension
        
        self.fc1 = nn.Linear(1024, 256)  # change according to encoder output dimension
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size, num_patches = x.size(0), x.size(1)
        x = x.view(batch_size * num_patches, 3, *(IMG_SIZE))
        x = self.patch_encoder(x)
        x = x.view(batch_size, num_patches, -1)
        x = self.cbam(x)
        x, attention_weights = self.attention(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)

        return x, None  # attention_weights
