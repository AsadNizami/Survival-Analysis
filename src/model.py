import timm
import torch.nn as nn
from attention import CBAM, AttentionLayer
from config import *
import torch
import torch.nn.functional as F


class WSISurvivalModel(nn.Module):
    def __init__(self, pretrained_model):
        super(WSISurvivalModel, self).__init__()
        self.patch_encoder = pretrained_model
        # self.patch_encoder = timm.create_model(
        #     "hf-hub:MahmoodLab/uni",
        #     pretrained=True,
        #     init_values=1e-5,
        #     dynamic_img_size=True)

        # hf-hub:MahmoodLab/uni
        # hf-hub:1aurent/resnet18.tiatoolbox-kather100k
        # hf-hub:1aurent/resnet50.tcga_brca_simclr
        # hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3
        # self.patch_encoder = timm.create_model(model_name="hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3", pretrained=True, num_classes=0, dynamic_img_size=True)

        for param in list(self.patch_encoder.parameters()):
            param.requires_grad = False

        self.cbam = CBAM(1024)  # Change according to encoder output dimension
        self.attention = AttentionLayer(1024)  # Change according to encoder output dimension

        # LSTM parameters
        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)

        # self.fc1 = nn.Linear(256, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size, num_patches = x.size(0), x.size(1)

        # x = x.view(batch_size * num_patches, 3, *(IMG_SIZE))
        # x = self.patch_encoder(x)
        # x = x.view(batch_size, num_patches, -1)
        outputs = []

        # Process each batch separately
        for i in range(batch_size):
            batch_patches = x[i]  # Shape: (num_patches, channel, IMG_SIZE[0], IMG_SIZE[1])
            encoded_patches = self.patch_encoder(batch_patches)  # Process the patches            
            outputs.append(encoded_patches)

        x = torch.stack(outputs)  # Shape: (batch_size, num_patches, encoded_dim)

        # x = self.cbam(x)
        x, attention_weights = self.attention(x)
        # x, _ = torch.max(x, dim=1)

        # Pass through LSTM
        x, (h_n, c_n) = self.lstm(x)  # LSTM output

        # Use the last hidden state for classification
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc4(x)

        return x, None
