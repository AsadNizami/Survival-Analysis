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

        for param in list(self.patch_encoder.parameters()):
            param.requires_grad = False

        self.cbam = CBAM(1024)  # Change according to encoder output dimension
        self.attention = AttentionLayer(768)  # Change according to encoder output dimension

        # LSTM parameters
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        batch_size, num_patches = x.size(0), x.size(1)

        outputs = []

        # Process each batch separately
        for i in range(batch_size):
            batch_patches = x[i]  # Shape: (num_patches, channel, IMG_SIZE[0], IMG_SIZE[1])
            encoded_patches = self.patch_encoder(batch_patches).last_hidden_state[:, 0, :].squeeze(0)
            # print(encoded_patches.shape, type(encoded_patches))
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
