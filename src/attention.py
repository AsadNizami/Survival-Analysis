import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio
        self.shared_dense_one = nn.Linear(
            in_channels, in_channels // self.ratio)
        self.shared_dense_two = nn.Linear(
            in_channels // self.ratio, in_channels)

    def forward(self, x):
        avg_pool = x.mean(dim=1)
        avg_pool = F.relu(self.shared_dense_one(avg_pool))
        avg_pool = torch.sigmoid(self.shared_dense_two(avg_pool)).unsqueeze(1)

        max_pool, _ = x.max(dim=1)
        max_pool = F.relu(self.shared_dense_one(max_pool))
        max_pool = torch.sigmoid(self.shared_dense_two(max_pool)).unsqueeze(1)

        cbam_feature = avg_pool + max_pool
        return x * cbam_feature


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False)

    def forward(self, x):
        avg_pool = x.mean(dim=2, keepdim=True)
        max_pool, _ = x.max(dim=2, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=2)

        cbam_feature = torch.sigmoid(self.conv2d(concat.transpose(1, 2)))
        return x * cbam_feature.transpose(1, 2)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        cbam_feature = self.channel_attention(x)
        cbam_feature = self.spatial_attention(cbam_feature)
        return cbam_feature


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        weighted_sum = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sum, attention_weights
