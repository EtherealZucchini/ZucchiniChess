import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # The "Skip Connection"
        return F.relu(x)


class ChessNetBody(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        # Initial Convolution to expand 12 channels to 64
        self.start_conv = nn.Conv2d(21, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Stack of Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(num_res_blocks)])



    def forward(self, x):
        x = F.relu(self.bn1(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        return x

class ChessNetValue(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        # Value Head: Predicts win/loss (-1 to 1)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc = nn.Linear(64, 1)
    def forward(self, x):
        # Global average pooling and value prediction
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 64)
        v = torch.tanh(self.value_fc(v))  # Squashes output to [-1, 1]
        return v

class ChessNetPolicy(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        self.policy_conv = nn.Conv2d(64, 73, kernel_size=3, padding=1)