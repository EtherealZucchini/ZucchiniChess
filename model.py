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
    def __init__(self, channels=64):
        super().__init__()
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        v = F.relu(self.bn(self.value_conv(x)))
        v = v.view(-1, 64)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        return v


class ChessNetPolicy(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # The "additional rectified, batch-normalized convolutional layer"
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # The "final convolution of 73 filters"
        # A 1x1 kernel is used here to project the channels down without altering spatial data
        self.conv2 = nn.Conv2d(channels, 73, kernel_size=1)

    def forward(self, x, legal_moves_mask):
        # 1. Pass through the hidden policy layer
        x = F.relu(self.bn1(self.conv1(x)))

        # 2. Get raw logits for all 73 planes (Shape: Batch, 73, 8, 8)
        logits = self.conv2(x)

        # 3. Flatten the spatial dimensions to a 1D vector per batch item (73 * 8 * 8 = 4672)
        logits = logits.view(logits.size(0), -1)

        # 4. Apply the Legal Move Mask BEFORE Softmax
        # legal_moves_mask should be a boolean tensor of shape (Batch, 4672)
        logits = logits.masked_fill(~legal_moves_mask, -1e9)

        # 5. Softmax normalizes the remaining legal moves to sum to 1.0
        policy = F.softmax(logits, dim=1)

        return policy

class ChessNet(nn.Module):
    def __init__(self, channels=21, num_res_blocks=4):
        super().__init__()
        self.body = ChessNetBody(num_res_blocks=4)
        self.value = ChessNetValue(channels=64)
        self.policy = ChessNetPolicy(channels=64)

    def forward(self, x, legal_moves_mask):
        latent = self.body(x)
        value = self.value(latent)
        policy = self.policy(latent, legal_moves_mask)
        return value, policy