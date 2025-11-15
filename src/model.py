"""Chess neural network model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """Policy head for move prediction."""

    def __init__(self, in_channels, num_actions=4096):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, num_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x  # Return logits


class ValueHead(nn.Module):
    """Value head for position evaluation."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)  # Output in [-1, 1]
        return x


class ChessNet(nn.Module):
    """
    Complete chess neural network with ResNet architecture.

    Architecture:
        Input (14, 8, 8) →
        Initial Conv Block (num_filters, 8, 8) →
        Residual Blocks × N →
        ├─ Policy Head → (4096,)
        └─ Value Head → (1,)
    """

    def __init__(self, num_residual_blocks=5, num_filters=64, input_channels=14):
        super().__init__()

        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters

        # Initial convolution
        self.input_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters)
            for _ in range(num_residual_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(num_filters)
        self.value_head = ValueHead(num_filters)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 14, 8, 8)

        Returns:
            policy: Tensor of shape (batch, 4096) - move logits
            value: Tensor of shape (batch, 1) - position evaluation
        """
        # Initial block
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Dual heads
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_residual_blocks=5, num_filters=64):
    """
    Create chess model with specified configuration.

    Args:
        num_residual_blocks: Number of residual blocks
        num_filters: Number of convolutional filters

    Returns:
        ChessNet model
    """
    model = ChessNet(
        num_residual_blocks=num_residual_blocks,
        num_filters=num_filters
    )

    num_params = model.get_num_parameters()
    print(f"Created model with {num_params:,} parameters")
    print(f"  - Residual blocks: {num_residual_blocks}")
    print(f"  - Filters: {num_filters}")

    return model


def load_model(checkpoint_path, num_residual_blocks=5, num_filters=64, device='cuda'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        num_residual_blocks: Number of residual blocks
        num_filters: Number of filters
        device: Device to load model on

    Returns:
        Loaded model and checkpoint dict
    """
    model = create_model(num_residual_blocks, num_filters)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  - Loss: {checkpoint['loss']:.4f}")

    return model, checkpoint


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")

    # Small model (MVP)
    small_model = create_model(num_residual_blocks=5, num_filters=64)

    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 14, 8, 8)

    policy, value = small_model(test_input)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Policy output shape: {policy.shape}")
    print(f"  Value output shape: {value.shape}")

    # Medium model
    print("\n" + "="*50)
    medium_model = create_model(num_residual_blocks=10, num_filters=128)

    # Large model
    print("\n" + "="*50)
    large_model = create_model(num_residual_blocks=15, num_filters=256)
