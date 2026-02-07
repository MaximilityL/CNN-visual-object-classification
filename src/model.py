import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

class ConvBlock(nn.Module):
    """
    A single convolutional block consisting of Conv2D + Activation + optional BatchNorm + MaxPool.
    This implements basic building block as specified in architecture.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_pooling: bool = True,
                 pool_size: int = 2,
                 pool_stride: int = 2,
                 activation: str = 'relu',
                 batch_norm: bool = False):
        """
        Initialize a convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            padding: Padding for convolution
            use_pooling: Whether to use max pooling
            pool_size: Size of pooling kernel
            pool_stride: Stride for pooling operation
            activation: Activation function ('relu', 'gelu', etc.)
            batch_norm: Whether to use batch normalization
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        # Select activation function
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'swish':
            self.activation = nn.SiLU()
        else:  # Default to ReLU
            self.activation = nn.ReLU(inplace=True)
        
        self.use_pooling = use_pooling
        self.batch_norm = batch_norm
        
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if use_pooling:
            self.pool = nn.MaxPool2d(pool_size, stride=pool_stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional block."""
        x = self.conv(x)
        
        # Apply batch normalization if enabled
        if self.batch_norm:
            x = self.bn(x)
        
        # Apply activation function
        x = self.activation(x)
        
        # Apply pooling if enabled
        if self.use_pooling:
            x = self.pool(x)
        
        return x

class CompactCNN(nn.Module):
    """
    Compact CNN Classifier as specified in the architecture.
    
    Architecture flow:
    Input -> [ConvBlock] -> [ConvBlock] -> ... -> Flatten -> FC -> Softmax
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 196,
                 num_conv_blocks: int = 2,
                 conv_filters: List[int] = [64, 128],
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_pooling: bool = True,
                 pool_size: int = 2,
                 pool_stride: int = 2,
                 dropout_rate: float = 0.5,
                 fc_hidden_size: int = 512,
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 residual: bool = False):
        """
        Initialize the Compact CNN.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (196 for Stanford Cars)
            num_conv_blocks: Number of convolutional blocks
            conv_filters: List of filter counts for each conv block
            kernel_size: Size of convolution kernels
            padding: Padding for convolutions
            use_pooling: Whether to use max pooling after each conv block
            pool_size: Size of pooling kernel
            pool_stride: Stride for pooling
            dropout_rate: Dropout rate for regularization
            fc_hidden_size: Size of the fully connected hidden layer
            activation: Activation function ('relu', 'gelu', etc.)
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
        """
        super(CompactCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_conv_blocks = num_conv_blocks
        
        # Validate inputs
        if len(conv_filters) != num_conv_blocks:
            raise ValueError(f"Length of conv_filters ({len(conv_filters)}) must match num_conv_blocks ({num_conv_blocks})")
        
        # Build feature extractor (convolutional blocks)
        self.feature_extractor = nn.ModuleList()
        
        # First conv block
        self.feature_extractor.append(
            ConvBlock(input_channels, conv_filters[0], kernel_size, padding, use_pooling, pool_size, pool_stride, activation, batch_norm)
        )
        
        # Remaining conv blocks
        for i in range(1, num_conv_blocks):
            self.feature_extractor.append(
                ConvBlock(conv_filters[i-1], conv_filters[i], kernel_size, padding, use_pooling, pool_size, pool_stride, activation, batch_norm)
            )
        
        # Calculate the size of flattened features
        # This will be computed dynamically in the first forward pass
        self.flattened_size = None
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(conv_filters[-1], fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_size, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logging.info(f"CompactCNN initialized with {num_conv_blocks} conv blocks, {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Feature extraction through convolutional blocks
        for conv_block in self.feature_extractor:
            x = conv_block(x)
        
        # Classification head
        logits = self.classifier(x)
        
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get feature maps from each convolutional block for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps from each conv block
        """
        feature_maps = []
        
        for conv_block in self.feature_extractor:
            x = conv_block(x)
            feature_maps.append(x)
        
        return feature_maps
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        return predicted_classes, probabilities
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params, trainable_params = self.count_parameters()
        
        summary = f"""
CompactCNN Model Summary:
========================
Input Channels: {self.feature_extractor[0].conv.in_channels}
Number of Classes: {self.num_classes}
Number of Conv Blocks: {self.num_conv_blocks}

Architecture:
Input -> """
        
        for i, conv_block in enumerate(self.feature_extractor):
            in_ch = conv_block.conv.in_channels
            out_ch = conv_block.conv.out_channels
            summary += f"ConvBlock{i+1}({in_ch}->{out_ch}) -> "
        
        summary += "GlobalAvgPool -> Flatten -> Dropout -> FC -> ReLU -> Dropout -> FC(Output)"
        
        summary += f"""

Parameters:
- Total: {total_params:,}
- Trainable: {trainable_params:,}
"""
        return summary

def create_model_from_config(config) -> CompactCNN:
    """
    Create a CompactCNN model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized CompactCNN model
    """
    return CompactCNN(
        input_channels=config.get('model.input_channels', 3),
        num_classes=config.get('dataset.num_classes', 196),
        num_conv_blocks=config.get('model.num_conv_blocks', 2),
        conv_filters=config.get('model.conv_filters', [64, 128]),
        kernel_size=config.get('model.kernel_size', 3),
        padding=config.get('model.padding', 1),
        use_pooling=config.get('model.use_pooling', True),
        pool_size=config.get('model.pool_size', 2),
        pool_stride=config.get('model.pool_stride', 2),
        dropout_rate=config.get('model.dropout_rate', 0.5),
        fc_hidden_size=config.get('model.fc_hidden_size', 512),
        activation=config.get('model.activation', 'relu'),
        batch_norm=config.get('model.batch_norm', False),
        residual=config.get('model.residual', False)
    )
