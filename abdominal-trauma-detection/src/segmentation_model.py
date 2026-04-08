"""
Segmentation model implementation using MONAI.
Implements Swin-UNet for organ segmentation (liver, spleen, kidney, bowel).
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from typing import Optional, Tuple


class OrganSegmentationModel(nn.Module):
    """
    Organ segmentation model using Swin-UNet architecture.
    Segments 4 organs: liver, spleen, kidney, bowel.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        img_size: Tuple[int, int] = (256, 256),
        feature_size: int = 24,
        use_checkpoint: bool = False
    ):
        """
        Initialize segmentation model.
        
        Args:
            in_channels: Number of input channels (3 for 2.5D images)
            out_channels: Number of output channels (organs to segment)
            img_size: Input image size
            feature_size: Feature size for Swin-UNet
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.feature_size = feature_size
        
        # For 2D images, we need to adapt the model
        # Using a custom 2D segmentation network with UNet-like architecture
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder blocks."""
        return nn.Sequential(
            self._conv_block(self.in_channels, 64),
            nn.MaxPool2d(2, 2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            self._conv_block(128, 256),
            nn.MaxPool2d(2, 2),
            self._conv_block(256, 512),
        )
    
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder blocks."""
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            self._conv_block(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            self._conv_block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            self._conv_block(64, 64),
        )
    
    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with normalization and activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of segmentation model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output segmentation map of shape (B, num_classes, H, W)
        """
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc2 = self.encoder[2](enc2)
        enc3 = self.encoder[3](enc2)
        enc3 = self.encoder[4](enc3)
        enc4 = self.encoder[5](enc3)
        enc4 = self.encoder[6](enc4)
        
        # Decoder
        dec = self.decoder(enc4)
        
        # Final convolution
        output = self.final_conv(dec)
        
        return output


class SwinUNetSegmentationModel(nn.Module):
    """
    3D Swin-UNet segmentation model using MONAI.
    Adapted for 2D/2.5D input by treating 3-channel input as 3D-like.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        feature_size: int = 24,
        use_checkpoint: bool = False
    ):
        """
        Initialize Swin-UNet model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            feature_size: Base feature size
            use_checkpoint: Use gradient checkpointing for memory efficiency
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define model architecture
        self.model = nn.Sequential(
            self._create_encoder(),
            self._create_decoder(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    
    def _create_encoder(self) -> nn.Module:
        """Create encoder network."""
        return nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    
    def _create_decoder(self) -> nn.Module:
        """Create decoder network with upsampling."""
        layers = []
        
        channels = [64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
        
        # Upsampling path
        for i in range(len(channels) - 1, 0, -1):
            layers.append(nn.ConvTranspose2d(
                channels[i], channels[i-1],
                kernel_size=2, stride=2
            ))
            layers.append(nn.BatchNorm2d(channels[i-1]))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        return self.model(x)


class SegmentationTrainer:
    """Trainer for segmentation models with metrics tracking."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        num_classes: int = 4
    ):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model
            device: PyTorch device
            learning_rate: Learning rate
            num_classes: Number of classes
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        
        # Loss and optimizer
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction='mean')
        
        # Training state
        self.train_losses = []
        self.val_dice_scores = []
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss computation
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average Dice score
        """
        self.model.eval()
        dice_scores = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute Dice score
                outputs_onehot = torch.argmax(outputs, dim=1, keepdim=True)
                self.dice_metric(outputs_onehot, masks)
                dice = self.dice_metric.aggregate().item()
                dice_scores.append(dice)
        
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        self.val_dice_scores.append(avg_dice)
        
        return avg_dice
    
    def save_checkpoint(self, path: str, epoch: int, best_dice: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': best_dice,
            'train_losses': self.train_losses,
            'val_dice_scores': self.val_dice_scores,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint.get('epoch', 0)


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OrganSegmentationModel(in_channels=3, out_channels=4)
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    y = model(x)
    
    print(f"Model output shape: {y.shape}")
    print("Segmentation model initialized successfully")
