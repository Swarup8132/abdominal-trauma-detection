"""
Training script for organ segmentation model.
Trains Swin-UNet for liver, spleen, kidney, and bowel segmentation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from segmentation_model import OrganSegmentationModel, SegmentationTrainer
from dataset import SegmentationDataset, create_dataloaders
from preprocessing import CTPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train segmentation model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/seg_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Directory with training images'
    )
    parser.add_argument(
        '--masks-dir',
        type=str,
        help='Directory with training masks'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train():
    """Main training function."""
    args = parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = {}
        print("No config file found, using default settings")
    
    # Update config with command line arguments
    if args.images_dir:
        config['images_dir'] = args.images_dir
    if args.masks_dir:
        config['masks_dir'] = args.masks_dir
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Creating segmentation model...")
    model = OrganSegmentationModel(
        in_channels=3,
        out_channels=4,
        img_size=(256, 256)
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        num_classes=4
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Create dataloaders
    print("Creating dataloaders...")
    if 'images_dir' in config and 'masks_dir' in config:
        train_loader, val_loader = create_dataloaders(
            images_dir=config['images_dir'],
            masks_dir=config['masks_dir'],
            batch_size=args.batch_size,
            num_workers=4,
            train_split=0.8,
            image_size=(256, 256)
        )
    else:
        print("Warning: No training data directories specified")
        print("Please provide --images-dir and --masks-dir arguments")
        return
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_dice = trainer.validate(val_loader)
        print(f"Validation Dice Score: {val_dice:.4f}")
        
        # Update learning rate
        trainer.scheduler.step(val_dice)
        
        # Save checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint_path = output_dir / f"best_model_dice_{val_dice:.4f}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch + 1, best_dice)
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch + 1, best_dice)
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(
        trainer.train_losses,
        trainer.val_dice_scores,
        output_dir / "training_curves.png"
    )
    
    print("Training completed!")


def plot_training_curves(losses, dice_scores, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Dice score curve
    ax2.plot(dice_scores, label='Validation Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    train()
