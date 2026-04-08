"""
Training script for trauma detection model using YOLOv8.
"""

import argparse
from pathlib import Path
import yaml
import torch
from detection_model import TraumaDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train trauma detection model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/detect_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-yaml',
        type=str,
        required=True,
        help='Path to YAML file with dataset configuration'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8m',
        help='YOLOv8 model variant (n, s, m, l, x)'
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
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./runs/detect',
        help='Directory to save results'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience'
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
        print("Using default settings")
    
    # Verify data YAML exists
    if not Path(args.data_yaml).exists():
        print(f"Error: Dataset YAML file not found: {args.data_yaml}")
        print("Please create a YAML file with format:")
        print("""
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 5  # number of classes
names: ['liver_injury', 'spleen_injury', 'kidney_injury', 'bowel_injury', 'active_bleed']
        """)
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create detector
    print(f"Creating YOLOv8 detector with model: {args.model}...")
    detector = TraumaDetector(
        model_name=args.model,
        device=device,
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Train model
    print("Starting training...")
    results = detector.train(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        patience=args.patience
    )
    
    print("\nTraining completed!")
    print(f"Results saved to {args.output_dir}")
    
    # Export model
    print("Exporting model...")
    export_dir = Path(args.output_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    detector.export(str(export_dir), format='onnx')
    detector.export(str(export_dir), format='torchscript')
    
    print("Model exported successfully!")


if __name__ == "__main__":
    train()
