"""
Inference pipeline for complete trauma detection workflow.
Combines segmentation and detection models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import argparse

from segmentation_model import OrganSegmentationModel
from detection_model import TraumaDetector, DetectionVisualizer
from preprocessing import CTPreprocessor
from visualization import create_comparison_image


class TraumaDetectionPipeline:
    """
    Complete pipeline for abdominal trauma detection.
    
    Steps:
    1. Load CT image
    2. Preprocess (normalize, resize)
    3. Segment organs using Swin-UNet
    4. Crop segmented organs
    5. Detect trauma using YOLOv8
    6. Visualize results
    """
    
    def __init__(
        self,
        segmentation_weights: str,
        detection_weights: Optional[str] = None,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize detection pipeline.
        
        Args:
            segmentation_weights: Path to segmentation model weights
            detection_weights: Path to detection model weights
            device: Device to use ('cuda' or 'cpu')
            image_size: Input image size
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.image_size = image_size
        
        # Initialize preprocessor
        self.preprocessor = CTPreprocessor(target_size=image_size)
        
        # Load segmentation model
        print("Loading segmentation model...")
        self.segmentation_model = OrganSegmentationModel(
            in_channels=3,
            out_channels=4,
            img_size=image_size
        )
        
        if Path(segmentation_weights).exists():
            checkpoint = torch.load(segmentation_weights, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.segmentation_model.load_state_dict(checkpoint)
            print(f"Loaded segmentation weights from {segmentation_weights}")
        else:
            print(f"Warning: Segmentation weights not found at {segmentation_weights}")
            print("Using randomly initialized segmentation model")
        
        self.segmentation_model = self.segmentation_model.to(self.device)
        self.segmentation_model.eval()
        
        # Load detection model
        if detection_weights:
            print("Loading detection model...")
            self.detector = TraumaDetector(device=str(self.device))
            self.detector.load_weights(detection_weights)
        else:
            print("Detection model not specified - detection disabled")
            self.detector = None
        
        # Organ names
        self.organ_names = ['liver', 'spleen', 'kidney', 'bowel']
        self.organ_colors = [
            (255, 0, 0),    # liver - red
            (0, 255, 0),    # spleen - green
            (0, 0, 255),    # kidney - blue
            (255, 255, 0),  # bowel - yellow
        ]
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Segment organs in CT image.
        
        Args:
            image: Input CT image (H, W) or (H, W, C)
            
        Returns:
            Segmentation mask and results dictionary
        """
        # Ensure 3-channel input
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Preprocess
        processed = cv2.resize(image, self.image_size)
        processed = processed / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.segmentation_model(tensor)
        
        # Get segmentation mask
        segmentation_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
        
        # Create colored segmentation
        colored_mask = self._colorize_segmentation(segmentation_mask)
        
        results = {
            'segmentation_mask': segmentation_mask,
            'colored_mask': colored_mask,
            'input_image': processed
        }
        
        return colored_mask, results
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect trauma regions in image.
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        if self.detector is None:
            print("Warning: Detection model not loaded")
            return {'boxes': [], 'confidences': [], 'class_ids': [], 'class_names': []}
        
        # Ensure image is in appropriate format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        detections = self.detector.detect(image)
        return detections
    
    def crop_organs(
        self,
        image: np.ndarray,
        segmentation_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Crop individual organs from segmented image.
        
        Args:
            image: Original image
            segmentation_mask: Segmentation mask
            
        Returns:
            Dictionary of organ crops
        """
        crops = {}
        
        for organ_idx, organ_name in enumerate(self.organ_names):
            # Get organ pixels
            organ_mask = (segmentation_mask == organ_idx + 1).astype(np.uint8)
            
            if organ_mask.sum() == 0:
                continue
            
            # Find bounding box
            contours = cv2.findContours(organ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            if len(contours) == 0:
                continue
            
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Crop with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            crop = image[y1:y2, x1:x2]
            crops[organ_name] = crop
        
        return crops
    
    def process(
        self,
        image_path: str,
        save_dir: Optional[str] = None,
        visualize: bool = True
    ) -> Dict:
        """
        Complete processing pipeline.
        
        Args:
            image_path: Path to input image
            save_dir: Directory to save outputs
            visualize: Whether to create visualizations
            
        Returns:
            Processing results
        """
        # Load image
        print(f"Loading image: {image_path}")
        if image_path.lower().endswith('.dcm'):
            image = self.preprocessor.preprocess(image_path, apply_window=True)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Segment
        print("Performing segmentation...")
        colored_mask, seg_results = self.segment(image)
        
        # Crop organs
        print("Cropping organs...")
        crops = self.crop_organs(image, seg_results['segmentation_mask'])
        
        # Detect trauma
        print("Detecting trauma...")
        detections = self.detect(image)
        
        # Visualize detections
        if detections['boxes']:
            viz_image = DetectionVisualizer.draw_detections(image, detections)
        else:
            viz_image = image
        
        results = {
            'original_image': image,
            'segmentation': colored_mask,
            'detections': detections,
            'organ_crops': crops,
            'visualization': viz_image
        }
        
        # Save results
        if save_dir:
            self._save_results(results, save_dir, image_path)
        
        return results
    
    def _colorize_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """Create colored segmentation visualization."""
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for idx, color in enumerate(self.organ_colors):
            colored[mask == idx + 1] = color
        
        return colored
    
    def _save_results(self, results: Dict, save_dir: str, image_path: str):
        """Save results to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        stem = Path(image_path).stem
        
        # Save images
        cv2.imwrite(str(save_path / f"{stem}_original.png"), results['original_image'])
        cv2.imwrite(str(save_path / f"{stem}_segmentation.png"), results['segmentation'])
        cv2.imwrite(str(save_path / f"{stem}_detection.png"), results['visualization'])
        
        print(f"Results saved to {save_dir}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run trauma detection inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--seg-weights', type=str, required=True, help='Segmentation model weights')
    parser.add_argument('--det-weights', type=str, default=None, help='Detection model weights')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TraumaDetectionPipeline(
        segmentation_weights=args.seg_weights,
        detection_weights=args.det_weights,
        device=args.device
    )
    
    # Process image
    results = pipeline.process(
        image_path=args.image,
        save_dir=args.output_dir,
        visualize=True
    )
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
