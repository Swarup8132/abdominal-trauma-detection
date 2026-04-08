"""
Detection model implementation using YOLOv8.
Detects trauma regions in abdominal CT scans.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from ultralytics import YOLO
import cv2


class TraumaDetector:
    """
    Trauma detection model using YOLOv8.
    Detects injury regions in abdominal organs.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8m",
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize trauma detector.
        
        Args:
            model_name: YOLOv8 model variant ('n', 's', 'm', 'l', 'x')
            device: Device to use ('cpu', 'cuda:0', etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load model
        self.model = YOLO(f'{model_name}.pt')
        self.model.to(self.device)
        
        # Class names for trauma detection
        self.class_names = {
            0: 'liver_injury',
            1: 'spleen_injury',
            2: 'kidney_injury',
            3: 'bowel_injury',
            4: 'active_bleed'
        }
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect trauma regions in image.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Dictionary with detection results
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Extract results
        detections = {
            'boxes': [],
            'confidences': [],
            'class_ids': [],
            'class_names': []
        }
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                detections['boxes'].append(box)
                detections['confidences'].append(conf)
                detections['class_ids'].append(class_id)
                detections['class_names'].append(
                    self.class_names.get(class_id, 'unknown')
                )
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect trauma in batch of images.
        
        Args:
            images: List of images
            
        Returns:
            List of detection dictionaries
        """
        return [self.detect(img) for img in images]
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        patience: int = 20
    ) -> Dict:
        """
        Train model on custom dataset.
        
        Args:
            data_yaml: Path to YAML file with dataset configuration
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Training results
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=patience,
            device=self.device,
            save=True,
            device_key='cuda' if 'cuda' in self.device else 'cpu'
        )
        
        return results
    
    def export(self, output_path: str, format: str = 'onnx'):
        """
        Export model to different formats.
        
        Args:
            output_path: Path to save exported model
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
        """
        self.model.export(format=format, save_dir=output_path)
        print(f"Model exported to {output_path} as {format}")
    
    def load_weights(self, weights_path: str):
        """
        Load pretrained weights.
        
        Args:
            weights_path: Path to weights file
        """
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        print(f"Weights loaded from {weights_path}")


class DetectionVisualizer:
    """Utility for visualizing detection results."""
    
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: Dict,
        color_map: Optional[Dict] = None,
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: Input image
            detections: Detection results from detector
            color_map: Mapping of class names to colors
            thickness: Bounding box thickness
            font_scale: Font size for text
            
        Returns:
            Image with drawn detections
        """
        output_image = image.copy()
        
        if color_map is None:
            # Default color map
            color_map = {
                'liver_injury': (0, 255, 0),
                'spleen_injury': (255, 0, 0),
                'kidney_injury': (0, 0, 255),
                'bowel_injury': (255, 255, 0),
                'active_bleed': (0, 255, 255)
            }
        
        for box, conf, class_name in zip(
            detections['boxes'],
            detections['confidences'],
            detections['class_names']
        ):
            x1, y1, x2, y2 = map(int, box)
            color = color_map.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, 1)[0]
            cv2.rectangle(output_image, (x1, y1 - text_size[1] - 4),
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(output_image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        return output_image


class DetectionMetrics:
    """Compute detection metrics."""
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.
        
        Boxes should be in format [x1, y1, x2, y2].
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU value
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_ap(
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Compute Average Precision.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            iou_threshold: IoU threshold for positive match
            
        Returns:
            Average precision value
        """
        tp = 0
        fp = 0
        
        for pred in predictions:
            best_iou = 0
            best_idx = -1
            
            for idx, gt in enumerate(ground_truth):
                iou = DetectionMetrics.compute_iou(
                    pred['box'], gt['box']
                )
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold:
                tp += 1
                ground_truth.pop(best_idx)
            else:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + len(ground_truth)) if ground_truth else 0.0
        
        ap = (precision * recall) if recall > 0 else 0.0
        return ap


if __name__ == "__main__":
    # Test detector initialization
    try:
        detector = TraumaDetector(model_name='yolov8n')
        print("Trauma detector initialized successfully")
    except Exception as e:
        print(f"Note: YOLOv8 requires training on custom trauma dataset")
        print(f"Error: {e}")
