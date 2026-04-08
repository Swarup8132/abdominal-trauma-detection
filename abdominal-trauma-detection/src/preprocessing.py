"""
Image preprocessing utilities for CT scan data.
Handles DICOM conversion, normalization, and image preparation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Optional, List
from tqdm import tqdm
import os


class CTPreprocessor:
    """
    Preprocessor for CT scan images.
    Converts DICOM to standard image formats and applies normalization.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        window_center: int = 40,
        window_width: int = 400,
        normalize_method: str = "minmax"
    ):
        """
        Initialize CT preprocessor.
        
        Args:
            target_size: Target image dimensions (H, W)
            window_center: Hounsfield units window center for CT images
            window_width: Hounsfield units window width for CT images
            normalize_method: Normalization method ('minmax', 'standard', 'clip')
        """
        self.target_size = target_size
        self.window_center = window_center
        self.window_width = window_width
        self.normalize_method = normalize_method
    
    def read_dicom(self, file_path: str) -> Optional[np.ndarray]:
        """
        Read DICOM file and extract pixel array.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Pixel array or None if error
        """
        try:
            dcm = pydicom.dcmread(file_path)
            
            # Get pixel array
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Apply modality LUT if present
            if hasattr(dcm, 'PixelData'):
                rescale_intercept = float(dcm.RescaleIntercept or 0)
                rescale_slope = float(dcm.RescaleSlope or 1)
                pixel_array = pixel_array * rescale_slope + rescale_intercept
            
            return pixel_array
        except Exception as e:
            print(f"Error reading DICOM: {e}")
            return None
    
    def apply_window(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Apply Hounsfield window to CT image.
        
        Args:
            pixel_array: Input DICOM pixel array
            
        Returns:
            Windowed image
        """
        # Calculate window bounds
        lower_bound = self.window_center - self.window_width / 2
        upper_bound = self.window_center + self.window_width / 2
        
        # Apply windowing
        windowed = np.clip(pixel_array, lower_bound, upper_bound)
        
        # Normalize to 0-255
        windowed = (windowed - lower_bound) / (upper_bound - lower_bound) * 255
        
        return windowed.astype(np.uint8)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using specified method.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image in range [0, 1]
        """
        if self.normalize_method == "minmax":
            # Min-max normalization
            if image.min() == image.max():
                return np.zeros_like(image, dtype=np.float32)
            normalized = (image - image.min()) / (image.max() - image.min())
        
        elif self.normalize_method == "standard":
            # Standardization
            mean = image.mean()
            std = image.std()
            if std == 0:
                normalized = np.zeros_like(image, dtype=np.float32)
            else:
                normalized = (image - mean) / std
                normalized = np.clip(normalized, -3, 3) / 6 + 0.5
        
        elif self.normalize_method == "clip":
            # Clip and normalize
            percentile_2 = np.percentile(image, 2)
            percentile_98 = np.percentile(image, 98)
            clipped = np.clip(image, percentile_2, percentile_98)
            normalized = (clipped - percentile_2) / (percentile_98 - percentile_2)
        
        else:
            normalized = image / 255.0
        
        return normalized.astype(np.float32)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def preprocess(
        self,
        file_path: str,
        apply_window: bool = True,
        resize: bool = True,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            file_path: Path to DICOM file
            apply_window: Whether to apply Hounsfield window
            resize: Whether to resize to target size
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image or None if error
        """
        # Read DICOM
        pixel_array = self.read_dicom(file_path)
        if pixel_array is None:
            return None
        
        # Apply window
        if apply_window:
            pixel_array = self.apply_window(pixel_array)
        
        # Resize
        if resize:
            pixel_array = self.resize_image(pixel_array)
        
        # Normalize
        if normalize:
            pixel_array = self.normalize_image(pixel_array)
        
        return pixel_array
    
    def create_2_5d_image(
        self,
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Create 2.5D image from three consecutive slices.
        
        Args:
            images: List of three preprocessed images
            
        Returns:
            2.5D image with shape (3, H, W)
        """
        if len(images) != 3:
            raise ValueError("Expected 3 images for 2.5D creation")
        
        # Stack images
        image_2_5d = np.stack(images, axis=0)
        
        return image_2_5d


def batch_preprocess_dicoms(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    file_format: str = "png",
    max_files: Optional[int] = None
) -> int:
    """
    Batch preprocess all DICOM files in directory.
    
    Args:
        input_dir: Directory containing DICOM files
        output_dir: Output directory for processed images
        target_size: Target image size
        file_format: Output format ('png' or 'jpg')
        max_files: Maximum files to process (None = all)
        
    Returns:
        Number of successful conversions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all DICOM files
    dicom_files = sorted(list(input_path.glob("**/*.dcm")))
    
    if max_files:
        dicom_files = dicom_files[:max_files]
    
    preprocessor = CTPreprocessor(target_size=target_size)
    successful = 0
    
    print(f"Processing {len(dicom_files)} DICOM files...")
    
    for dicom_file in tqdm(dicom_files):
        # Preprocess
        processed = preprocessor.preprocess(str(dicom_file))
        
        if processed is None:
            continue
        
        # Convert to uint8 if needed
        if processed.dtype == np.float32:
            processed = (processed * 255).astype(np.uint8)
        
        # Save
        output_file = output_path / f"{dicom_file.stem}.{file_format}"
        cv2.imwrite(str(output_file), processed)
        successful += 1
    
    print(f"Successfully processed {successful}/{len(dicom_files)} files")
    return successful


def augment_image(image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply random augmentations to image.
    
    Args:
        image: Input image
        seed: Random seed for reproducibility
        
    Returns:
        Augmented image
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random brightness
    if np.random.rand() > 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(image.dtype)
    
    # Random contrast
    if np.random.rand() > 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        image = np.clip(image * contrast, 0, 255).astype(image.dtype)
    
    return image


if __name__ == "__main__":
    print("Preprocessing utilities loaded successfully")
