"""
Dataset loading and preprocessing utilities for Abdominal Trauma Detection.
Handles DICOM file loading, conversion, and data streaming without local storage.
"""

import sys
sys.path.insert(0, os.path.dirname(__file__))

import os
import numpy as np
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DicomDataset(Dataset):
    """
    Custom Dataset for loading DICOM CT scan images.
    
    Features:
    - Loads DICOM files from specified directory
    - Converts to PNG format
    - Creates 2.5D slices using consecutive frames
    - Normalizes pixel values
    - Supports streaming without loading entire dataset
    """
    
    def __init__(
        self,
        dicom_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        use_2_5d: bool = True,
        normalize: bool = True,
        cache_size: Optional[int] = None
    ):
        """
        Initialize DICOM dataset.
        
        Args:
            dicom_dir: Directory containing DICOM files
            image_size: Target image size (height, width)
            use_2_5d: Whether to create 2.5D slices from consecutive frames
            normalize: Whether to normalize pixel values
            cache_size: Number of samples to cache in memory (None = no caching)
        """
        self.dicom_dir = Path(dicom_dir)
        self.image_size = image_size
        self.use_2_5d = use_2_5d
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}
        
        # Collect all DICOM files
        self.dicom_files = sorted(list(self.dicom_dir.glob("**/*.dcm")))
        
        if not self.dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        
        print(f"Found {len(self.dicom_files)} DICOM files")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dicom_files)
    
    def _read_dicom(self, file_path: str) -> np.ndarray:
        """
        Read DICOM file and extract pixel array.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Pixel array as numpy array
        """
        try:
            dcm = pydicom.dcmread(file_path)
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Normalize to 0-255 range
            if pixel_array.min() != pixel_array.max():
                pixel_array = (pixel_array - pixel_array.min()) / \
                              (pixel_array.max() - pixel_array.min()) * 255
            
            return pixel_array
        except Exception as e:
            print(f"Error reading DICOM file {file_path}: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: resize and normalize.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Resize to target size
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        if self.normalize:
            image = image / 255.0
        
        return image
    
    def _create_2_5d_slice(self, idx: int) -> np.ndarray:
        """
        Create 2.5D slice using three consecutive frames.
        
        Args:
            idx: Index of middle frame
            
        Returns:
            2.5D image with shape (3, H, W)
        """
        prev_idx = max(0, idx - 1)
        next_idx = min(len(self.dicom_files) - 1, idx + 1)
        
        frames = []
        for frame_idx in [prev_idx, idx, next_idx]:
            image = self._read_dicom(str(self.dicom_files[frame_idx]))
            if image is not None:
                image = self._preprocess_image(image)
                frames.append(image)
        
        # Stack frames
        if len(frames) == 3:
            return np.stack(frames, axis=0)
        else:
            # Fallback: repeat single frame
            image = self._read_dicom(str(self.dicom_files[idx]))
            image = self._preprocess_image(image)
            return np.stack([image, image, image], axis=0)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Tensor of shape (C, H, W) or (H, W) depending on use_2_5d
        """
        # Check cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load image
        if self.use_2_5d:
            image = self._create_2_5d_slice(idx)
        else:
            image = self._read_dicom(str(self.dicom_files[idx]))
            image = self._preprocess_image(image)
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float()
        
        # Cache if enabled
        if self.cache_size is not None and len(self.cache) < self.cache_size:
            self.cache[idx] = tensor
        
        return tensor


class SegmentationDataset(Dataset):
    """Dataset for segmentation with image-mask pairs."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        transform=None
    ):
        """
        Initialize segmentation dataset.
        
        Args:
            images_dir: Directory with input images
            masks_dir: Directory with segmentation masks
            image_size: Target image size
            transform: Optional transforms to apply
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Get matching image-mask pairs
        self.image_files = sorted(list(self.images_dir.glob("*.png")) + 
                                 list(self.images_dir.glob("*.jpg")))
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and corresponding mask."""
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / (img_path.stem + "_mask.png")
        
        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        
        # Read mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size)
            mask = mask / 255.0
        else:
            mask = np.zeros_like(image)
        
        # Convert to tensors
        image = torch.from_numpy(np.expand_dims(image, 0)).float()
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask


def create_dataloaders(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        images_dir: Directory with images
        masks_dir: Directory with masks
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        image_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = SegmentationDataset(images_dir, masks_dir, image_size)
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("Dataset utilities loaded successfully")
