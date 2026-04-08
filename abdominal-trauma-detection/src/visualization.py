"""
Visualization utilities for CT scan analysis results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class CTVisualizer:
    """Utilities for visualizing CT scan analysis."""
    
    @staticmethod
    def create_comparison_image(
        original: np.ndarray,
        segmentation: np.ndarray,
        detection: Optional[np.ndarray] = None,
        title: str = "Trauma Detection Results"
    ) -> np.ndarray:
        """
        Create comparison image with original, segmentation, and detection.
        
        Args:
            original: Original CT image
            segmentation: Segmentation mask image
            detection: Detection visualization image
            title: Title for the figure
            
        Returns:
            Combined comparison image
        """
        # Ensure all images are uint8
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        if segmentation.max() <= 1.0:
            segmentation = (segmentation * 255).astype(np.uint8)
        
        # Convert grayscale to 3-channel if needed
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(segmentation.shape) == 2:
            segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
        if detection is not None and len(detection.shape) == 2:
            detection = cv2.cvtColor(detection, cv2.COLOR_GRAY2BGR)
        
        # Resize to same size
        h, w = original.shape[:2]
        segmentation = cv2.resize(segmentation, (w, h))
        if detection is not None:
            detection = cv2.resize(detection, (w, h))
        
        # Create composite
        if detection is not None:
            composite = np.hstack([original, segmentation, detection])
            labels = ['Original', 'Segmentation', 'Detection']
        else:
            composite = np.hstack([original, segmentation])
            labels = ['Original', 'Segmentation']
        
        return composite
    
    @staticmethod
    def overlay_segmentation(
        image: np.ndarray,
        segmentation_mask: np.ndarray,
        alpha: float = 0.5,
        organ_colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Overlay segmentation mask on original image.
        
        Args:
            image: Original image
            segmentation_mask: Segmentation mask
            alpha: Transparency level (0-1)
            organ_colors: List of colors for each organ
            
        Returns:
            Overlaid image
        """
        if organ_colors is None:
            organ_colors = [
                (255, 0, 0),    # liver - red
                (0, 255, 0),    # spleen - green
                (0, 0, 255),    # kidney - blue
                (255, 255, 0),  # bowel - yellow
            ]
        
        # Ensure image is 3-channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        overlay = image.copy().astype(np.float32)
        
        for organ_idx, color in enumerate(organ_colors):
            mask = (segmentation_mask == organ_idx + 1)
            overlay[mask] = (
                overlay[mask] * (1 - alpha) + 
                np.array(color) * alpha
            )
        
        return overlay.astype(np.uint8)
    
    @staticmethod
    def create_multi_slice_visualization(
        images: List[np.ndarray],
        segmentations: List[np.ndarray],
        detections: Optional[List[np.ndarray]] = None,
        rows: int = 3,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Create visualization of multiple slices.
        
        Args:
            images: List of original images
            segmentations: List of segmentation masks
            detections: List of detection visualizations (optional)
            rows: Number of rows in grid
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        num_slices = len(images)
        
        if detections is None:
            cols = 2
        else:
            cols = 3
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx in range(min(num_slices, rows)):
            # Original
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.imshow(images[idx], cmap='gray')
            ax1.set_title(f'Slice {idx + 1}')
            ax1.axis('off')
            
            # Segmentation
            ax2 = fig.add_subplot(gs[idx, 1])
            ax2.imshow(segmentations[idx])
            ax2.set_title(f'Segmentation {idx + 1}')
            ax2.axis('off')
            
            # Detection
            if detections is not None:
                ax3 = fig.add_subplot(gs[idx, 2])
                ax3.imshow(detections[idx])
                ax3.set_title(f'Detection {idx + 1}')
                ax3.axis('off')
        
        return fig
    
    @staticmethod
    def plot_organ_statistics(
        segmentation_masks: List[np.ndarray],
        organ_names: List[str],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot organ segmentation statistics.
        
        Args:
            segmentation_masks: List of segmentation masks
            organ_names: Names of organs
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate organ areas
        organ_areas = {name: [] for name in organ_names}
        
        for mask in segmentation_masks:
            for idx, name in enumerate(organ_names):
                area = np.sum(mask == idx + 1)
                organ_areas[name].append(area)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for name in organ_names:
            ax.plot(organ_areas[name], label=name, marker='o')
        
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('Organ Area (pixels²)')
        ax.set_title('Organ Segmentation Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def create_heatmap(
        data: np.ndarray,
        title: str = "Heatmap",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Create heatmap visualization.
        
        Args:
            data: 2D data array
            title: Title for heatmap
            cmap: Colormap name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax)
        
        return fig
    
    @staticmethod
    def save_figure_with_text(
        figure: plt.Figure,
        save_path: str,
        text_info: Optional[str] = None,
        dpi: int = 150
    ):
        """
        Save figure with optional text information.
        
        Args:
            figure: Matplotlib figure
            save_path: Path to save figure
            text_info: Optional text to add to figure
            dpi: Resolution in DPI
        """
        if text_info:
            figure.text(0.5, 0.02, text_info, ha='center', fontsize=10)
        
        figure.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def visualize_batch_results(
    batch_results: List[Dict],
    output_dir: str,
    batch_name: str = "batch"
):
    """
    Visualize results from a batch of images.
    
    Args:
        batch_results: List of result dictionaries
        output_dir: Output directory
        batch_name: Name for batch
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, result in enumerate(batch_results):
        # Create comparison
        composite = CTVisualizer.create_comparison_image(
            original=result['original_image'],
            segmentation=result['segmentation'],
            detection=result.get('detection'),
            title=f"Sample {idx + 1}"
        )
        
        # Save
        cv2.imwrite(
            str(output_path / f"{batch_name}_{idx:03d}_composite.png"),
            composite
        )


def create_report(
    results: Dict,
    output_path: str,
    patient_id: Optional[str] = None
):
    """
    Create a comprehensive report with visualizations.
    
    Args:
        results: Processing results dictionary
        output_path: Path to save report
        patient_id: Optional patient ID
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    title = f"Abdominal Trauma Detection Report"
    if patient_id:
        title += f" - Patient {patient_id}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['original_image'], cmap='gray')
    ax1.set_title('Original CT Image')
    ax1.axis('off')
    
    # Segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['segmentation'])
    ax2.set_title('Organ Segmentation')
    ax2.axis('off')
    
    # Detection overlay
    if 'visualization' in results:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(results['visualization'])
        ax3.set_title('Trauma Detection')
        ax3.axis('off')
    
    # Organ crops
    organ_crops = results.get('organ_crops', {})
    for idx, (organ_name, crop) in enumerate(organ_crops.items()):
        if idx >= 3:
            break
        ax = fig.add_subplot(gs[1, idx])
        if len(crop.shape) == 3:
            ax.imshow(crop)
        else:
            ax.imshow(crop, cmap='gray')
        ax.set_title(f"{organ_name.capitalize()} Crop")
        ax.axis('off')
    
    # Statistics
    ax = fig.add_subplot(gs[2, :])
    detections = results.get('detections', {})
    
    if detections.get('boxes'):
        detection_text = "Detected Trauma Regions:\n"
        for class_name, conf in zip(detections['class_names'], detections['confidences']):
            detection_text += f"• {class_name}: {conf:.2%} confidence\n"
    else:
        detection_text = "No trauma regions detected"
    
    ax.text(0.5, 0.5, detection_text, ha='center', va='center',
           fontsize=12, transform=ax.transAxes, family='monospace')
    ax.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Report saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
