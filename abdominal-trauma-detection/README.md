# Abdominal Trauma Detection using CT Scan Images

## Overview

A deep learning project for detecting abdominal trauma (injuries) in CT scan images using a two-stage pipeline:
1. **Organ Segmentation**: Swin-UNet with MONAI to segment liver, spleen, kidney, and bowel
2. **Trauma Detection**: YOLOv8 for localizing injury regions

This project is designed to work with the **RSNA Abdominal Trauma Detection** dataset (~450GB) without requiring local storage of the entire dataset. The training is performed on **Google Colab**, while development is done locally in VS Code.

## Dataset

**Dataset Name**: RSNA Abdominal Trauma Detection  
**Source**: [Kaggle](https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection)  
**Size**: ~450GB (not for local download)  
**Format**: DICOM CT scan images with organ segmentation annotations

### Dataset Handling Strategy

- **Local Development**: Code development only (no dataset storage)
- **Google Colab**: Dataset download and training
  - Uses Kaggle API for automated dataset access
  - Mounts Google Drive for checkpoint storage
  - Supports streaming and on-the-fly preprocessing

## Project Structure

```
abdominal-trauma-detection/
├── notebooks/
│   └── colab_training.ipynb          # Google Colab training notebook
├── src/
│   ├── dataset.py                    # Dataset loading utilities
│   ├── preprocessing.py              # CT image preprocessing
│   ├── segmentation_model.py         # Swin-UNet segmentation model
│   ├── detection_model.py            # YOLOv8 detection model
│   ├── train_segmentation.py         # Segmentation training script
│   ├── train_detection.py            # Detection training script
│   ├── inference.py                  # Inference pipeline
│   └── visualization.py              # Result visualization utilities
├── configs/
│   ├── seg_config.yaml              # Segmentation config
│   ├── detect_config.yaml           # Detection config
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Local Setup (VS Code)

1. **Clone the repository**
```bash
git clone <repository-url>
cd abdominal-trauma-detection
```

2. **Create virtual environment**
```bash
# Using Python venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n trauma-detection python=3.10
conda activate trauma-detection
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
python -c "import ultralytics; print('YOLOv8: OK')"
```

## Model Architecture

### Segmentation Model: Swin-UNet

**Purpose**: Segment 4 abdominal organs (liver, spleen, kidney, bowel)

**Architecture**:
- Input: 3-channel 2.5D CT images (256×256 pixels)
- Encoder: Swin Transformer blocks for feature extraction
- Decoder: Upsampling with skip connections
- Output: 4-channel segmentation masks

**Key Features**:
- Transformer-based architecture for better context understanding
- Handles variable slice thickness via 2.5D approach
- Dice loss for imbalanced class handling
- Checkpoint saving and resumable training

### Detection Model: YOLOv8

**Purpose**: Detect trauma regions within segmented organs

**Architecture**:
- Backbone: YOLOv8 CSPDarknet
- Neck: FPN-like multi-scale feature fusion
- Head: Detection heads for 5 trauma classes

**Trauma Classes**:
1. Liver injury
2. Spleen injury
3. Kidney injury
4. Bowel injury
5. Active bleeding

## How to Use

### 1. Local Development in VS Code

#### Option A: Prepare Data Locally (if available)

```bash
# Place DICOM files in data/dicoms/
# Preprocess to PNG format
python -c "
from src.preprocessing import batch_preprocess_dicoms
batch_preprocess_dicoms(
    input_dir='data/dicoms',
    output_dir='data/images'
)
"
```

#### Option B: Skip Local Data (Recommended)

The code supports training without local data. Follow the Google Colab workflow below.

### 2. Training on Google Colab

1. **Open the notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `notebooks/colab_training.ipynb`

2. **Set up Kaggle API**
   ```python
   # In Colab, upload kaggle.json from Kaggle account settings
   ```

3. **Run the notebook cells in order**
   - Install libraries
   - Authenticate with Kaggle API
   - Download dataset
   - Preprocess images
   - Train segmentation model
   - Train detection model
   - Evaluate results

4. **Save checkpoints to Google Drive**
   - Models are automatically saved to Drive
   - Download locally after training

### 3. Inference

#### Using a trained model

```python
from src.inference import TraumaDetectionPipeline

# Initialize pipeline
pipeline = TraumaDetectionPipeline(
    segmentation_weights='path/to/seg_weights.pt',
    detection_weights='path/to/det_weights.pt',
    device='cuda'
)

# Process image
results = pipeline.process(
    image_path='path/to/ct_image.png',
    save_dir='outputs/'
)

# Access results
print(f"Organs detected: {results['organ_crops'].keys()}")
print(f"Trauma regions: {results['detections']['class_names']}")
```

#### Command line inference

```bash
python src/inference.py \
    --image path/to/image.png \
    --seg-weights checkpoints/best_seg_model.pt \
    --det-weights checkpoints/best_det_model.pt \
    --output-dir results/
```

### 4. Training Locally (if data available)

#### Segmentation training

```bash
python src/train_segmentation.py \
    --images-dir data/images \
    --masks-dir data/masks \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 0.001
```

#### Detection training

Create a `data/trauma_detection.yaml`:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 5
names: ['liver_injury', 'spleen_injury', 'kidney_injury', 'bowel_injury', 'active_bleed']
```

Then train:
```bash
python src/train_detection.py \
    --data-yaml data/trauma_detection.yaml \
    --model yolov8m \
    --epochs 100 \
    --batch-size 16
```

## Key Features

### 1. No Local Dataset Required
- Code works without downloading full dataset
- Streaming support for Colab
- Modular preprocessing pipeline

### 2. Modular Design
- **dataset.py**: Loading, caching, streaming
- **preprocessing.py**: DICOM conversion, windowing, normalization
- **segmentation_model.py**: Organ segmentation
- **detection_model.py**: Trauma detection
- **inference.py**: Complete end-to-end pipeline
- **visualization.py**: Results visualization

### 3. Production-Ready
- Checkpoint saving and resumable training
- Error handling and logging
- Configuration file support
- Batch processing capabilities

### 4. Google Colab Optimized
- Automatic dataset downloading
- GPU acceleration (T4/P100)
- Google Drive integration
- Memory-efficient training

## Configuration

### Segmentation Config (`seg_config.yaml`)
```yaml
model:
  name: "swin_unet"
  out_channels: 4  # organs
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
```

### Detection Config (`detect_config.yaml`)
```yaml
model:
  name: "yolov8m"
  num_classes: 5  # trauma types
training:
  epochs: 100
  batch_size: 16
```

## Example Workflow

### Quick Start: Google Colab
```
1. Open colab_training.ipynb in Colab
2. Run "Install libraries" cell
3. Setup Kaggle authentication
4. Run "Download dataset" cell
5. Train segmentation model
6. Train detection model
7. Run inference on test samples
8. Download trained weights
```

### Full Workflow: Local + Colab
```
1. Clone repository locally (VS Code)
2. Develop/test code locally
3. Push to GitHub
4. Open colab_training.ipynb in Colab
5. Setup and train
6. Pull weights back to local
7. Test inference locally
```

## Results and Metrics

### Segmentation
- **Dice Score**: Similarity between predicted and ground truth masks
- **IoU Score**: Intersection over Union for each organ
- **Per-organ metrics**: Individual performance for each organ class

### Detection
- **mAP (mean Average Precision)**: Overall detection performance
- **Precision/Recall**: Per-class detection metrics
- **Confusion Matrix**: Class-wise classification accuracy

## Citation and References

**RSNA Abdominal Trauma Detection Dataset**
```
RSNA 2023 Abdominal Trauma Detection Challenge
https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection
```

**Key Papers**:
- Swin Transformer: https://arxiv.org/abs/2103.14030
- MONAI: https://arxiv.org/abs/2104.13963
- YOLOv8: https://github.com/ultralytics/ultralytics

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use smaller model variant
```python
# In config or code
batch_size = 4  # Instead of 8
model_name = "yolov8n"  # Instead of yolov8m
```

### Issue: DICOM files not loading
**Solution**: Check file format and permissions
```python
import pydicom
dcm = pydicom.dcmread('file.dcm')
print(dcm)  # Verify readable
```

### Issue: Model not converging
**Solution**: Adjust learning rate and check data
```python
learning_rate = 0.0001  # Start smaller
# Verify data preprocessing is correct
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 8GB+ GPU VRAM (16GB recommended)
- Internet connection (for Colab)

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please create an issue on GitHub.

---

**Happy Training!** 🚀

For detailed step-by-step guidance on Colab training, see `notebooks/colab_training.ipynb`.
