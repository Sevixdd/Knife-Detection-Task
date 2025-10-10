# Knife Detection Task

A deep learning project for knife detection using various CNN architectures including EfficientNet, ResNet, and custom FBSD models. This project implements multiple approaches for knife classification with 192 different classes.

## 🎯 Project Overview

This project focuses on knife detection and classification using state-of-the-art deep learning models. It includes multiple model architectures and training strategies to achieve optimal performance on knife detection tasks.

**Performance Note**: The project uses Parquet format for data storage, which provides a **3x speed improvement** over traditional CSV format during training.

## 🏗️ Architecture

The project implements several model architectures:

### 1. EfficientNet Models
- **EfficientNet-B0**: Lightweight and efficient model for fast inference
- **EfficientNet-V2-B0**: Improved version with better training dynamics

### 2. ResNet Models
- **ResNet-18**: Standard ResNet architecture
- **ResNet-50**: Deeper ResNet with custom FBSD (Fine-grained Backbone with Strip Detection) architecture
- **ResNet-101**: Even deeper ResNet variant

### 3. Custom FBSD Architecture
- **Fine-grained Backbone with Strip Detection**: Custom architecture that combines multiple feature levels
- **Cross-level attention mechanism**: Enhances feature representation
- **Multi-scale feature fusion**: Combines features from different network depths

## 📁 Project Structure

```
Knife-Detection-Task/
├── main/                          # Main source code
│   ├── backbone.py                # ResNet and DenseNet backbone implementations
│   ├── config.py                  # Configuration parameters
│   ├── csv_to_parquet.py         # Data preprocessing utilities
│   ├── data.py                    # Dataset class and data loaders
│   ├── model.py                   # Custom FBSD model architecture
│   ├── train.py                   # EfficientNet training script
│   ├── trainResnet.py            # ResNet training script
│   ├── test.py                    # Model testing script
│   ├── utils.py                   # Utility functions and loss functions
│   ├── requirements.txt           # Python dependencies
│   ├── train.parquet             # Training data
│   ├── val.parquet               # Validation data
│   └── test.parquet              # Test data
├── Effb0-Best/                   # Best EfficientNet-B0 results
├── EffV2B0/                      # EfficientNet-V2-B0 results
├── Resnet AdamW 0002/            # ResNet with AdamW optimizer results
├── ResnetSGD/                    # ResNet with SGD optimizer results
└── logs/                         # Training logs and outputs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- PyTorch 1.12.1+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Knife-Detection-Task
```

2. Install dependencies:
```bash
pip install -r main/requirements.txt
```

3. Prepare your data:
   - Place your knife images in the appropriate directory
   - Update the data paths in the configuration files
   - **Convert CSV data to Parquet format using `csv_to_parquet.py`** (provides 3x faster training)

### Training

#### EfficientNet Training
```bash
cd main
python train.py
```

#### ResNet Training
```bash
cd main
python trainResnet.py
```

### Testing
```bash
cd main
python test.py
```

## ⚙️ Configuration

Key parameters can be modified in `config.py`:

```python
class DefaultConfigs(object):
    n_classes = 192          # Number of classes
    img_weight = 224         # Image width
    img_height = 224         # Image height
    batch_size = 32          # Batch size
    epochs = 30              # Number of epochs
    learning_rate = 0.0002   # Learning rate
    new_weight_decay = 0.0001 # Weight decay
```

## 🧠 Model Details

### FBSD Architecture Features

1. **Multi-level Feature Extraction**: Extracts features from three different network depths
2. **Cross-level Attention**: Implements Feature Distribution Matching (FDM) for attention
3. **Strip-based Processing**: Uses Fine-grained Strip-based Module (FSBM) for enhanced feature processing
4. **Ensemble Prediction**: Combines predictions from multiple levels for final classification

### Training Features

- **Mixed Precision Training**: Uses PyTorch's automatic mixed precision for faster training
- **Learning Rate Scheduling**: Cosine annealing learning rate scheduler
- **Data Augmentation**: Random rotation, flipping, and color jittering
- **Multiple Optimizers**: Support for AdamW and SGD optimizers
- **Loss Functions**: Cross-entropy loss with optional focal loss and ArcFace loss

## 📊 Performance Metrics

The project tracks several performance metrics:

- **mAP (Mean Average Precision)**: Primary evaluation metric
- **Top-1 Accuracy**: Standard classification accuracy
- **Top-5 Accuracy**: Top-5 classification accuracy
- **Training Loss**: Cross-entropy loss during training

## 🔧 Data Format

The project expects data in Parquet format with the following structure:
- **Id**: Image file path
- **Label**: Class label (0-191 for 192 classes)

### Performance Optimization
**Important**: Converting CSV data to Parquet format provides a **3x speed improvement** in training time. The project includes `csv_to_parquet.py` utility to convert your CSV datasets to the optimized Parquet format.

## 📈 Results

The project includes multiple experimental results stored in separate directories:

- **Effb0-Best/**: Best performing EfficientNet-B0 model results
- **EffV2B0/**: EfficientNet-V2-B0 experimental results
- **Resnet AdamW 0002/**: ResNet-50 with AdamW optimizer results
- **ResnetSGD/**: ResNet with SGD optimizer results

## 🛠️ Customization

### Adding New Models
1. Implement your model in `model.py` or create a new file
2. Add training script following the pattern in `train.py`
3. Update configuration in `config.py`

### Modifying Data Pipeline
1. Update `data.py` for custom dataset classes
2. Modify data preprocessing in the dataset `__getitem__` method
3. Update data loading parameters in training scripts

### Custom Loss Functions
1. Add new loss functions in `utils.py`
2. Update training scripts to use the new loss function
3. Modify the loss calculation in the training loop

## 📝 Logging

The project includes comprehensive logging:
- Training progress with real-time metrics
- Loss and accuracy tracking
- Learning rate monitoring
- Training time estimation

Logs are saved in the `logs/` directory with timestamps.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- TIMM library for pre-trained models
- OpenCV for image processing utilities
- The computer vision community for research and implementations

## 📞 Contact

For questions or support, please open an issue in the repository or contact the maintainers.

---

**Note**: This project is designed for research and educational purposes. Please ensure compliance with local laws and regulations when using knife detection systems.