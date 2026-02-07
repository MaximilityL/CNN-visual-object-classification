# CNN Visual Object Classification

A comprehensive implementation of a Convolutional Neural Network for visual object classification using the Stanford Cars dataset. This project follows the exact architecture specification provided and includes complete training, evaluation, hyperparameter analysis, and visualization pipelines.

## Features

- **Compact CNN Architecture**: Implements the specified architecture with configurable convolutional blocks
- **GPU Support**: Full support for AMD Radeon GPUs via CUDA/ROCm
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **Hyperparameter Analysis**: Automated grid search with detailed analysis of hyperparameter effects
- **Rich Visualizations**: Training curves, confusion matrices, feature maps, and sample predictions
- **Configuration-Driven**: All parameters controlled via YAML configuration file
- **Modular Design**: Clean, modular codebase for easy extension and maintenance

## Architecture

The model follows the specified Compact CNN architecture:

```
Input → [ConvBlock] → [ConvBlock] → ... → Flatten → FC → Softmax
```

Where each ConvBlock consists of:
- Conv3x3 (with padding)
- ReLU activation
- Optional MaxPool2x2

## Project Structure

```
├── config.yaml              # Configuration file
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── dataset/                # Dataset directory (you need to create this)
│   ├── train/
│   │   ├── class_001/
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── class_001/
│       │   ├── 1.jpg
│       │   └── ...
│       └── ...
├── src/
│   ├── config_manager.py   # Configuration management
│   ├── model.py            # CNN model implementation
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── trainer.py          # Training loop and device management
│   ├── evaluation.py       # Model evaluation and metrics
│   ├── visualization.py    # Plotting and visualization
│   └── hyperparameter_analysis.py  # Hyperparameter optimization
├── models/                 # Saved models
├── results/               # Results and visualizations
│   ├── plots/
│   ├── metrics.json
│   └── hyperparameter_analysis/
└── logs/                  # Training logs
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd windsurf-project-3
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **For AMD Radeon GPU Support**:
   - Install PyTorch with ROCm support: [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)
   - Ensure ROCm is properly installed on your system

4. **Dataset Setup**:
   - Download the Stanford Cars dataset or create your own dataset
   - Organize it in the specified directory structure under `dataset/`
   - Update `config.yaml` with the correct paths and number of classes

## Configuration

All parameters are controlled through `config.yaml`. Key sections:

- **Dataset**: Data paths, image size, batch size
- **Model**: Architecture parameters (conv blocks, filters, etc.)
- **Training**: Learning rate, optimizer, epochs, scheduler
- **Device**: GPU configuration for AMD Radeon support
- **Evaluation**: Metrics and saving options
- **Visualization**: Plot settings and options
- **Hyperparameter Analysis**: Parameter ranges for optimization

## Usage

### 1. Train the Model

```bash
python main.py --mode train --config config.yaml
```

### 2. Evaluate a Trained Model

```bash
python main.py --mode evaluate --model_path models/best_model.pth
```

### 3. Run Hyperparameter Analysis

```bash
python main.py --mode hyperparameter_search --max_experiments 10
```

### 4. Full Pipeline (Train → Evaluate → Visualize)

```bash
python main.py --mode full_pipeline
```

## Configuration Examples

### Basic Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  batch_size: 32

model:
  num_conv_blocks: 2
  conv_filters: [64, 128]
  kernel_size: 3
  dropout_rate: 0.5

device:
  use_gpu: true
  gpu_type: "amd"
```

### Hyperparameter Analysis Configuration

```yaml
hyperparameter_analysis:
  enabled: true
  learning_rates: [0.001, 0.0001, 0.01]
  batch_sizes: [16, 32, 64]
  num_conv_blocks: [2, 3, 4]
  filter_sizes: [[32, 64], [64, 128], [128, 256]]
```

## Results and Outputs

### Training Outputs
- **Model Checkpoints**: Saved in `models/` directory
- **Training Logs**: Detailed logs in `logs/training.log`
- **Training History**: Loss and accuracy curves

### Evaluation Outputs
- **Metrics**: Comprehensive metrics saved in `results/metrics.json`
- **Confusion Matrix**: Visual representation of classification performance
- **Class Performance**: Per-class precision, recall, and F1-scores

### Visualizations
- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrix**: Heatmap of prediction vs true labels
- **Sample Predictions**: Visual examples with predictions
- **Feature Maps**: Visualization of learned features
- **Hyperparameter Analysis**: Effects of different parameters

### Hyperparameter Analysis
- **Experiment Results**: All experiment configurations and results
- **Analysis Report**: Comprehensive analysis of hyperparameter effects
- **Best Configurations**: Optimal parameter settings

## GPU Support (AMD Radeon)

The project includes comprehensive GPU support with automatic detection:

1. **Automatic Detection**: Automatically detects CUDA/ROCm availability
2. **AMD GPU Recognition**: Identifies AMD Radeon GPUs specifically
3. **Fallback Support**: Graceful fallback to CPU if GPU unavailable
4. **Memory Management**: Proper GPU memory handling

## Key Features

### 1. Compact CNN Implementation
- Follows the exact specified architecture
- Configurable number of convolutional blocks
- Proper weight initialization (He initialization)
- Global average pooling for robustness

### 2. Comprehensive Evaluation
- Multiple metrics (accuracy, precision, recall, F1-score)
- Top-K accuracy evaluation
- Per-class performance analysis
- Misclassification analysis

### 3. Advanced Hyperparameter Analysis
- Grid search over multiple parameters
- Statistical analysis of parameter effects
- Best configuration identification
- Comprehensive reporting

### 4. Rich Visualizations
- Training progress visualization
- Confusion matrix heatmaps
- Feature map visualization
- Sample prediction displays

## Performance Optimization

- **Data Loading**: Multi-threaded data loading with prefetching
- **Memory Management**: Efficient GPU memory usage
- **Batch Processing**: Optimized batch processing
- **Mixed Precision**: Support for mixed precision training (configurable)

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   - Ensure ROCm/CUDA is properly installed
   - Check PyTorch installation with GPU support
   - Verify GPU drivers are up to date

2. **Dataset Loading Issues**:
   - Verify dataset directory structure
   - Check image file formats
   - Ensure correct permissions

3. **Memory Issues**:
   - Reduce batch size in configuration
   - Enable gradient checkpointing if needed
   - Use mixed precision training

### Logging and Debugging

- All operations are logged with detailed information
- Check `logs/training.log` for detailed training information
- Use the evaluation mode for detailed performance analysis

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{cnn_stanford_cars,
  title={CNN Visual Object Classification on Stanford Cars Dataset},
  author={Your Name},
  year={2024},
  description={Implementation of Compact CNN for car model classification with comprehensive evaluation and hyperparameter analysis}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Stanford Cars Dataset providers
- PyTorch team for the deep learning framework
- The computer vision community for inspiration and best practices
