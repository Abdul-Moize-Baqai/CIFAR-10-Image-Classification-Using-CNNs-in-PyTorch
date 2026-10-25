# CIFAR-10 Image Classification Using CNNs in PyTorch

This project trains and compares convolutional neural networks (CNNs) on CIFAR-10 to explore three objectives:

1. Test different convolutional neural network architectures.
2. Improve classification accuracy.
3. Reduce overfitting and underfitting.

The Python source file has **no code comments**. This README provides the explanation separately.

## Requirements

- Python 3.10 or newer (recommended)
- PyTorch and torchvision (compatible versions)
- NumPy
- Matplotlib
- Internet connection on the first run to download CIFAR-10

Create an optional virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build that supports your CUDA setup using the instructions at https://pytorch.org/get-started/locally/ . The script automatically uses CUDA when available and otherwise runs on the CPU.

## Run

From this project directory, execute:

```bash
python cnn_assignment.py
```

**Note:** All six models are trained sequentially, so this may take substantial time, especially on a CPU. No pretrained weights are required. CIFAR-10 is downloaded into the `data/` directory as needed.

## Dataset and evaluation

The code uses the **CIFAR-10** dataset (32 × 32 RGB images, 10 object classes):

| Subset | Images | Purpose |
| --- | ---: | --- |
| Training | 45,000 | Fit model weights |
| Validation | 5,000 | Compare architectures and select the best checkpoint |
| Test | 10,000 | Evaluate the selected model once after model selection |

The training/validation split is generated using seed `42` and reused for every experiment. The baseline models use `ToTensor()` alone. The improved model uses random cropping, horizontal flipping and normalization for training; its validation and test images are normalized without random augmentation.

## Task 1 — Architecture experiments

The script trains the following baseline and modified CNNs for **10 epochs each**:

| Model | Architecture change |
| --- | --- |
| `A_original` | Three convolutional layers, 3 × 3 kernels, filters `(32, 64, 64)`, one dense hidden layer of 64 neurons |
| `B_kernel_5x5` | Replaces the first 3 × 3 kernel with a 5 × 5 kernel |
| `C_four_convs` | Adds a fourth convolutional layer |
| `D_more_filters` | Increases filters to `(64, 128, 128)` |
| `E_more_dense` | Uses two dense hidden layers with 128 and 64 neurons |

All baseline models use ReLU, and max pooling follows the first two convolutions.

## Tasks 2 and 3 — Improved accuracy and generalization

`F_improved` uses a separate CNN architecture and training procedure, including:

- Three padded convolutional layers with 32, 64 and 128 filters.
- Batch normalization and ReLU after each convolution.
- Max pooling after each convolution.
- A dense hidden layer with 128 neurons and dropout (`0.4`).
- Random-crop and horizontal-flip augmentation during training.
- AdamW with learning rate `0.001` and weight decay `0.0001`.
- Learning-rate reduction when validation loss stops improving.
- Early stopping after five consecutive epochs without a new best validation loss.

The improved model trains for **up to 20 epochs**; early stopping may end training sooner. The script restores the model weights from the epoch with the lowest validation loss.

These techniques are intended to improve accuracy and generalization, but the script **does not guarantee** that the improved CNN will outperform all baseline variants. Actual performance must be measured by running the experiments.

## Training settings

| Parameter | Value |
| --- | --- |
| Random seed | `42` |
| Batch size | `32` |
| Architecture experiment epochs | `10` |
| Improved model maximum epochs | `20` |
| Learning rate | `0.001` |
| Baseline optimizer | Adam |
| Improved optimizer | AdamW |
| Loss function | Cross-entropy |
| Model-selection criterion | Lowest validation loss |

To change these settings, edit the constants near the top of `cnn_assignment.py`.

## Outputs

After training, the script writes files to `cnn_task_results/`:

```text
cnn_task_results/
├── A_original.pth
├── B_kernel_5x5.pth
├── C_four_convs.pth
├── D_more_filters.pth
├── E_more_dense.pth
├── F_improved.pth
├── comparison.csv
├── model_comparison.png
├── learning_curves.png
├── test_predictions.png
└── test_result.txt
```

- **`.pth`**: Best checkpoint (model state dictionary) for each experiment, selected by minimum validation loss.
- **`comparison.csv`**: Experiment name, trainable parameter count, best epoch, training accuracy, validation accuracy, validation loss and checkpoint location; rows are ordered by validation loss.
- **`model_comparison.png`**: Bar chart of model validation accuracies.
- **`learning_curves.png`**: Training/validation accuracy and loss for `A_original` versus `F_improved`.
- **`test_predictions.png`**: Predicted and true labels for the first five test images using the selected model.
- **`test_result.txt`**: Final test accuracy and test loss of the selected model.

The selected final model is the one with the **lowest validation loss**, not necessarily the one with the highest validation accuracy.

## How the script is organized

| Function/class | Role |
| --- | --- |
| `set_seed()` | Set random seeds for reproducibility |
| `get_datasets()` | Download CIFAR-10, prepare transforms and split the data |
| `get_loaders()` | Build training and validation data loaders |
| `CNN` | Configurable CNN used in Task 1 |
| `ImprovedCNN` | Architecture for Tasks 2 and 3 |
| `EXPERIMENTS` | Configurations for the five Task 1 models |
| `make_model()` | Instantiate the requested architecture |
| `one_epoch()` | Run one full training or evaluation epoch |
| `train_experiment()` | Train, validate, save the best checkpoint and return scores |
| `save_comparison()` | Rank models, write CSV and create comparison plots |
| `final_test()` | Test the selected model and save example predictions |
| `main()` | Execute the complete training and evaluation pipeline |
