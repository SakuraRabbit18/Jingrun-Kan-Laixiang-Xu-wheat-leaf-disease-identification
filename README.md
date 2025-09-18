# LPNet Image Classification Model Implementation and Training

## Project Overview

This project implements an improved model LPNet based on MobileNetV3 for image classification tasks. It includes complete functionalities such as model definition, data processing, training 流程 and performance evaluation, supporting training and testing on custom datasets.

## Environment Dependencies

- Python 3.9
- PyTorch == 2.3.1
- torchvision == 0.18.1

## Model Architecture

LPNet is improved based on the MobileNetV3 Small architecture, with the following main features:

1. Uses Inverted Residual structure as the basic building block
2. Introduces LGCA (Local Group Channel Attention) module to enhance feature extraction capability
3. Replaces traditional pooling layers with ConvInceptionPool to improve feature aggregation 效果
4. Maintains lightweight characteristics, suitable for deployment in resource-constrained environments

The model definition is located in the LPNet.py file, which mainly includes:

- `InvertedResidualConfig`: Controls the configuration of inverted residual blocks
- `LGCA`: Local Group Channel Attention module
- `InvertedResidual`: Implementation of inverted residual blocks
- `ConvInceptionPool`: Convolutional inception pooling layer
- `MobileNetV3`: Main model class
- `get_LPNet`: Function to obtain the LPNet model

## Dataset Preparation

1. The dataset should be organized according to the following directory structure:

```plaintext
split_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

1. `train`, `val`, and `test` correspond to the training set, validation set, and test set respectively, with one subfolder per category.

## Data Preprocessing

In this project, the data preprocessing 流程 is mainly implemented through PyTorch's `transforms` module, including operations such as image loading, augmentation, and normalization. The preprocessing logic is located in the `ExperimentRunner` class in the train.py file, with different strategies adopted for the training set and validation/test set.

### Preprocessing 流程

#### 1. Training Set Preprocessing (`train_transform`)

Data augmentation techniques are applied to the training set to improve the generalization ability of the model:

```python
self.train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert to Tensor format (0-255 -> 0.0-1.0)
    transforms.Normalize(  # Normalization
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

#### 2. Validation/Test Set Preprocessing (`val_transform`)

A simpler preprocessing is used for the validation and test sets without data augmentation:

```python
self.val_transform = transforms.Compose([
    transforms.Resize(int(224/0.875)),  # Resize proportionally
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert to Tensor format
    transforms.Normalize(  # Use the same normalization parameters as the training set
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

### Explanation of Preprocessing Details

1. **Size Adjustment**:
   - The training set uses `RandomResizedCrop` for random cropping to increase sample diversity
   - The validation/test set is first resized to a larger size (256x256) proportionally, then center-cropped to 224x224 to ensure complete image information
2. **Data Augmentation**:
   - Only `RandomHorizontalFlip` (50% probability of horizontal flip) is applied to the training set
   - Augmentation operations help prevent model overfitting and improve performance on new data
3. **Normalization**:
   - Uses the mean and standard deviation of the ImageNet dataset for normalization
   - Mean: `[0.485, 0.456, 0.406]` (corresponding to RGB three channels)
   - Standard deviation: `[0.229, 0.224, 0.225]`
   - Normalization stabilizes the distribution of input data and accelerates model convergence
4. **Data Loading**:
   - The custom `CustomDataset` class is responsible for loading images and applying preprocessing
   - Supports reading data from folder structures organized by category
   - Automatically builds category index mappings (`class_to_idx` and `idx_to_class`)

## Training the Model

Run the training script:

```bash
python train.py
```

The main parameters during training can be adjusted when initializing the `ExperimentRunner` class:

- `num_classes`: Number of categories (automatically inferred from the training set folders)
- `batch_size`: Batch size, default 64
- `num_epochs`: Number of training epochs, default 200

During training, the following will be done automatically:

- Save the best model (based on validation set accuracy)
- Record training/validation loss and accuracy
- Evaluate final performance on the test set
- Calculate metrics for each category (accuracy, precision, recall, etc.)
- Measure model computational complexity (FLOPs, parameter count) and inference speed (FPS)

## Result Output

Training results will be saved in the `results` folder:

- `{model_name}_training.csv`: Records of loss and accuracy during training
- `summary_FullTest.csv`: Summary of model performance
- `class_metrics/{model_name}_class_metrics.csv`: Detailed metrics for each category

## Model Evaluation Metrics

Model evaluation includes the following metrics:

- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1-score
- Number of parameters
- Computational complexity (FLOPs, MACs)
- Inference speed (FPS on GPU/CPU)

## Customization and Extension

1. To modify the model structure, adjust the network definition in LPNet.py
2. To change the data augmentation strategy, modify `train_transform` and `val_transform` in `ExperimentRunner`
3. To adjust training parameters (learning rate, optimizer, etc.), modify the relevant settings in the `run_experiment` method

## Notes

1. When running for the first time, if using pre-trained weights, ensure that the `mobilenet_v3_small-047dcff4.pth` file is in the project root directory
2. The `results` folder and subdirectories will be created automatically during training
3. More models can be added for comparative experiments by adjusting the `models_to_test` list