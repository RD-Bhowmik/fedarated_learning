"""
Model architectures for the federated learning project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import IMAGE_SIZE, NUM_CLASSES
from src.utils.helpers import setup_logger, count_parameters

# Set up logger
logger = setup_logger('model', 'model.log')

# PyTorch Models

class SimpleCNN(nn.Module):
    """A simple CNN model for image classification."""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EfficientNetModel(nn.Module):
    """EfficientNet-based model for image classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(EfficientNetModel, self).__init__()
        # Import EfficientNet from torchvision
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        # Load pretrained model
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ResNetModel(nn.Module):
    """ResNet-based model for image classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ResNetModel, self).__init__()
        # Import ResNet from torchvision
        from torchvision.models import resnet50, ResNet50_Weights
        
        # Load pretrained model
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)
        
        # Replace classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# TensorFlow Models

def create_simple_cnn_tf(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """Create a simple CNN model using TensorFlow/Keras."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_efficientnet_tf(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, pretrained=True):
    """Create an EfficientNet model using TensorFlow/Keras."""
    # Load pretrained EfficientNetB0
    if pretrained:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

def create_resnet_tf(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, pretrained=True):
    """Create a ResNet model using TensorFlow/Keras."""
    # Load pretrained ResNet50
    if pretrained:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

def get_pytorch_model(model_name='efficientnet', num_classes=NUM_CLASSES, pretrained=True):
    """
    Get a PyTorch model by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'efficientnet', or 'resnet')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == 'efficientnet':
        model = EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet':
        model = ResNetModel(num_classes=num_classes, pretrained=pretrained)
    else:
        logger.warning(f"Unknown model name: {model_name}. Using EfficientNet.")
        model = EfficientNetModel(num_classes=num_classes, pretrained=pretrained)
    
    logger.info(f"Created PyTorch {model_name} model with {count_parameters(model)} trainable parameters")
    return model

def get_tensorflow_model(model_name='efficientnet', num_classes=NUM_CLASSES, pretrained=True):
    """
    Get a TensorFlow model by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'efficientnet', or 'resnet')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        TensorFlow model
    """
    if model_name == 'simple_cnn':
        model = create_simple_cnn_tf(num_classes=num_classes)
    elif model_name == 'efficientnet':
        model = create_efficientnet_tf(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet':
        model = create_resnet_tf(num_classes=num_classes, pretrained=pretrained)
    else:
        logger.warning(f"Unknown model name: {model_name}. Using EfficientNet.")
        model = create_efficientnet_tf(num_classes=num_classes, pretrained=pretrained)
    
    logger.info(f"Created TensorFlow {model_name} model with {model.count_params()} parameters")
    return model

def main():
    """Test the model architectures."""
    # Test PyTorch models
    logger.info("Testing PyTorch models...")
    for model_name in ['simple_cnn', 'efficientnet', 'resnet']:
        model = get_pytorch_model(model_name)
        # Test forward pass
        x = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        y = model(x)
        logger.info(f"PyTorch {model_name} output shape: {y.shape}")
    
    # Test TensorFlow models
    logger.info("Testing TensorFlow models...")
    for model_name in ['simple_cnn', 'efficientnet', 'resnet']:
        model = get_tensorflow_model(model_name)
        # Test forward pass
        x = tf.random.normal((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        y = model(x)
        logger.info(f"TensorFlow {model_name} output shape: {y.shape}")
    
    logger.info("Model testing completed successfully!")

if __name__ == "__main__":
    main() 