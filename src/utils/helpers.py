"""
Helper functions for the federated learning project.
"""

import os
import random
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging

def set_seed(seed=42):
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger that writes to both file and console."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """Print classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    return report

def get_available_device():
    """Get the available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_client_partitions(data, num_clients, iid=False):
    """
    Partition data among clients.
    
    Args:
        data: List of data samples
        num_clients: Number of clients
        iid: If True, partition data in an IID manner; otherwise, non-IID
        
    Returns:
        Dictionary mapping client_id to data indices
    """
    if iid:
        # IID partitioning
        indices = list(range(len(data)))
        random.shuffle(indices)
        partitions = {}
        
        # Divide indices equally among clients
        partition_size = len(indices) // num_clients
        for i in range(num_clients):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < num_clients - 1 else len(indices)
            partitions[i] = indices[start_idx:end_idx]
    else:
        # Non-IID partitioning (skewed class distribution)
        # Assuming data has labels and we can sort by label
        partitions = {i: [] for i in range(num_clients)}
        
        # Group data by label
        label_indices = {}
        for idx, (_, label) in enumerate(data):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
        
        # Distribute each label's data among clients with skew
        for label, indices in label_indices.items():
            # Determine primary client for this label (gets more samples)
            primary_client = label % num_clients
            
            # Give primary client more samples
            primary_share = 0.6  # Primary client gets 60% of this label's data
            primary_count = int(len(indices) * primary_share)
            
            # Shuffle indices
            random.shuffle(indices)
            
            # Assign to primary client
            partitions[primary_client].extend(indices[:primary_count])
            
            # Distribute remaining indices among other clients
            remaining = indices[primary_count:]
            secondary_clients = [i for i in range(num_clients) if i != primary_client]
            
            for i, idx in enumerate(remaining):
                client = secondary_clients[i % len(secondary_clients)]
                partitions[client].append(idx)
    
    return partitions 