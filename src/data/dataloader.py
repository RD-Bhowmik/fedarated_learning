"""
Data loader module for the federated learning project.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as transforms

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import BATCH_SIZE, NUM_CLASSES
from src.utils.helpers import setup_logger

# Set up logger
logger = setup_logger('dataloader', 'dataloader.log')

class CervicalCancerDataset(Dataset):
    """PyTorch Dataset for the cervical cancer dataset."""
    
    def __init__(self, metadata_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            metadata_file: Path to the metadata CSV file
            transform: Optional transform to apply to the images
        """
        self.metadata = pd.read_csv(metadata_file)
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.metadata.iloc[idx]['image_path']
        label = self.metadata.iloc[idx]['label']
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Apply transform
            img = self.transform(img)
            
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder image and the label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, label

class TFCervicalCancerDataset:
    """TensorFlow Dataset for the cervical cancer dataset."""
    
    @staticmethod
    def load_dataset(metadata_file, batch_size=BATCH_SIZE, shuffle=True):
        """
        Load the dataset as a TensorFlow dataset.
        
        Args:
            metadata_file: Path to the metadata CSV file
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            
        Returns:
            TensorFlow dataset
        """
        metadata = pd.read_csv(metadata_file)
        
        def generator():
            for _, row in metadata.iterrows():
                img_path = row['image_path']
                label = row['label']
                
                try:
                    # Load image using TensorFlow
                    img = tf.io.read_file(img_path)
                    img = tf.image.decode_jpeg(img, channels=3)
                    
                    # Resize and normalize
                    img = tf.image.resize(img, [224, 224])
                    img = tf.cast(img, tf.float32) / 255.0
                    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                except tf.errors.NotFoundError:
                    print(f"Error loading image {img_path}")
                    # Return a placeholder image
                    img = tf.zeros([224, 224, 3], dtype=tf.float32)
                
                yield img, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            )
        )
        
        # One-hot encode labels
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))
        
        # Shuffle and batch
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def get_pytorch_dataloaders(train_metadata, val_metadata, batch_size=BATCH_SIZE, num_workers=4):
    """
    Get PyTorch DataLoaders for training and validation.
    
    Args:
        train_metadata: Path to the training metadata CSV file
        val_metadata: Path to the validation metadata CSV file
        batch_size: Batch size
        num_workers: Number of worker threads for data loading
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = CervicalCancerDataset(train_metadata)
    val_dataset = CervicalCancerDataset(val_metadata)
    
    # Create data loaders
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

def get_tensorflow_datasets(train_metadata, val_metadata, batch_size=BATCH_SIZE):
    """
    Get TensorFlow datasets for training and validation.
    
    Args:
        train_metadata: Path to the training metadata CSV file
        val_metadata: Path to the validation metadata CSV file
        batch_size: Batch size
        
    Returns:
        train_dataset, val_dataset
    """
    train_dataset = TFCervicalCancerDataset.load_dataset(
        train_metadata, batch_size=batch_size, shuffle=True
    )
    
    val_dataset = TFCervicalCancerDataset.load_dataset(
        val_metadata, batch_size=batch_size, shuffle=False
    )
    
    return train_dataset, val_dataset

def create_client_datasets(metadata_file, num_clients, iid=False):
    """
    Create datasets for federated learning clients.
    
    Args:
        metadata_file: Path to the metadata CSV file
        num_clients: Number of clients
        iid: Whether to use IID partitioning
        
    Returns:
        List of client datasets
    """
    metadata = pd.read_csv(metadata_file)
    
    if iid:
        # IID partitioning: randomly shuffle and split
        metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        client_dfs = np.array_split(metadata, num_clients)
    else:
        # Non-IID partitioning: skew the class distribution
        client_dfs = [pd.DataFrame(columns=metadata.columns) for _ in range(num_clients)]
        
        # Group by label
        for label, group in metadata.groupby('label'):
            # Shuffle the group
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Determine primary client for this label
            primary_client = label % num_clients
            
            # Give primary client more samples
            primary_share = 0.6  # Primary client gets 60% of this label's data
            primary_count = int(len(group) * primary_share)
            
            # Assign to primary client
            client_dfs[primary_client] = pd.concat([client_dfs[primary_client], group.iloc[:primary_count]])
            
            # Distribute remaining samples among other clients
            remaining = group.iloc[primary_count:]
            secondary_clients = [i for i in range(num_clients) if i != primary_client]
            
            # Split remaining samples among secondary clients
            secondary_dfs = np.array_split(remaining, len(secondary_clients))
            for i, client_idx in enumerate(secondary_clients):
                client_dfs[client_idx] = pd.concat([client_dfs[client_idx], secondary_dfs[i]])
    
    # Save client metadata files
    client_metadata_files = []
    for i, df in enumerate(client_dfs):
        output_dir = Path(metadata_file).parent / "clients"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = output_dir / f"client_{i}_metadata.csv"
        df.to_csv(output_file, index=False)
        client_metadata_files.append(str(output_file))
        
        logger.info(f"Client {i} dataset: {len(df)} samples, class distribution: {df['label'].value_counts().to_dict()}")
    
    return client_metadata_files

def main():
    """Test the data loading functionality."""
    # Paths to metadata files
    base_dir = Path(__file__).resolve().parent.parent.parent / "processed_data"
    train_metadata = base_dir / "train_metadata.csv"
    val_metadata = base_dir / "val_metadata.csv"
    
    if not train_metadata.exists() or not val_metadata.exists():
        logger.error("Metadata files not found. Please run preprocess.py first.")
        return
    
    # Create client datasets
    logger.info("Creating client datasets...")
    client_metadata_files = create_client_datasets(train_metadata, num_clients=3, iid=False)
    
    # Test PyTorch data loading
    logger.info("Testing PyTorch data loading...")
    train_loader, val_loader = get_pytorch_dataloaders(train_metadata, val_metadata)
    
    # Print some statistics
    logger.info(f"PyTorch train loader: {len(train_loader)} batches")
    logger.info(f"PyTorch val loader: {len(val_loader)} batches")
    
    # Test TensorFlow data loading
    logger.info("Testing TensorFlow data loading...")
    train_dataset, val_dataset = get_tensorflow_datasets(train_metadata, val_metadata)
    
    # Print some statistics
    for x, y in train_dataset.take(1):
        logger.info(f"TensorFlow batch shape: {x.shape}, labels shape: {y.shape}")
    
    logger.info("Data loading test completed successfully!")

if __name__ == "__main__":
    main() 