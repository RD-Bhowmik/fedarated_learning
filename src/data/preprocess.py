"""
Data preprocessing module for the cervical cancer dataset.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import sys
import shutil

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    TRAIN_DIR, TEST_DIR, ADDITIONAL_TYPE1_DIR, ADDITIONAL_TYPE2_DIR, 
    ADDITIONAL_TYPE3_DIR, IMAGE_SIZE, VALIDATION_SPLIT, DATA_DIR
)
from src.utils.helpers import setup_logger, set_seed
# Import visualization module
from src.visualization.visualize import (
    visualize_class_distribution,
    visualize_sample_images,
    visualize_augmentations,
    visualize_preprocessing_steps
)

# Set up logger
logger = setup_logger('preprocess', 'preprocess.log')

def load_image(image_path, target_size=IMAGE_SIZE):
    """Load and preprocess an image."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        return img
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def get_augmentation_pipeline(augmentation_level='standard'):
    """
    Create an augmentation pipeline based on the specified level.
    
    Args:
        augmentation_level: 'minimal', 'standard', or 'advanced'
        
    Returns:
        Albumentations transformation pipeline
    """
    if augmentation_level == 'minimal':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    elif augmentation_level == 'standard':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    elif augmentation_level == 'advanced':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    else:
        logger.warning(f"Unknown augmentation level: {augmentation_level}. Using standard.")
        return get_augmentation_pipeline('standard')

def load_dataset(include_additional=True, include_test=True):
    """
    Load the dataset.
    
    Args:
        include_additional: Whether to include additional data
        include_test: Whether to include test data
        
    Returns:
        X: List of image paths
        y: List of labels
    """
    logger.info("Loading dataset...")
    
    # Initialize lists to store image paths and labels
    X = []
    y = []
    
    # Load fixed labels for correction
    fixed_labels_path = os.path.join(DATA_DIR, "fixed_labels_v2.csv")
    fixed_labels = {}
    if os.path.exists(fixed_labels_path):
        logger.info("Loading fixed labels for correction...")
        fixed_labels_df = pd.read_csv(fixed_labels_path)
        for _, row in fixed_labels_df.iterrows():
            fixed_labels[row['filename']] = row['new_label']
        logger.info(f"Loaded {len(fixed_labels)} fixed labels")
    
    # Helper function to get the correct label
    def get_label(filename, default_label):
        if filename in fixed_labels:
            corrected_label = fixed_labels[filename]
            if corrected_label == "Type_1":
                return 0
            elif corrected_label == "Type_2":
                return 1
            elif corrected_label == "Type_3":
                return 2
        
        # If not in fixed labels or label not recognized, use default
        return default_label
    
    # Load Type 1 images from train directory
    logger.info("Loading Type 1 images from train directory...")
    type1_dir = Path(TRAIN_DIR) / "Type_1"
    for img_path in tqdm(list(type1_dir.glob("*.jpg"))):
        filename = img_path.name
        label = get_label(filename, 0)  # Default Type_1 label
        X.append(str(img_path))
        y.append(label)
    
    # Load Type 2 images from train directory
    logger.info("Loading Type 2 images from train directory...")
    type2_dir = Path(TRAIN_DIR) / "Type_2"
    for img_path in tqdm(list(type2_dir.glob("*.jpg"))):
        filename = img_path.name
        label = get_label(filename, 1)  # Default Type_2 label
        X.append(str(img_path))
        y.append(label)
    
    # Load Type 3 images from train directory
    logger.info("Loading Type 3 images from train directory...")
    type3_dir = Path(TRAIN_DIR) / "Type_3"
    for img_path in tqdm(list(type3_dir.glob("*.jpg"))):
        filename = img_path.name
        label = get_label(filename, 2)  # Default Type_3 label
        X.append(str(img_path))
        y.append(label)
    
    # Load additional data if requested
    if include_additional:
        # Load additional Type 1 images
        logger.info("Loading additional Type 1 images...")
        add_type1_dir = Path(ADDITIONAL_TYPE1_DIR) / "Type_1"
        for img_path in tqdm(list(add_type1_dir.glob("*.jpg"))):
            filename = img_path.name
            label = get_label(filename, 0)  # Default Type_1 label
            X.append(str(img_path))
            y.append(label)
        
        # Load additional Type 2 images
        logger.info("Loading additional Type 2 images...")
        add_type2_dir = Path(ADDITIONAL_TYPE2_DIR) / "Type_2"
        for img_path in tqdm(list(add_type2_dir.glob("*.jpg"))):
            filename = img_path.name
            label = get_label(filename, 1)  # Default Type_2 label
            X.append(str(img_path))
            y.append(label)
        
        # Load additional Type 3 images
        logger.info("Loading additional Type 3 images...")
        add_type3_dir = Path(ADDITIONAL_TYPE3_DIR) / "Type_3"
        for img_path in tqdm(list(add_type3_dir.glob("*.jpg"))):
            filename = img_path.name
            label = get_label(filename, 2)  # Default Type_3 label
            X.append(str(img_path))
            y.append(label)
    
    # Load test data if requested
    if include_test:
        logger.info("Loading test images...")
        test_dir = Path(TEST_DIR) / "test"
        
        # Load solution file if available for test labels
        solution_path = os.path.join(DATA_DIR, "solution_stg1_release.csv")
        test_labels = {}
        if os.path.exists(solution_path):
            logger.info("Loading test labels from solution file...")
            solution_df = pd.read_csv(solution_path)
            for _, row in solution_df.iterrows():
                if 'image_name' in solution_df.columns and 'label' in solution_df.columns:
                    filename = row['image_name']
                    label_str = row['label']
                    if label_str == "Type_1":
                        test_labels[filename] = 0
                    elif label_str == "Type_2":
                        test_labels[filename] = 1
                    elif label_str == "Type_3":
                        test_labels[filename] = 2
            logger.info(f"Loaded {len(test_labels)} test labels")
        
        # Process test images
        for img_path in tqdm(list(test_dir.glob("*.jpg"))):
            filename = img_path.name
            # Try to get label from solution file, fixed labels, or skip if unknown
            if filename in test_labels:
                label = test_labels[filename]
                X.append(str(img_path))
                y.append(label)
            elif filename in fixed_labels:
                label = get_label(filename, None)
                if label is not None:
                    X.append(str(img_path))
                    y.append(label)
            else:
                # Skip images with unknown labels
                logger.warning(f"Skipping test image with unknown label: {filename}")
    
    logger.info(f"Loaded {len(X)} images")
    logger.info(f"Class distribution: Type_1: {y.count(0)}, Type_2: {y.count(1)}, Type_3: {y.count(2)}")
    
    # Visualize class distribution
    visualize_class_distribution(y, ['Type_1', 'Type_2', 'Type_3'], title="Class Distribution (Original Dataset)")
    
    return X, y

def create_train_val_split(X, y, val_split=VALIDATION_SPLIT, random_state=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        X: List of image paths
        y: List of labels
        val_split: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(X, y, test_size=val_split, random_state=random_state, stratify=y)

def preprocess_dataset(X, y, output_dir, augmentation_level='advanced', preload=False):
    """
    Preprocess the dataset.
    
    Args:
        X: List of image paths
        y: List of labels
        output_dir: Output directory
        augmentation_level: Augmentation level
        preload: Whether to preload images
        
    Returns:
        Metadata file paths
    """
    logger.info("Preprocessing dataset...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = create_train_val_split(X, y)
    
    logger.info(f"Train set: {len(X_train)} images")
    logger.info(f"Validation set: {len(X_val)} images")
    
    # Get augmentation pipeline
    transform = get_augmentation_pipeline(augmentation_level)
    
    # Preload images if requested
    if preload:
        logger.info("Preloading images...")
        X_train_processed = []
        y_train_processed = []
        
        for i, (img_path, label) in enumerate(tqdm(zip(X_train, y_train), total=len(X_train))):
            img = load_image(img_path)
            if img is not None:
                # Apply augmentation
                augmented = transform(image=img)['image']
                X_train_processed.append(augmented)
                y_train_processed.append(label)
        
        X_val_processed = []
        y_val_processed = []
        
        for i, (img_path, label) in enumerate(tqdm(zip(X_val, y_val), total=len(X_val))):
            img = load_image(img_path)
            if img is not None:
                # No augmentation for validation set
                X_val_processed.append(img)
                y_val_processed.append(label)
        
        # Save processed data
        logger.info("Saving processed data...")
        np.save(output_dir / "X_train.npy", np.array(X_train_processed))
        np.save(output_dir / "y_train.npy", np.array(y_train_processed))
        np.save(output_dir / "X_val.npy", np.array(X_val_processed))
        np.save(output_dir / "y_val.npy", np.array(y_val_processed))
        
        # Visualize sample images
        sample_indices = np.random.choice(len(X_train_processed), min(10, len(X_train_processed)), replace=False)
        sample_images = [X_train_processed[i] for i in sample_indices]
        sample_labels = [y_train_processed[i] for i in sample_indices]
        visualize_sample_images(sample_images, sample_labels, ['Type_1', 'Type_2', 'Type_3'], num_samples=5)
    
    # Create metadata files
    logger.info("Creating metadata files...")
    train_df = pd.DataFrame({
        'image_path': X_train,
        'label': y_train
    })
    
    val_df = pd.DataFrame({
        'image_path': X_val,
        'label': y_val
    })
    
    train_metadata_path = output_dir / "train_metadata.csv"
    val_metadata_path = output_dir / "val_metadata.csv"
    
    train_df.to_csv(train_metadata_path, index=False)
    val_df.to_csv(val_metadata_path, index=False)
    
    # Visualize augmentations
    if augmentation_level != 'none':
        # Load a sample image
        sample_image = load_image(X_train[0])
        if sample_image is not None:
            # Create augmentation function
            def augment_image(image):
                augmented = transform(image=image)['image']
                return augmented
            
            # Visualize augmentations
            visualize_augmentations(sample_image, augment_image, num_augmentations=5)
    
    # Visualize preprocessing steps
    sample_image_path = X_train[0]
    visualize_preprocessing_steps(sample_image_path)
    
    logger.info("Preprocessing completed successfully!")
    return train_metadata_path, val_metadata_path

def main():
    """Main function to preprocess the dataset."""
    set_seed(42)
    
    logger.info("Loading dataset...")
    X, y = load_dataset(include_additional=True, include_test=True)
    logger.info(f"Loaded {len(X)} images with class distribution: {np.bincount(y)}")
    
    logger.info("Splitting into train and validation sets...")
    X_train, X_val, y_train, y_val = create_train_val_split(X, y)
    logger.info(f"Train set: {len(X_train)} images, Validation set: {len(X_val)} images")
    
    # Create output directories
    output_dir = Path(__file__).resolve().parent.parent.parent / "processed_data"
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process training set with standard augmentation
    logger.info("Processing training set...")
    train_metadata_path, _ = preprocess_dataset(
        X_train, y_train, train_dir, augmentation_level='standard', preload=False
    )
    
    # Process validation set with minimal augmentation
    logger.info("Processing validation set...")
    _, val_metadata_path = preprocess_dataset(
        X_val, y_val, val_dir, augmentation_level='minimal', preload=False
    )
    
    # Copy metadata files to the base directory for federated learning
    logger.info("Copying metadata files to the base directory for federated learning...")
    base_train_metadata = output_dir / "train_metadata.csv"
    base_val_metadata = output_dir / "val_metadata.csv"
    shutil.copy(train_metadata_path, base_train_metadata)
    shutil.copy(val_metadata_path, base_val_metadata)
    
    logger.info(f"Preprocessing completed. Train metadata saved to {train_metadata_path} and {base_train_metadata}, validation metadata saved to {val_metadata_path} and {base_val_metadata}")

if __name__ == "__main__":
    main() 