"""
Visualization module for generating and saving visualizations for the research paper.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import cv2
from tqdm import tqdm
import torch
import tensorflow as tf
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import IMAGE_SIZE, NUM_CLASSES
from src.utils.helpers import setup_logger

# Set up logger
logger = setup_logger('visualization', 'visualization.log')

# Create visualization directory
VISUALIZATION_DIR = Path(__file__).resolve().parent.parent.parent / "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Create subdirectories for different types of visualizations
DATA_VIZ_DIR = VISUALIZATION_DIR / "data"
TRAINING_VIZ_DIR = VISUALIZATION_DIR / "training"
EVALUATION_VIZ_DIR = VISUALIZATION_DIR / "evaluation"
FEDERATED_VIZ_DIR = VISUALIZATION_DIR / "federated"
INTERPRETABILITY_VIZ_DIR = VISUALIZATION_DIR / "interpretability"

# Create all directories
for dir_path in [DATA_VIZ_DIR, TRAINING_VIZ_DIR, EVALUATION_VIZ_DIR, FEDERATED_VIZ_DIR, INTERPRETABILITY_VIZ_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def save_figure(fig, filename, dir_path=VISUALIZATION_DIR, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib figure to the specified directory.
    
    Args:
        fig: Matplotlib figure
        filename: Filename (without extension)
        dir_path: Directory to save the figure
        dpi: DPI for the saved figure
        bbox_inches: Bounding box inches
    """
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.png"
    
    # Save the figure
    fig.savefig(os.path.join(dir_path, full_filename), dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Saved figure to {os.path.join(dir_path, full_filename)}")
    
    return os.path.join(dir_path, full_filename)

def visualize_class_distribution(labels, class_names, title="Class Distribution", save=True):
    """
    Visualize the distribution of classes in the dataset.
    
    Args:
        labels: List or array of labels
        class_names: List of class names
        title: Title for the plot
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Count the occurrences of each class
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Class': [class_names[i] for i in unique],
        'Count': counts
    })
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Class', y='Count', data=df)
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count + 50, str(count), ha='center')
    
    plt.title(title)
    plt.ylabel('Number of Images')
    plt.xlabel('Class')
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), "class_distribution", DATA_VIZ_DIR)
    
    plt.show()
    return None

def visualize_sample_images(images, labels, class_names, num_samples=5, save=True):
    """
    Visualize sample images from each class.
    
    Args:
        images: List or array of images
        labels: List or array of labels
        class_names: List of class names
        num_samples: Number of samples to show for each class
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Convert to numpy arrays if not already
    images = np.array(images)
    labels = np.array(labels)
    
    # Create a figure
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(15, 10))
    
    # Plot samples for each class
    for i, class_idx in enumerate(range(len(class_names))):
        # Get indices of images for this class
        indices = np.where(labels == class_idx)[0]
        
        # Select random samples
        if len(indices) >= num_samples:
            sample_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            sample_indices = indices
        
        # Plot each sample
        for j, idx in enumerate(sample_indices):
            if j < num_samples:  # Ensure we don't exceed the number of columns
                ax = axes[i, j]
                ax.imshow(images[idx])
                ax.axis('off')
                
                # Add class name for the first column
                if j == 0:
                    ax.set_title(class_names[class_idx])
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        return save_figure(fig, "sample_images", DATA_VIZ_DIR)
    
    plt.show()
    return None

def visualize_augmentations(image, augmentation_fn, num_augmentations=5, save=True):
    """
    Visualize different augmentations of a single image.
    
    Args:
        image: Original image
        augmentation_fn: Augmentation function that takes an image and returns an augmented image
        num_augmentations: Number of augmentations to show
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Create a figure
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 5))
    
    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Plot augmentations
    for i in range(num_augmentations):
        # Apply augmentation
        augmented = augmentation_fn(image)
        
        # Plot augmented image
        axes[i + 1].imshow(augmented)
        axes[i + 1].set_title(f'Augmentation {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        return save_figure(fig, "augmentations", DATA_VIZ_DIR)
    
    plt.show()
    return None

def visualize_preprocessing_steps(image_path, save=True):
    """
    Visualize the preprocessing steps for an image.
    
    Args:
        image_path: Path to the image
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Load the image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Preprocessing steps
    resized = cv2.resize(original, IMAGE_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    
    # Create a figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Plot resized image
    axes[1].imshow(resized)
    axes[1].set_title(f'Resized to {IMAGE_SIZE}')
    axes[1].axis('off')
    
    # Plot normalized image
    axes[2].imshow(normalized)
    axes[2].set_title('Normalized')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        return save_figure(fig, "preprocessing_steps", DATA_VIZ_DIR)
    
    plt.show()
    return None

def visualize_training_history(history, save=True):
    """
    Visualize the training history.
    
    Args:
        history: Dictionary containing training history (loss, accuracy, etc.)
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot training and validation accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        return save_figure(fig, "training_history", TRAINING_VIZ_DIR)
    
    plt.show()
    return None

def visualize_federated_metrics(metrics_file, save=True):
    """
    Visualize federated learning metrics.
    
    Args:
        metrics_file: Path to the metrics CSV file
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Load metrics
    metrics_df = pd.read_csv(metrics_file)
    
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    if 'train_loss' in metrics_df.columns:
        axes[0, 0].plot(metrics_df['round'], metrics_df['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
    
    # Plot validation loss
    if 'val_loss' in metrics_df.columns:
        axes[0, 1].plot(metrics_df['round'], metrics_df['val_loss'])
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
    
    # Plot training accuracy
    if 'train_accuracy' in metrics_df.columns:
        axes[1, 0].plot(metrics_df['round'], metrics_df['train_accuracy'])
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Accuracy')
    
    # Plot validation accuracy
    if 'val_accuracy' in metrics_df.columns:
        axes[1, 1].plot(metrics_df['round'], metrics_df['val_accuracy'])
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        return save_figure(fig, "federated_metrics", FEDERATED_VIZ_DIR)
    
    plt.show()
    return None

def visualize_client_metrics(client_metrics, save=True):
    """
    Visualize metrics for different clients.
    
    Args:
        client_metrics: Dictionary mapping client IDs to metrics
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss for each client
    for client_id, metrics in client_metrics.items():
        axes[0].plot(metrics['loss'], label=f'Client {client_id}')
    
    axes[0].set_title('Training Loss by Client')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot training accuracy for each client
    for client_id, metrics in client_metrics.items():
        axes[1].plot(metrics['accuracy'], label=f'Client {client_id}')
    
    axes[1].set_title('Training Accuracy by Client')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        return save_figure(fig, "client_metrics", FEDERATED_VIZ_DIR)
    
    plt.show()
    return None

def visualize_confusion_matrix(y_true, y_pred, class_names, save=True):
    """
    Visualize the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), "confusion_matrix", EVALUATION_VIZ_DIR)
    
    plt.show()
    return None

def visualize_roc_curves(y_true, y_score, class_names, save=True):
    """
    Visualize ROC curves for each class.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_score: Predicted probabilities
        class_names: List of class names
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    n_classes = len(class_names)
    
    # Convert labels to one-hot encoding if needed
    if len(y_true.shape) == 1:
        y_true_onehot = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_onehot[i, y_true[i]] = 1
        y_true = y_true_onehot
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), "roc_curves", EVALUATION_VIZ_DIR)
    
    plt.show()
    return None

def visualize_feature_maps(model, image_tensor, layer_name, save=True):
    """
    Visualize feature maps from a specific layer of the model.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor
        layer_name: Name of the layer to visualize
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Create a hook to get the feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # Register the hook
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    
    # Forward pass
    with torch.no_grad():
        model(image_tensor)
    
    # Remove the hook
    hook.remove()
    
    # Get the feature maps
    if not feature_maps:
        logger.error(f"Layer {layer_name} not found in the model")
        return None
    
    feature_map = feature_maps[0][0].cpu().numpy()
    
    # Create a figure
    num_features = min(16, feature_map.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # Plot feature maps
    for i in range(num_features):
        row, col = i // 4, i % 4
        axes[row, col].imshow(feature_map[i], cmap='viridis')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Feature Maps from {layer_name}')
    
    # Save the figure if requested
    if save:
        return save_figure(fig, f"feature_maps_{layer_name}", INTERPRETABILITY_VIZ_DIR)
    
    plt.show()
    return None

def custom_gradcam(model, image_tensor, target_layer_name, class_idx=None, save=True):
    """
    Custom implementation of Grad-CAM that doesn't rely on the pytorch-grad-cam package.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor
        target_layer_name: Name of the target layer for Grad-CAM
        class_idx: Class index to visualize (None for predicted class)
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    try:
        # Set the model to evaluation mode
        model.eval()
        
        # Get the target layer
        target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            logger.error(f"Layer with name '{target_layer_name}' not found in the model")
            # Print available layers for debugging
            logger.info("Available layers:")
            for name, _ in model.named_modules():
                logger.info(f"  {name}")
            return None
        
        # Register hooks
        gradients = []
        activations = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register the hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        model.zero_grad()
        output = model(image_tensor)
        
        # If class_idx is None, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # One-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Remove the hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Get the gradients and activations
        gradients = gradients[0].detach().cpu().numpy()[0]  # [C, H, W]
        activations = activations[0].detach().cpu().numpy()[0]  # [C, H, W]
        
        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Create weighted activation map
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to the CAM
        cam = np.maximum(cam, 0)
        
        # Normalize the CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize the CAM to the input image size
        original_h, original_w = image_tensor.shape[2], image_tensor.shape[3]
        cam = cv2.resize(cam, (original_w, original_h))
        
        # Convert image tensor to numpy array
        image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        
        # Normalize image for visualization
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on image
        cam_image = (heatmap / 255.0 * 0.4 + image * 0.6)
        cam_image = cam_image / np.max(cam_image)
        cam_image = np.uint8(255 * cam_image)
        
        # Create a figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot Grad-CAM
        axes[1].imshow(cam_image)
        axes[1].set_title('Grad-CAM Visualization')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save:
            return save_figure(fig, "custom_gradcam", INTERPRETABILITY_VIZ_DIR)
        
        plt.show()
        return None
    
    except Exception as e:
        logger.error(f"Error generating custom Grad-CAM: {str(e)}")
        return None

def visualize_gradcam(model, image_tensor, target_layer, class_idx=None, save=True):
    """
    Visualize Grad-CAM for the model.
    
    Note: This function requires the grad-cam package.
    You can install it with: pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
    If the package is not available, this function will return None.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor
        target_layer: Target layer for Grad-CAM
        class_idx: Class index to visualize (None for predicted class)
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    try:
        # Import grad_cam
        try:
            from grad_cam.grad_cam import GradCAM
            from grad_cam.utils.image import show_cam_on_image
        except ImportError:
            logger.warning("grad-cam package not found. Using custom_gradcam instead.")
            # Get the layer name from the target_layer
            target_layer_name = None
            for name, module in model.named_modules():
                if module is target_layer:
                    target_layer_name = name
                    break
            
            if target_layer_name is None:
                logger.error("Could not find the layer name for the target layer")
                return None
            
            return custom_gradcam(model, image_tensor, target_layer_name, class_idx, save)
        
        # Create GradCAM
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=image_tensor, target_category=class_idx)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert image tensor to numpy array
        image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        
        # Normalize image for visualization
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create visualization
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        
        # Create a figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot Grad-CAM
        axes[1].imshow(cam_image)
        axes[1].set_title('Grad-CAM Visualization')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save:
            return save_figure(fig, "gradcam", INTERPRETABILITY_VIZ_DIR)
        
        plt.show()
        return None
    
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return None

def visualize_tsne(features, labels, class_names, perplexity=30, n_iter=1000, save=True):
    """
    Visualize t-SNE embedding of features.
    
    Args:
        features: Feature vectors
        labels: Labels
        class_names: List of class names
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot t-SNE
    for i, class_name in enumerate(class_names):
        indices = np.where(labels == i)[0]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=class_name)
    
    plt.title('t-SNE Visualization of Features')
    plt.legend()
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), "tsne", EVALUATION_VIZ_DIR)
    
    plt.show()
    return None

def visualize_privacy_budget(epsilon_values, delta=1e-5, save=True):
    """
    Visualize privacy budget over training rounds.
    
    Args:
        epsilon_values: List of epsilon values for each round
        delta: Delta value for differential privacy
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot privacy budget
    plt.plot(range(1, len(epsilon_values) + 1), epsilon_values, marker='o')
    plt.title(f'Privacy Budget (ε, δ) with δ = {delta}')
    plt.xlabel('Round')
    plt.ylabel('Epsilon (ε)')
    plt.grid(True)
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), "privacy_budget", FEDERATED_VIZ_DIR)
    
    plt.show()
    return None

def visualize_model_comparison(models, metrics, metric_name='accuracy', save=True):
    """
    Visualize comparison of different models.
    
    Args:
        models: List of model names
        metrics: List of metric values
        metric_name: Name of the metric
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot model comparison
    bars = plt.bar(models, metrics)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.title(f'Model Comparison ({metric_name.capitalize()})')
    plt.xlabel('Model')
    plt.ylabel(metric_name.capitalize())
    plt.ylim(0, max(metrics) * 1.1)
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), f"model_comparison_{metric_name}", EVALUATION_VIZ_DIR)
    
    plt.show()
    return None

def visualize_data_distribution_by_client(client_labels, class_names, save=True):
    """
    Visualize data distribution across clients.
    
    Args:
        client_labels: Dictionary mapping client IDs to labels
        class_names: List of class names
        save: Whether to save the figure
        
    Returns:
        Path to the saved figure if save=True, None otherwise
    """
    # Count class distribution for each client
    client_counts = {}
    for client_id, labels in client_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        client_counts[client_id] = {class_names[i]: count for i, count in zip(unique, counts)}
    
    # Create a DataFrame for easier plotting
    data = []
    for client_id, counts in client_counts.items():
        for class_name in class_names:
            data.append({
                'Client': f'Client {client_id}',
                'Class': class_name,
                'Count': counts.get(class_name, 0)
            })
    
    df = pd.DataFrame(data)
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Plot data distribution
    ax = sns.barplot(x='Client', y='Count', hue='Class', data=df)
    
    plt.title('Data Distribution Across Clients')
    plt.xlabel('Client')
    plt.ylabel('Number of Images')
    plt.legend(title='Class')
    
    # Save the figure if requested
    if save:
        return save_figure(plt.gcf(), "data_distribution_by_client", DATA_VIZ_DIR)
    
    plt.show()
    return None

def main():
    """Test the visualization functions."""
    logger.info("Testing visualization functions...")
    
    # Create some dummy data
    np.random.seed(42)
    num_samples = 100
    num_classes = 3
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Generate random features
    features = np.random.randn(num_samples, 10)
    
    # Generate random probabilities
    probabilities = np.random.rand(num_samples, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Generate random predictions
    predictions = np.argmax(probabilities, axis=1)
    
    # Class names
    class_names = ['Type_1', 'Type_2', 'Type_3']
    
    # Test class distribution visualization
    visualize_class_distribution(labels, class_names)
    
    # Test confusion matrix visualization
    visualize_confusion_matrix(labels, predictions, class_names)
    
    # Test ROC curves visualization
    y_true_onehot = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        y_true_onehot[i, labels[i]] = 1
    
    visualize_roc_curves(y_true_onehot, probabilities, class_names)
    
    # Test t-SNE visualization
    visualize_tsne(features, labels, class_names)
    
    # Test model comparison visualization
    models = ['SimpleCNN', 'EfficientNet', 'ResNet']
    metrics = [0.85, 0.92, 0.89]
    visualize_model_comparison(models, metrics)
    
    # Test privacy budget visualization
    epsilon_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    visualize_privacy_budget(epsilon_values)
    
    # Test data distribution by client visualization
    client_labels = {
        0: np.random.randint(0, num_classes, 30),
        1: np.random.randint(0, num_classes, 40),
        2: np.random.randint(0, num_classes, 30)
    }
    visualize_data_distribution_by_client(client_labels, class_names)
    
    logger.info("Visualization tests completed!")

if __name__ == "__main__":
    main() 