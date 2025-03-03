"""
Script to evaluate the trained model on the test dataset.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import IMAGE_SIZE, MODEL_SAVE_DIR
from src.utils.helpers import setup_logger, get_available_device, plot_confusion_matrix, print_classification_report
from src.models.model import get_pytorch_model
from src.data.dataloader import CervicalCancerDataset
from src.data.preprocess import load_image, get_augmentation_pipeline
# Import visualization module
from src.visualization.visualize import (
    visualize_confusion_matrix,
    visualize_roc_curves,
    visualize_gradcam,
    custom_gradcam
)
# Import performance report generator
from src.evaluation.report import generate_performance_report, generate_federated_performance_report

# Set up logger
logger = setup_logger('evaluate', 'evaluate.log')

def load_model(model_name='efficientnet', model_path=None, device=None):
    """
    Load the trained model.
    
    Args:
        model_name: Name of the model
        model_path: Path to the model weights
        device: Device to use for inference
        
    Returns:
        Loaded model
    """
    if device is None:
        device = get_available_device()
    
    # Create model
    model = get_pytorch_model(model_name=model_name)
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model weights from {model_path}")
    else:
        # Try to find the latest model in the model save directory
        model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.pth') and model_name in f]
        if model_files:
            # Sort by round number (assuming format: model_name_round_X.pth)
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_model = os.path.join(MODEL_SAVE_DIR, model_files[-1])
            model.load_state_dict(torch.load(latest_model, map_location=device))
            logger.info(f"Loaded latest model weights from {latest_model}")
        else:
            logger.warning("No model weights found. Using untrained model.")
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device=None):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: Trained model
        test_loader: DataLoader for the test dataset
        device: Device to use for inference
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = get_available_device()
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate the model
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            probs = torch.nn.functional.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_roc_curves(y_true, y_score, class_names, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_score: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
        
    Returns:
        AUC scores for each class
    """
    # Use our visualization module instead
    visualize_roc_curves(y_true, y_score, class_names)
    
    # Calculate AUC for each class (keep this part)
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return roc_auc

def evaluate_and_visualize(model_name='efficientnet', model_path=None, test_metadata=None, batch_size=32, output_dir=None):
    """
    Evaluate the model and visualize the results.
    
    Args:
        model_name: Name of the model
        model_path: Path to the model weights
        test_metadata: Path to the test metadata CSV file
        batch_size: Batch size for evaluation
        output_dir: Directory to save the results
        
    Returns:
        Evaluation metrics
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_available_device()
    
    # Load model
    model = load_model(model_name, model_path, device)
    
    # Load test data
    if test_metadata is None:
        logger.error("Test metadata not provided")
        return None
    
    test_dataset = CervicalCancerDataset(
        metadata_file=test_metadata,
        transform=get_augmentation_pipeline(augmentation_level='none')
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    eval_results = evaluate_model(model, test_loader, device)
    
    y_true = eval_results['labels']
    y_pred = eval_results['predictions']
    y_score = eval_results['probabilities']
    
    # Get class names
    class_names = ['Type_1', 'Type_2', 'Type_3']
    
    # Print classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print_classification_report(report)
    
    # Convert labels to one-hot encoding for ROC curves
    y_true_onehot = np.zeros((len(y_true), len(class_names)))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    
    # Plot confusion matrix using our visualization module
    visualize_confusion_matrix(y_true, y_pred, class_names)
    
    # Plot ROC curves
    roc_auc = plot_roc_curves(y_true_onehot, y_score, class_names)
    
    # Generate Grad-CAM visualizations for a few examples
    generate_gradcam_visualizations(model, test_dataset, class_names, device, num_samples=5)
    
    # Calculate test loss (approximate since we don't have the actual loss)
    test_loss = 0.0  # This would normally be calculated during evaluation
    
    # Generate performance report
    if output_dir:
        logger.info("Generating performance report...")
        # Get model input shape
        input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        
        # Try to load training history if available
        train_metrics = None
        val_metrics = None
        try:
            history_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_history.csv")
            if os.path.exists(history_path):
                history_df = pd.read_csv(history_path)
                if not history_df.empty:
                    # Get the last epoch metrics
                    last_epoch = history_df.iloc[-1]
                    train_metrics = {
                        'accuracy': last_epoch.get('train_accuracy', None),
                        'loss': last_epoch.get('train_loss', None)
                    }
                    val_metrics = {
                        'accuracy': last_epoch.get('val_accuracy', None),
                        'loss': last_epoch.get('val_loss', None)
                    }
                    logger.info(f"Loaded training history from {history_path}")
        except Exception as e:
            logger.warning(f"Could not load training history: {str(e)}")
        
        # Generate the performance report
        report_path = generate_performance_report(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_score,
            model_name=model_name,
            input_shape=input_shape,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_loss=test_loss,
            output_dir=output_dir,
            class_names=class_names
        )
        logger.info(f"Generated performance report at {report_path}")
        
        # Save results to CSV (keep the original functionality as well)
        results = {
            'accuracy': (y_true == y_pred).mean(),
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
        
        # Add AUC for each class
        for i, class_name in enumerate(class_names):
            results[f'auc_{class_name}'] = roc_auc[i]
        
        # Save to CSV
        results_df = pd.DataFrame([results])
        results_path = os.path.join(output_dir, 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved evaluation results to {results_path}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score,
        'report': report,
        'roc_auc': roc_auc
    }

def generate_gradcam_visualizations(model, dataset, class_names, device, num_samples=5):
    """
    Generate Grad-CAM visualizations for a few examples.
    
    Args:
        model: Trained model
        dataset: Dataset containing the images
        class_names: List of class names
        device: Device to use for inference
        num_samples: Number of samples to visualize
    """
    logger.info(f"Generating Grad-CAM visualizations for {num_samples} samples per class...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get the last convolutional layer
    # This will depend on the model architecture
    if hasattr(model, 'features'):
        # For models with a features attribute (like EfficientNet)
        # Find the last convolutional layer
        last_conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
                last_conv_layer_name = name
        
        if last_conv_layer is None:
            logger.error("Could not find a convolutional layer in the model")
            return
    else:
        # For ResNet-like models
        if hasattr(model, 'layer4'):
            last_conv_layer = model.layer4[-1]
            last_conv_layer_name = 'layer4.1'
        else:
            logger.error("Model architecture not supported for Grad-CAM")
            return
    
    # Get indices for each class
    class_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Select random samples for each class
    for class_idx, indices in class_indices.items():
        if len(indices) > num_samples:
            selected_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            selected_indices = indices
        
        for idx in selected_indices:
            # Get the image and label
            image, label = dataset[idx]
            
            # Convert to tensor and add batch dimension
            input_tensor = image.unsqueeze(0).to(device)
            
            # Generate Grad-CAM visualization
            try:
                visualize_gradcam(model, input_tensor, last_conv_layer, class_idx=int(label))
                logger.info(f"Generated Grad-CAM visualization for class {class_names[label]}")
            except Exception as e:
                logger.error(f"Error generating Grad-CAM visualization: {str(e)}")
                
                # Try custom Grad-CAM implementation
                try:
                    custom_gradcam(model, input_tensor, last_conv_layer_name, class_idx=int(label))
                    logger.info(f"Generated custom Grad-CAM visualization for class {class_names[label]}")
                except Exception as e:
                    logger.error(f"Error generating custom Grad-CAM visualization: {str(e)}")

def main():
    """Parse command line arguments and evaluate the model."""
    parser = argparse.ArgumentParser(description='Evaluate Trained Model')
    parser.add_argument('--model_name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--test_metadata', type=str, default=None, help='Path to test metadata CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save the results')
    parser.add_argument('--federated', action='store_true', help='Generate federated performance report')
    parser.add_argument('--metrics_file', type=str, default=None, help='Path to federated metrics CSV file (required if --federated is set)')
    args = parser.parse_args()
    
    if args.federated:
        if not args.metrics_file or not os.path.exists(args.metrics_file):
            logger.error("Metrics file is required for federated performance report")
            sys.exit(1)
        
        # Generate federated performance report
        logger.info(f"Generating federated performance report from {args.metrics_file}...")
        report_path = generate_federated_performance_report(
            metrics_file=args.metrics_file,
            model_name=args.model_name,
            output_dir=args.output_dir or 'federated_evaluation_results'
        )
        logger.info(f"Generated federated performance report at {report_path}")
    else:
        # Regular model evaluation
        evaluate_and_visualize(
            model_name=args.model_name,
            model_path=args.model_path,
            test_metadata=args.test_metadata,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()

 