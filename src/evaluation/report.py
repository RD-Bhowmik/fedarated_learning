"""
Module for generating and saving performance reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.helpers import setup_logger

# Set up logger
logger = setup_logger('report', 'report.log')

def generate_performance_report(
    y_true, 
    y_pred, 
    y_prob=None, 
    model_name="model", 
    input_shape="[(None, 224, 224, 3)]",
    train_metrics=None, 
    val_metrics=None,
    test_loss=None,
    output_dir="evaluation_results"
):
    """
    Generate a comprehensive performance report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        model_name: Name of the model
        input_shape: Input shape of the model
        train_metrics: Dictionary containing training metrics
        val_metrics: Dictionary containing validation metrics
        test_loss: Test loss value
        output_dir: Directory to save the report
        
    Returns:
        report_text: The generated report as text
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        specificity = np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_true == 0)
        precision = precision_score(y_true, y_pred, average='binary')
    else:
        specificity = None
        precision = precision_score(y_true, y_pred, average='macro')
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert to DataFrame for tabulate
    report_df = pd.DataFrame(class_report).transpose()
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate probability statistics if provided
    prob_stats = None
    if y_prob is not None:
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            # For multi-class, take the max probability
            prob_values = np.max(y_prob, axis=1)
        else:
            # For binary classification
            prob_values = y_prob.flatten()
            
        prob_stats = {
            'Mean': np.mean(prob_values),
            'Std': np.std(prob_values),
            'Min': np.min(prob_values),
            'Max': np.max(prob_values)
        }
    
    # Create the report text
    report_text = f"{model_name} Performance Report\n"
    report_text += "=" * 50 + "\n"
    report_text += f"Model Architecture: {model_name}\n"
    report_text += f"Input Shape: {input_shape}\n\n"
    
    # Add training metrics if provided
    if train_metrics and val_metrics:
        report_text += "Training Metrics:\n"
        report_text += f"- Final Training Accuracy: {train_metrics.get('accuracy', 'N/A'):.4f}\n"
        report_text += f"- Final Validation Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}\n"
        report_text += f"- Final Training Loss: {train_metrics.get('loss', 'N/A'):.4f}\n"
        report_text += f"- Final Validation Loss: {val_metrics.get('loss', 'N/A'):.4f}\n\n"
    
    # Add test metrics
    report_text += "Test Metrics:\n"
    if test_loss is not None:
        report_text += f"- loss: {test_loss:.4f}\n"
    report_text += f"- accuracy: {accuracy:.4f}\n"
    if specificity is not None:
        report_text += f"- specificity: {specificity:.4f}\n"
    report_text += f"- precision: {precision:.4f}\n\n"
    
    # Add classification report
    report_text += "Classification Report:\n"
    report_text += tabulate(report_df, headers='keys', tablefmt='pretty') + "\n\n"
    
    # Add confusion matrix
    report_text += "Confusion Matrix:\n"
    cm_str = str(cm).replace('\n', '\n ')
    report_text += cm_str + "\n\n"
    
    # Add probability statistics if available
    if prob_stats:
        report_text += "Probabilities Summary:\n"
        report_text += f"Mean: {prob_stats['Mean']:.4f}\n"
        report_text += f"Std: {prob_stats['Std']:.4f}\n"
        report_text += f"Min: {prob_stats['Min']:.4f}\n"
        report_text += f"Max: {prob_stats['Max']:.4f}\n"
    
    # Save the report to a text file
    report_path = os.path.join(output_dir, f"{model_name}_performance_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Performance report saved to {report_path}")
    
    # Generate and save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix visualization saved to {cm_path}")
    
    return report_text

def generate_federated_performance_report(
    metrics_file,
    model_name="federated_model",
    output_dir="evaluation_results"
):
    """
    Generate a performance report for federated learning.
    
    Args:
        metrics_file: Path to the CSV file containing metrics from federated learning
        model_name: Name of the model
        output_dir: Directory to save the report
        
    Returns:
        report_text: The generated report as text
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics data from file
    try:
        metrics_data = pd.read_csv(metrics_file)
        logger.info(f"Loaded metrics data from {metrics_file}")
    except Exception as e:
        logger.error(f"Error loading metrics data from {metrics_file}: {str(e)}")
        return f"Error loading metrics data: {str(e)}"
    
    if metrics_data is None or metrics_data.empty or len(metrics_data.columns) <= 1:
        logger.warning(f"No valid metrics data in {metrics_file}. Creating a basic report.")
        
        # Create a basic report
        report_text = f"{model_name} Federated Learning Performance Report\n"
        report_text += "=" * 50 + "\n"
        report_text += f"Model Architecture: {model_name}\n"
        report_text += f"Input Shape: [(None, 224, 224, 3)]\n\n"
        report_text += "No valid metrics data available in the metrics file.\n"
        report_text += "This could be due to:\n"
        report_text += "1. No metrics were collected during training\n"
        report_text += "2. The metrics file format is incorrect\n"
        report_text += "3. The federated learning process did not complete successfully\n\n"
        report_text += f"Please check the metrics file at: {metrics_file}\n"
        
        # Save the report to a text file
        report_path = os.path.join(output_dir, f"{model_name}_federated_performance_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Basic federated performance report saved to {report_path}")
        
        return report_path
    
    # Check if the metrics file has the expected columns
    expected_columns = ['round', 'train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']
    missing_columns = [col for col in expected_columns if col not in metrics_data.columns]
    
    if missing_columns:
        logger.warning(f"Metrics file is missing expected columns: {missing_columns}")
        
        # Create a report with available data
        report_text = f"{model_name} Federated Learning Performance Report\n"
        report_text += "=" * 50 + "\n"
        report_text += f"Model Architecture: {model_name}\n"
        report_text += f"Input Shape: [(None, 224, 224, 3)]\n\n"
        
        report_text += "Available Metrics:\n"
        for col in metrics_data.columns:
            if col != 'round':
                report_text += f"- {col}\n"
        
        report_text += "\nNote: Some expected metrics are missing from the metrics file.\n"
        report_text += f"Missing columns: {missing_columns}\n\n"
        
        # Include available data
        report_text += "Available Data:\n"
        report_text += metrics_data.to_string() + "\n\n"
        
        # Save the report to a text file
        report_path = os.path.join(output_dir, f"{model_name}_federated_performance_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Limited federated performance report saved to {report_path}")
        
        return report_path
    
    # Extract the final round metrics
    final_round = metrics_data['round'].max()
    final_metrics = metrics_data[metrics_data['round'] == final_round].iloc[0].to_dict()
    
    # Prepare metrics for the report
    train_metrics = {
        'accuracy': final_metrics.get('train_accuracy', 'N/A'),
        'loss': final_metrics.get('train_loss', 'N/A')
    }
    
    val_metrics = {
        'accuracy': final_metrics.get('val_accuracy', 'N/A'),
        'loss': final_metrics.get('val_loss', 'N/A')
    }
    
    # Extract precision, recall, and F1 scores for each class
    class_metrics = {}
    for i in range(3):  # Assuming 3 classes
        if f'val_precision_class_{i}' in final_metrics:
            class_metrics[f'Class {i}'] = {
                'precision': final_metrics.get(f'val_precision_class_{i}', 'N/A'),
                'recall': final_metrics.get(f'val_recall_class_{i}', 'N/A'),
                'f1-score': final_metrics.get(f'val_f1_class_{i}', 'N/A'),
                'support': 'N/A'  # We don't have support information
            }
    
    # Add macro and weighted averages
    class_metrics['macro avg'] = {
        'precision': final_metrics.get('val_macro_precision', 'N/A'),
        'recall': final_metrics.get('val_macro_recall', 'N/A'),
        'f1-score': final_metrics.get('val_macro_f1', 'N/A'),
        'support': 'N/A'
    }
    
    class_metrics['weighted avg'] = {
        'precision': final_metrics.get('val_weighted_precision', 'N/A'),
        'recall': final_metrics.get('val_weighted_recall', 'N/A'),
        'f1-score': final_metrics.get('val_weighted_f1', 'N/A'),
        'support': 'N/A'
    }
    
    # Convert to DataFrame for tabulate
    report_df = pd.DataFrame(class_metrics).transpose()
    
    # Create the report text
    report_text = f"{model_name} Federated Learning Performance Report\n"
    report_text += "=" * 50 + "\n"
    report_text += f"Model Architecture: {model_name}\n"
    report_text += f"Input Shape: [(None, 224, 224, 3)]\n\n"
    
    # Add training metrics
    report_text += "Training Metrics:\n"
    
    # Handle the case where metrics might be strings or missing
    try:
        train_acc = float(train_metrics['accuracy'])
        report_text += f"- Final Training Accuracy: {train_acc:.4f}\n"
    except (ValueError, TypeError):
        report_text += f"- Final Training Accuracy: {train_metrics['accuracy']}\n"
        
    try:
        val_acc = float(val_metrics['accuracy'])
        report_text += f"- Final Validation Accuracy: {val_acc:.4f}\n"
    except (ValueError, TypeError):
        report_text += f"- Final Validation Accuracy: {val_metrics['accuracy']}\n"
        
    try:
        train_loss = float(train_metrics['loss'])
        report_text += f"- Final Training Loss: {train_loss:.4f}\n"
    except (ValueError, TypeError):
        report_text += f"- Final Training Loss: {train_metrics['loss']}\n"
        
    try:
        val_loss = float(val_metrics['loss'])
        report_text += f"- Final Validation Loss: {val_loss:.4f}\n\n"
    except (ValueError, TypeError):
        report_text += f"- Final Validation Loss: {val_metrics['loss']}\n\n"
    
    # Add test metrics
    report_text += "Test Metrics:\n"
    
    try:
        val_loss = float(val_metrics['loss'])
        report_text += f"- loss: {val_loss:.4f}\n"
    except (ValueError, TypeError):
        report_text += f"- loss: {val_metrics['loss']}\n"
        
    try:
        val_acc = float(val_metrics['accuracy'])
        report_text += f"- accuracy: {val_acc:.4f}\n"
    except (ValueError, TypeError):
        report_text += f"- accuracy: {val_metrics['accuracy']}\n"
        
    try:
        macro_precision = float(class_metrics['macro avg']['precision'])
        report_text += f"- precision: {macro_precision:.4f}\n\n"
    except (ValueError, TypeError):
        report_text += f"- precision: {class_metrics['macro avg']['precision']}\n\n"
    
    # Add classification report
    report_text += "Classification Report:\n"
    report_text += tabulate(report_df, headers='keys', tablefmt='pretty') + "\n\n"
    
    # Save the report to a text file
    report_path = os.path.join(output_dir, f"{model_name}_federated_performance_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Federated performance report saved to {report_path}")
    
    # Generate and save training curves if possible
    if all(col in metrics_data.columns for col in ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss']):
        try:
            # Plot training and validation accuracy
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(metrics_data['round'], metrics_data['train_accuracy'], 'b-', label='Training Accuracy')
            plt.plot(metrics_data['round'], metrics_data['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.title('Accuracy vs. Round')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(metrics_data['round'], metrics_data['train_loss'], 'b-', label='Training Loss')
            plt.plot(metrics_data['round'], metrics_data['val_loss'], 'r-', label='Validation Loss')
            plt.title('Loss vs. Round')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            curves_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
            plt.savefig(curves_path)
            plt.close()
            logger.info(f"Training curves saved to {curves_path}")
        except Exception as e:
            logger.error(f"Error generating training curves: {str(e)}")
    else:
        logger.warning("Cannot generate training curves: required columns are missing")
    
    return report_path 