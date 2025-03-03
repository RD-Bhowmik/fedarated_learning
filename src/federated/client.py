"""
Federated learning client implementation using Flower.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS, DIFFERENTIAL_PRIVACY,
    NOISE_MULTIPLIER, MAX_GRAD_NORM
)
from src.utils.helpers import setup_logger, get_available_device
from src.data.dataloader import CervicalCancerDataset
from src.models.model import get_pytorch_model

# Set up logger
logger = setup_logger('client', 'client.log')

class CervicalCancerClient(fl.client.NumPyClient):
    """Flower client for federated learning with the cervical cancer dataset."""
    
    def __init__(
        self,
        client_id,
        train_dataset,
        val_dataset,
        model_name='efficientnet',
        batch_size=BATCH_SIZE,
        num_workers=4,
        learning_rate=LEARNING_RATE,
        local_epochs=LOCAL_EPOCHS,
        device=None,
        differential_privacy=False,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM
    ):
        """
        Initialize the client.
        
        Args:
            client_id: Client identifier
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_name: Name of the model to use
            batch_size: Batch size for training
            num_workers: Number of worker threads for data loading
            learning_rate: Learning rate for optimization
            local_epochs: Number of local training epochs
            device: Device to use for training (CPU or GPU)
            differential_privacy: Whether to use differential privacy
            noise_multiplier: Noise multiplier for differential privacy
            max_grad_norm: Maximum gradient norm for differential privacy
        """
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device if device is not None else get_available_device()
        self.differential_privacy = differential_privacy
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Create model
        self.model = get_pytorch_model(model_name=self.model_name)
        self.model.to(self.device)
        
        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up differential privacy if enabled
        if self.differential_privacy:
            try:
                from opacus import PrivacyEngine
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )
                logger.info(f"Client {self.client_id}: Differential privacy enabled")
            except ImportError:
                logger.warning("Opacus not installed. Differential privacy disabled.")
                self.differential_privacy = False
        
        logger.info(f"Client {self.client_id} initialized with {len(self.train_dataset)} training samples")
    
    def get_parameters(self, config):
        """
        Get model parameters as a list of NumPy arrays.
        
        Args:
            config: Configuration from the server
            
        Returns:
            List of model parameters as NumPy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """
        Set model parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of model parameters as NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train the model on the local dataset.
        
        Args:
            parameters: List of model parameters as NumPy arrays
            config: Configuration from the server
            
        Returns:
            Updated model parameters, number of training samples, and metrics
        """
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Get training configuration
            epochs = config.get('epochs', self.local_epochs)
            
            # Train the model
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # For precision, recall, and F1 score calculation
            all_preds = []
            all_targets = []
            
            logger.info(f"Client {self.client_id} starting training for {epochs} epochs")
            
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                logger.info(f"Client {self.client_id} - Epoch {epoch}/{epochs} starting")
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    try:
                        # Move data to device
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Zero the parameter gradients
                        self.optimizer.zero_grad()
                        
                        # Forward pass
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                        # Backward pass and optimize
                        loss.backward()
                        
                        # Apply differential privacy if enabled
                        if self.differential_privacy:
                            self.privacy_engine.step()
                        else:
                            self.optimizer.step()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        _, predicted = output.max(1)
                        epoch_total += target.size(0)
                        epoch_correct += predicted.eq(target).sum().item()
                        
                        # Store predictions and targets for precision, recall, and F1
                        all_preds.extend(predicted.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                        
                        if batch_idx % 10 == 0:
                            logger.info(f"Client {self.client_id} - Epoch {epoch}/{epochs} - Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
                    
                    except Exception as e:
                        logger.error(f"Client {self.client_id} - Error in training batch {batch_idx}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Calculate epoch metrics
                epoch_loss /= len(self.train_loader)
                epoch_acc = 100. * epoch_correct / epoch_total
                
                # Update overall metrics
                train_loss += epoch_loss
                train_correct += epoch_correct
                train_total += epoch_total
                
                logger.info(f"Client {self.client_id} - Epoch {epoch}/{epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
            # Calculate average metrics
            train_loss /= epochs
            train_acc = 100. * train_correct / train_total
            
            # Calculate precision, recall, and F1 score
            from sklearn.metrics import precision_recall_fscore_support
            
            # Convert lists to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Calculate metrics for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_preds, average=None, zero_division=0
            )
            
            # Calculate macro average metrics
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )
            
            # Calculate weighted average metrics
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            
            # Log metrics
            logger.info(f"Client {self.client_id} - Training: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"Client {self.client_id} - Precision: {precision}, Recall: {recall}, F1: {f1}")
            logger.info(f"Client {self.client_id} - Macro Avg: Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
            logger.info(f"Client {self.client_id} - Weighted Avg: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
            
            # Get updated model parameters
            parameters_updated = self.get_parameters(config)
            
            # Get privacy budget if differential privacy is enabled
            epsilon = None
            if self.differential_privacy:
                epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
                logger.info(f"Client {self.client_id} - Privacy budget: ε = {epsilon:.2f}")
            
            # Return updated parameters, number of samples, and metrics
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_macro_precision': macro_precision,
                'train_macro_recall': macro_recall,
                'train_macro_f1': macro_f1,
                'train_weighted_precision': weighted_precision,
                'train_weighted_recall': weighted_recall,
                'train_weighted_f1': weighted_f1
            }
            
            # Add per-class metrics
            for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
                metrics[f'train_precision_class_{i}'] = p
                metrics[f'train_recall_class_{i}'] = r
                metrics[f'train_f1_class_{i}'] = f
            
            # Add privacy budget if available
            if epsilon is not None:
                metrics['epsilon'] = epsilon
            
            logger.info(f"Client {self.client_id} completed training with metrics: {metrics}")
            
            return parameters_updated, train_total, metrics
        
        except Exception as e:
            logger.error(f"Client {self.client_id} - Error in fit: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local validation dataset.
        
        Args:
            parameters: List of model parameters as NumPy arrays
            config: Configuration from the server
            
        Returns:
            Loss, number of evaluation samples, and metrics
        """
        try:
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate the model
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # For precision, recall, and F1 score calculation
            all_preds = []
            all_targets = []
            
            logger.info(f"Client {self.client_id} starting evaluation")
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.val_loader):
                    try:
                        # Move data to device
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Forward pass
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        
                        # Update metrics
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                        
                        # Store predictions and targets for precision, recall, and F1
                        all_preds.extend(predicted.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                        
                        if batch_idx % 10 == 0:
                            logger.info(f"Client {self.client_id} - Evaluation - Batch {batch_idx}/{len(self.val_loader)} - Loss: {loss.item():.4f}")
                    
                    except Exception as e:
                        logger.error(f"Client {self.client_id} - Error in evaluation batch {batch_idx}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # Calculate average metrics
            val_loss /= len(self.val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Calculate precision, recall, and F1 score
            # Convert lists to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Calculate metrics for each class
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_preds, average=None, zero_division=0
            )
            
            # Calculate macro average metrics
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )
            
            # Calculate weighted average metrics
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            
            # Log metrics
            logger.info(f"Client {self.client_id} - Evaluation: Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            logger.info(f"Client {self.client_id} - Precision: {precision}, Recall: {recall}, F1: {f1}")
            logger.info(f"Client {self.client_id} - Macro Avg: Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
            logger.info(f"Client {self.client_id} - Weighted Avg: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
            
            # Return loss, number of samples, and metrics
            metrics = {
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_macro_precision': macro_precision,
                'val_macro_recall': macro_recall,
                'val_macro_f1': macro_f1,
                'val_weighted_precision': weighted_precision,
                'val_weighted_recall': weighted_recall,
                'val_weighted_f1': weighted_f1
            }
            
            # Add per-class metrics
            for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
                metrics[f'val_precision_class_{i}'] = p
                metrics[f'val_recall_class_{i}'] = r
                metrics[f'val_f1_class_{i}'] = f
            
            logger.info(f"Client {self.client_id} completed evaluation with metrics: {metrics}")
            
            return float(val_loss), val_total, metrics
        
        except Exception as e:
            logger.error(f"Client {self.client_id} - Error in evaluate: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

def create_client(client_id, train_metadata, val_metadata, model_name='efficientnet'):
    """
    Create a federated learning client.
    
    Args:
        client_id: Client identifier
        train_metadata: Path to the training metadata CSV file
        val_metadata: Path to the validation metadata CSV file
        model_name: Name of the model to use
        
    Returns:
        Flower client
    """
    # Create datasets
    train_dataset = CervicalCancerDataset(train_metadata)
    val_dataset = CervicalCancerDataset(val_metadata)
    
    # Create client
    client = CervicalCancerClient(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name
    )
    
    return client

def main():
    """Start a federated learning client."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    parser.add_argument('--server_address', type=str, default='[::]:8080', help='Server address')
    parser.add_argument('--train_metadata', type=str, required=True, help='Path to training metadata CSV')
    parser.add_argument('--val_metadata', type=str, required=True, help='Path to validation metadata CSV')
    parser.add_argument('--model_name', type=str, default='efficientnet', help='Model name')
    args = parser.parse_args()
    
    # Create client
    client = create_client(
        client_id=args.client_id,
        train_metadata=args.train_metadata,
        val_metadata=args.val_metadata,
        model_name=args.model_name
    )
    
    # Start client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main() 