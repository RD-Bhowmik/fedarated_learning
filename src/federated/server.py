"""
Federated learning server implementation using Flower.
"""

import os
import sys
import numpy as np
import torch
import flwr as fl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    NUM_ROUNDS, FRACTION_FIT, LOCAL_EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR
)
from src.utils.helpers import setup_logger, set_seed
from src.models.model import get_pytorch_model
# Import performance report generator
from src.evaluation.report import generate_federated_performance_report

# Set up logger
logger = setup_logger('server', 'server.log')

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Strategy for federated learning with model saving and metrics tracking."""
    
    def __init__(
        self,
        model_name: str = 'efficientnet',
        save_dir: str = MODEL_SAVE_DIR,
        **kwargs
    ):
        """
        Initialize the strategy.
        
        Args:
            model_name: Name of the model
            save_dir: Directory to save models
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.save_dir = save_dir
        self.round_metrics: Dict[int, Dict[str, float]] = {}
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize the global model
        self.model = get_pytorch_model(model_name=self.model_name)
        
        logger.info(f"SaveModelStrategy initialized with model: {model_name}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """
        Aggregate model updates from clients and save the global model.
        
        Args:
            server_round: Current round of federated learning
            results: List of tuples (client, fit_res) from successful clients
            failures: List of failures
            
        Returns:
            Tuple of (parameters, metrics)
        """
        # Log fit results
        logger.info(f"Round {server_round} fit results: {len(results)} successful clients, {len(failures)} failures")
        if failures:
            logger.warning(f"Round {server_round} fit failures: {failures}")
        
        # Call aggregate_fit from base class (FedAvg)
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if parameters is not None:
            try:
                # Get model parameters as NumPy arrays
                params_dict = zip(self.model.state_dict().keys(), parameters)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                
                # Update the model with aggregated parameters
                self.model.load_state_dict(state_dict, strict=True)
                
                # Save the model
                save_path = os.path.join(self.save_dir, f"{self.model_name}_round_{server_round}.pth")
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved global model at round {server_round} to {save_path}")
            except Exception as e:
                logger.error(f"Error saving model at round {server_round}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"No parameters to save for round {server_round}")
        
        # Log metrics
        if metrics:
            logger.info(f"Round {server_round} aggregated fit metrics: {metrics}")
        else:
            logger.warning(f"Round {server_round} has no aggregated fit metrics")
        
        # Store metrics for this round
        self.round_metrics[server_round] = metrics
        logger.info(f"Stored metrics for round {server_round}: {metrics}")
        
        # Plot and save metrics
        self._save_metrics()
        
        return parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round of federated learning
            results: List of tuples (client, eval_res) from successful clients
            failures: List of failures
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Log evaluation results
        logger.info(f"Round {server_round} evaluation results: {results}")
        if failures:
            logger.warning(f"Round {server_round} evaluation failures: {failures}")
        
        # Call aggregate_evaluate from base class (FedAvg)
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Log metrics
        if metrics:
            logger.info(f"Round {server_round} aggregated evaluation metrics: {metrics}")
        else:
            logger.warning(f"Round {server_round} has no aggregated evaluation metrics")
        
        # Update round metrics with evaluation results
        if server_round in self.round_metrics:
            logger.info(f"Updating round {server_round} metrics with evaluation results")
            self.round_metrics[server_round].update(metrics)
        else:
            logger.info(f"Creating new entry for round {server_round} metrics with evaluation results")
            self.round_metrics[server_round] = metrics
        
        # Save metrics after each evaluation round
        self._save_metrics()
        
        return loss, metrics
    
    def _save_metrics(self):
        """Save metrics to disk and plot them."""
        # Save metrics to CSV
        metrics_path = os.path.join(self.save_dir, "metrics.csv")
        
        logger.info(f"Saving metrics to {metrics_path}")
        logger.info(f"Round metrics: {self.round_metrics}")
        
        try:
            with open(metrics_path, "w") as f:
                # Write header
                all_keys = set()
                for metrics in self.round_metrics.values():
                    all_keys.update(metrics.keys())
                
                header = "round," + ",".join(all_keys)
                f.write(header + "\n")
                logger.info(f"Metrics header: {header}")
                
                # Write data
                for round_idx, metrics in self.round_metrics.items():
                    line = f"{round_idx}"
                    for key in all_keys:
                        value = metrics.get(key, "")
                        line += f",{value}"
                    f.write(line + "\n")
                    logger.info(f"Metrics for round {round_idx}: {line}")
            
            # Plot metrics
            self._plot_metrics()
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _plot_metrics(self):
        """Plot metrics and save the plots."""
        # Extract rounds and metrics
        rounds = list(self.round_metrics.keys())
        
        if not rounds:
            logger.warning("No rounds data available for plotting")
            return
        
        logger.info(f"Plotting metrics for rounds: {rounds}")
        
        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(self.save_dir), "visualizations", "federated")
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # Plot training metrics
            if 'train_loss' in self.round_metrics[rounds[0]] and 'train_accuracy' in self.round_metrics[rounds[0]]:
                train_loss = [self.round_metrics[r].get('train_loss', 0) for r in rounds]
                train_acc = [self.round_metrics[r].get('train_accuracy', 0) for r in rounds]
                
                logger.info(f"Plotting training metrics: loss={train_loss}, accuracy={train_acc}")
                
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(rounds, train_loss, 'b-', label='Training Loss')
                plt.title('Training Loss vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(rounds, train_acc, 'r-', label='Training Accuracy')
                plt.title('Training Accuracy vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                
                plt.tight_layout()
                train_plot_path = os.path.join(vis_dir, "training_metrics.png")
                plt.savefig(train_plot_path)
                logger.info(f"Saved training metrics plot to {train_plot_path}")
                plt.close()
            else:
                logger.warning("Training metrics not available for plotting")
            
            # Plot validation metrics
            if 'val_loss' in self.round_metrics[rounds[0]] and 'val_accuracy' in self.round_metrics[rounds[0]]:
                val_loss = [self.round_metrics[r].get('val_loss', 0) for r in rounds]
                val_acc = [self.round_metrics[r].get('val_accuracy', 0) for r in rounds]
                
                logger.info(f"Plotting validation metrics: loss={val_loss}, accuracy={val_acc}")
                
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(rounds, val_loss, 'b-', label='Validation Loss')
                plt.title('Validation Loss vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(rounds, val_acc, 'r-', label='Validation Accuracy')
                plt.title('Validation Accuracy vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                
                plt.tight_layout()
                val_plot_path = os.path.join(vis_dir, "validation_metrics.png")
                plt.savefig(val_plot_path)
                logger.info(f"Saved validation metrics plot to {val_plot_path}")
                plt.close()
            else:
                logger.warning("Validation metrics not available for plotting")
            
            # Plot precision, recall, and F1 score metrics
            if all(key in self.round_metrics[rounds[0]] for key in ['val_macro_precision', 'val_macro_recall', 'val_macro_f1']):
                # Extract macro metrics
                macro_precision = [self.round_metrics[r].get('val_macro_precision', 0) for r in rounds]
                macro_recall = [self.round_metrics[r].get('val_macro_recall', 0) for r in rounds]
                macro_f1 = [self.round_metrics[r].get('val_macro_f1', 0) for r in rounds]
                
                logger.info(f"Plotting macro metrics: precision={macro_precision}, recall={macro_recall}, f1={macro_f1}")
                
                # Extract weighted metrics
                weighted_precision = [self.round_metrics[r].get('val_weighted_precision', 0) for r in rounds]
                weighted_recall = [self.round_metrics[r].get('val_weighted_recall', 0) for r in rounds]
                weighted_f1 = [self.round_metrics[r].get('val_weighted_f1', 0) for r in rounds]
                
                # Plot macro metrics
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(rounds, macro_precision, 'b-', label='Macro Precision')
                plt.title('Macro Precision vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Precision')
                plt.legend()
                
                plt.subplot(1, 3, 2)
                plt.plot(rounds, macro_recall, 'g-', label='Macro Recall')
                plt.title('Macro Recall vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Recall')
                plt.legend()
                
                plt.subplot(1, 3, 3)
                plt.plot(rounds, macro_f1, 'r-', label='Macro F1')
                plt.title('Macro F1 vs. Round')
                plt.xlabel('Round')
                plt.ylabel('F1 Score')
                plt.legend()
                
                plt.tight_layout()
                macro_plot_path = os.path.join(vis_dir, "macro_metrics.png")
                plt.savefig(macro_plot_path)
                logger.info(f"Saved macro metrics plot to {macro_plot_path}")
                plt.close()
                
                # Plot weighted metrics
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(rounds, weighted_precision, 'b-', label='Weighted Precision')
                plt.title('Weighted Precision vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Precision')
                plt.legend()
                
                plt.subplot(1, 3, 2)
                plt.plot(rounds, weighted_recall, 'g-', label='Weighted Recall')
                plt.title('Weighted Recall vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Recall')
                plt.legend()
                
                plt.subplot(1, 3, 3)
                plt.plot(rounds, weighted_f1, 'r-', label='Weighted F1')
                plt.title('Weighted F1 vs. Round')
                plt.xlabel('Round')
                plt.ylabel('F1 Score')
                plt.legend()
                
                plt.tight_layout()
                weighted_plot_path = os.path.join(vis_dir, "weighted_metrics.png")
                plt.savefig(weighted_plot_path)
                logger.info(f"Saved weighted metrics plot to {weighted_plot_path}")
                plt.close()
                
                # Plot per-class metrics for each class
                for class_idx in range(3):  # Assuming 3 classes
                    if f'val_precision_class_{class_idx}' in self.round_metrics[rounds[0]]:
                        # Extract per-class metrics
                        class_precision = [self.round_metrics[r].get(f'val_precision_class_{class_idx}', 0) for r in rounds]
                        class_recall = [self.round_metrics[r].get(f'val_recall_class_{class_idx}', 0) for r in rounds]
                        class_f1 = [self.round_metrics[r].get(f'val_f1_class_{class_idx}', 0) for r in rounds]
                        
                        logger.info(f"Plotting class {class_idx} metrics: precision={class_precision}, recall={class_recall}, f1={class_f1}")
                        
                        # Plot per-class metrics
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(1, 3, 1)
                        plt.plot(rounds, class_precision, 'b-', label=f'Class {class_idx} Precision')
                        plt.title(f'Class {class_idx} Precision vs. Round')
                        plt.xlabel('Round')
                        plt.ylabel('Precision')
                        plt.legend()
                        
                        plt.subplot(1, 3, 2)
                        plt.plot(rounds, class_recall, 'g-', label=f'Class {class_idx} Recall')
                        plt.title(f'Class {class_idx} Recall vs. Round')
                        plt.xlabel('Round')
                        plt.ylabel('Recall')
                        plt.legend()
                        
                        plt.subplot(1, 3, 3)
                        plt.plot(rounds, class_f1, 'r-', label=f'Class {class_idx} F1')
                        plt.title(f'Class {class_idx} F1 vs. Round')
                        plt.xlabel('Round')
                        plt.ylabel('F1 Score')
                        plt.legend()
                        
                        plt.tight_layout()
                        class_plot_path = os.path.join(vis_dir, f"class_{class_idx}_metrics.png")
                        plt.savefig(class_plot_path)
                        logger.info(f"Saved class {class_idx} metrics plot to {class_plot_path}")
                        plt.close()
            else:
                logger.warning("Precision, recall, and F1 metrics not available for plotting")
            
            # Plot privacy metrics if available
            if 'epsilon' in self.round_metrics[rounds[0]]:
                epsilon = [self.round_metrics[r].get('epsilon', 0) for r in rounds]
                
                logger.info(f"Plotting privacy metrics: epsilon={epsilon}")
                
                plt.figure(figsize=(8, 5))
                plt.plot(rounds, epsilon, 'g-', label='Privacy Budget (ε)')
                plt.title('Privacy Budget vs. Round')
                plt.xlabel('Round')
                plt.ylabel('Epsilon (ε)')
                plt.legend()
                
                plt.tight_layout()
                privacy_plot_path = os.path.join(vis_dir, "privacy_metrics.png")
                plt.savefig(privacy_plot_path)
                logger.info(f"Saved privacy metrics plot to {privacy_plot_path}")
                plt.close()
            else:
                logger.warning("Privacy metrics not available for plotting")
                
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average of metrics.
    
    Args:
        metrics: List of tuples (num_examples, metrics_dict)
        
    Returns:
        Weighted average metrics
    """
    # Calculate weighted average
    weighted_metrics = {}
    total_examples = 0
    
    for num_examples, metrics_dict in metrics:
        total_examples += num_examples
        for key, value in metrics_dict.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = 0
            weighted_metrics[key] += num_examples * value
    
    # Normalize by total number of examples
    if total_examples > 0:
        for key in weighted_metrics:
            weighted_metrics[key] /= total_examples
    
    return weighted_metrics

def get_evaluate_fn(model_name):
    """
    Get an evaluation function for the server.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Evaluation function
    """
    # Load validation dataset
    # Note: In a real-world scenario, you would load a separate test dataset here
    # For simplicity, we're not implementing this in this example
    
    def evaluate(server_round: int, parameters: Parameters, config: Dict[str, str]) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Evaluate the global model on the server.
        
        Args:
            server_round: Current round of federated learning
            parameters: Model parameters
            config: Configuration
            
        Returns:
            Tuple of (loss, metrics)
        """
        # In a real implementation, you would evaluate the model on a test dataset here
        # For simplicity, we're just returning None
        return None
    
    return evaluate

def start_server(
    server_address: str = "[::]:8080",
    num_rounds: int = NUM_ROUNDS,
    fraction_fit: float = FRACTION_FIT,
    model_name: str = 'efficientnet',
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2
):
    """
    Start the federated learning server.
    
    Args:
        server_address: Server address
        num_rounds: Number of federated learning rounds
        fraction_fit: Fraction of clients to sample in each round
        model_name: Name of the model
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create strategy
    strategy = SaveModelStrategy(
        model_name=model_name,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=get_evaluate_fn(model_name),
        on_fit_config_fn=lambda round_num: {
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "round_num": round_num
        },
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    
    # Start server
    logger.info(f"Starting server at {server_address}")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    # After training is complete, generate a performance report
    logger.info("Federated learning complete. Generating performance report...")
    try:
        metrics_path = os.path.join(MODEL_SAVE_DIR, "metrics.csv")
        if os.path.exists(metrics_path):
            # Create output directory for the report
            output_dir = os.path.join(os.path.dirname(MODEL_SAVE_DIR), "evaluation_results", "federated")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate the performance report
            report_path = generate_federated_performance_report(
                metrics_file=metrics_path,
                model_name=model_name,
                output_dir=output_dir
            )
            logger.info(f"Generated federated performance report at {report_path}")
        else:
            logger.warning(f"Metrics file not found at {metrics_path}. Cannot generate performance report.")
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Start the federated learning server."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--server_address', type=str, default='[::]:8080', help='Server address')
    parser.add_argument('--num_rounds', type=int, default=NUM_ROUNDS, help='Number of federated learning rounds')
    parser.add_argument('--fraction_fit', type=float, default=FRACTION_FIT, help='Fraction of clients to sample in each round')
    parser.add_argument('--model_name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('--min_fit_clients', type=int, default=2, help='Minimum number of clients for training')
    parser.add_argument('--min_evaluate_clients', type=int, default=2, help='Minimum number of clients for evaluation')
    parser.add_argument('--min_available_clients', type=int, default=2, help='Minimum number of available clients')
    args = parser.parse_args()
    
    # Start server
    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        model_name=args.model_name,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients
    )

if __name__ == "__main__":
    main() 