#!/usr/bin/env python3
"""
Script to run the entire federated learning pipeline.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent))
from src.utils.config import NUM_CLIENTS, NUM_ROUNDS
from src.utils.helpers import setup_logger

import tensorflow as tf
print("TensorFlow is using the following GPU(s):", tf.config.list_physical_devices('GPU'))


# Set up logger
logger = setup_logger('pipeline', 'pipeline.log')

def run_command(command, description=None):
    """
    Run a shell command and log the output.
    
    Args:
        command: Command to run
        description: Description of the command
    
    Returns:
        True if the command succeeded, False otherwise
    """
    if description:
        logger.info(f"{description}...")
    
    logger.info(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    if stdout:
        logger.info(f"Command output: {stdout}")
    
    if stderr:
        logger.error(f"Command error: {stderr}")
    
    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def run_pipeline(
    preprocess=True,
    train=True,
    evaluate=True,
    web_app=False,
    generate_report=False,
    num_clients=NUM_CLIENTS,
    num_rounds=NUM_ROUNDS,
    model_name='efficientnet',
    iid=False
):
    """
    Run the entire federated learning pipeline.
    
    Args:
        preprocess: Whether to run the preprocessing step
        train: Whether to run the training step
        evaluate: Whether to run the evaluation step
        web_app: Whether to run the web app
        generate_report: Whether to generate a performance report
        num_clients: Number of clients for federated learning
        num_rounds: Number of federated learning rounds
        model_name: Name of the model to use
        iid: Whether to use IID partitioning
    """
    # Step 1: Preprocess the data
    if preprocess:
        preprocess_cmd = ["python", "-m", "src.data.preprocess"]
        if not run_command(preprocess_cmd, "Preprocessing data"):
            logger.error("Preprocessing failed. Aborting pipeline.")
            return False
    
    # Step 2: Train the model using federated learning
    if train:
        train_cmd = [
            "python", "-m", "src.federated.run_federated",
            "--num_clients", str(num_clients),
            "--num_rounds", str(num_rounds),
            "--model_name", model_name
        ]
        
        if iid:
            train_cmd.append("--iid")
        
        if not run_command(train_cmd, "Training model using federated learning"):
            logger.error("Training failed. Aborting pipeline.")
            return False
    
    # Step 3: Evaluate the model
    if evaluate:
        # Find the latest model
        model_save_dir = os.path.join(os.path.dirname(__file__), "models")
        model_files = [f for f in os.listdir(model_save_dir) if f.endswith('.pth') and model_name in f]
        
        if model_files:
            # Sort by round number (assuming format: model_name_round_X.pth)
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_model = os.path.join(model_save_dir, model_files[-1])
            
            # Find the validation metadata
            val_metadata = os.path.join(os.path.dirname(__file__), "processed_data", "val_metadata.csv")
            
            if os.path.exists(val_metadata):
                evaluate_cmd = [
                    "python", "-m", "src.evaluation.evaluate",
                    "--model_name", model_name,
                    "--model_path", latest_model,
                    "--test_metadata", val_metadata,
                    "--output_dir", "evaluation_results"
                ]
                
                if not run_command(evaluate_cmd, "Evaluating model"):
                    logger.error("Evaluation failed.")
            else:
                logger.error(f"Validation metadata not found at {val_metadata}. Skipping evaluation.")
        else:
            logger.error(f"No model files found in {model_save_dir}. Skipping evaluation.")
    
    # Step 4: Generate performance report
    if generate_report:
        # Check if metrics file exists
        metrics_file = os.path.join(os.path.dirname(__file__), "models", "metrics.csv")
        
        if os.path.exists(metrics_file):
            report_cmd = [
                "python", "-m", "src.evaluation.evaluate",
                "--federated",
                "--metrics_file", metrics_file,
                "--model_name", model_name,
                "--output_dir", "federated_evaluation_results"
            ]
            
            if not run_command(report_cmd, "Generating performance report"):
                logger.error("Performance report generation failed.")
        else:
            logger.error(f"Metrics file not found at {metrics_file}. Skipping report generation.")
    
    # Step 5: Run the web app
    if web_app:
        web_app_cmd = ["python", "-m", "src.web_app.app", "--model_name", model_name]
        
        if not run_command(web_app_cmd, "Starting web app"):
            logger.error("Web app failed to start.")
            return False
    
    logger.info("Pipeline completed successfully!")
    return True

def main():
    """Parse command line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Run Federated Learning Pipeline')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip-train', action='store_true', help='Skip training step')
    parser.add_argument('--skip-evaluate', action='store_true', help='Skip evaluation step')
    parser.add_argument('--run-web-app', action='store_true', help='Run web app')
    parser.add_argument('--generate-report', action='store_true', help='Generate performance report')
    parser.add_argument('--num-clients', type=int, default=NUM_CLIENTS, help='Number of clients')
    parser.add_argument('--num-rounds', type=int, default=NUM_ROUNDS, help='Number of federated learning rounds')
    parser.add_argument('--model-name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('--iid', action='store_true', help='Use IID partitioning')
    args = parser.parse_args()
    
    run_pipeline(
        preprocess=not args.skip_preprocess,
        train=not args.skip_train,
        evaluate=not args.skip_evaluate,
        web_app=args.run_web_app,
        generate_report=args.generate_report,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        model_name=args.model_name,
        iid=args.iid
    )

if __name__ == "__main__":
    main() 