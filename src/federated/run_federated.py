"""
Script to run the federated learning system with multiple clients.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import NUM_CLIENTS, NUM_ROUNDS
from src.utils.helpers import setup_logger
from src.data.dataloader import create_client_datasets

# Set up logger
logger = setup_logger('run_federated', 'run_federated.log')

def run_server(server_address, num_rounds, model_name, min_clients):
    """
    Run the federated learning server.
    
    Args:
        server_address: Server address
        num_rounds: Number of federated learning rounds
        model_name: Name of the model
        min_clients: Minimum number of clients
    
    Returns:
        Server process
    """
    logger.info(f"Starting server at {server_address}")
    
    # Build command
    cmd = [
        "python", "-m", "src.federated.server",
        "--server_address", server_address,
        "--num_rounds", str(num_rounds),
        "--model_name", model_name,
        "--min_fit_clients", str(min_clients),
        "--min_evaluate_clients", str(min_clients),
        "--min_available_clients", str(min_clients)
    ]
    
    # Start server process with output redirected to server.log
    with open("server.log", "w") as server_log:
        server_process = subprocess.Popen(
            cmd,
            stdout=server_log,
            stderr=server_log,
            text=True
        )
    
    # Give the server some time to start
    time.sleep(3)
    
    return server_process

def run_client(client_id, server_address, train_metadata, val_metadata, model_name):
    """
    Run a federated learning client.
    
    Args:
        client_id: Client ID
        server_address: Server address
        train_metadata: Path to training metadata CSV
        val_metadata: Path to validation metadata CSV
        model_name: Name of the model
    
    Returns:
        Client process
    """
    logger.info(f"Starting client {client_id}")
    
    # Build command
    cmd = [
        "python", "-m", "src.federated.client",
        "--client_id", str(client_id),
        "--server_address", server_address,
        "--train_metadata", train_metadata,
        "--val_metadata", val_metadata,
        "--model_name", model_name
    ]
    
    # Start client process with output redirected to client_{client_id}.log
    with open(f"client_{client_id}.log", "w") as client_log:
        client_process = subprocess.Popen(
            cmd,
            stdout=client_log,
            stderr=client_log,
            text=True
        )
    
    return client_process

def run_federated_learning(
    num_clients=NUM_CLIENTS,
    num_rounds=NUM_ROUNDS,
    server_address="[::]:8080",
    model_name="efficientnet",
    iid=False
):
    """
    Run the federated learning system with multiple clients.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of federated learning rounds
        server_address: Server address
        model_name: Name of the model
        iid: Whether to use IID partitioning
    """
    # Check if processed data exists
    base_dir = Path(__file__).resolve().parent.parent.parent / "processed_data"
    train_metadata = base_dir / "train_metadata.csv"
    val_metadata = base_dir / "val_metadata.csv"
    
    if not train_metadata.exists() or not val_metadata.exists():
        logger.error("Processed data not found. Please run preprocess.py first.")
        return
    
    # Create client datasets
    logger.info(f"Creating {num_clients} client datasets (IID: {iid})")
    client_metadata_files = create_client_datasets(train_metadata, num_clients, iid=iid)
    
    # Start server
    server_process = run_server(server_address, num_rounds, model_name, min_clients=num_clients)
    
    # Start clients
    client_processes = []
    for i in range(num_clients):
        client_process = run_client(
            client_id=i,
            server_address=server_address,
            train_metadata=client_metadata_files[i],
            val_metadata=val_metadata,
            model_name=model_name
        )
        client_processes.append(client_process)
    
    # Wait for server to complete
    try:
        logger.info("Waiting for server to complete...")
        server_process.wait()
        logger.info("Server completed")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Terminating processes...")
        server_process.terminate()
        for client_process in client_processes:
            client_process.terminate()
    
    # Terminate any remaining client processes
    for i, client_process in enumerate(client_processes):
        if client_process.poll() is None:
            logger.info(f"Terminating client {i}")
            client_process.terminate()
    
    logger.info("Federated learning completed")

def main():
    """Parse command line arguments and run federated learning."""
    parser = argparse.ArgumentParser(description='Run Federated Learning')
    parser.add_argument('--num_clients', type=int, default=NUM_CLIENTS, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=NUM_ROUNDS, help='Number of federated learning rounds')
    parser.add_argument('--server_address', type=str, default='[::]:8080', help='Server address')
    parser.add_argument('--model_name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('--iid', action='store_true', help='Use IID partitioning')
    args = parser.parse_args()
    
    run_federated_learning(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        server_address=args.server_address,
        model_name=args.model_name,
        iid=args.iid
    )

if __name__ == "__main__":
    main() 