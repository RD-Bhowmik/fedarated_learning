"""
Web application for the federated learning project.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import MODEL_SAVE_DIR, WEB_APP_PORT, UPLOAD_FOLDER
from src.utils.helpers import setup_logger

# Set up logger
logger = setup_logger('web_app', 'web_app.log')

# Create Flask app
app = Flask(__name__)
app.secret_key = 'federated_learning_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the index page."""
    # Check if metrics file exists
    metrics_path = os.path.join(MODEL_SAVE_DIR, 'metrics.csv')
    metrics_data = None
    
    if os.path.exists(metrics_path):
        try:
            metrics_data = pd.read_csv(metrics_path)
            logger.info(f"Loaded metrics data: {metrics_data.shape}")
        except Exception as e:
            logger.error(f"Error loading metrics data: {str(e)}")
    else:
        logger.warning(f"Metrics file not found at {metrics_path}")
    
    # Get list of visualization files
    vis_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, 'visualizations')
    visualization_files = []
    
    for root, dirs, files in os.walk(vis_dir):
        for file in files:
            if file.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, file), vis_dir)
                visualization_files.append(rel_path)
    
    logger.info(f"Found {len(visualization_files)} visualization files")
    
    # Get list of model files
    model_files = []
    if os.path.exists(MODEL_SAVE_DIR):
        model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.pth')]
    
    logger.info(f"Found {len(model_files)} model files")
    
    return render_template(
        'index.html',
        metrics_data=metrics_data,
        visualization_files=visualization_files,
        model_files=model_files
    )

@app.route('/visualizations/<path:filename>')
def visualizations(filename):
    """Serve visualization files."""
    vis_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, 'visualizations')
    directory = os.path.dirname(os.path.join(vis_dir, filename))
    file = os.path.basename(filename)
    return send_from_directory(directory, file)

@app.route('/models/<filename>')
def models(filename):
    """Serve model files."""
    return send_from_directory(MODEL_SAVE_DIR, filename)

def main():
    """Run the web application."""
    app.run(host='0.0.0.0', port=WEB_APP_PORT, debug=True)

if __name__ == '__main__':
    main() 