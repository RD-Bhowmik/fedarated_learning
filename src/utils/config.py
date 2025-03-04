"""
Configuration settings for the federated learning project.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "intel-mobileodt-cervical-cancer-screening")
TRAIN_DIR = os.path.join(DATA_DIR, "train", "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
ADDITIONAL_TYPE1_DIR = os.path.join(DATA_DIR, "additional_Type_1_v2")
ADDITIONAL_TYPE2_DIR = os.path.join(DATA_DIR, "additional_Type_2_v2")
ADDITIONAL_TYPE3_DIR = os.path.join(DATA_DIR, "additional_Type_3_v2")

# Model settings
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
IMAGE_SIZE = (224, 224)  # Standard size for many CNN architectures
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 3

# Federated learning settings
NUM_CLIENTS = 3
NUM_ROUNDS = 20
LOCAL_EPOCHS = 10
FRACTION_FIT = 1.0  # Fraction of clients used for training in each round

# Privacy settings
DIFFERENTIAL_PRIVACY = True
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Evaluation settings
METRICS = ["accuracy", "precision", "recall", "f1"]
VALIDATION_SPLIT = 0.2

# Web app settings
WEB_APP_PORT = 5000
UPLOAD_FOLDER = os.path.join(BASE_DIR, "src", "web_app", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create necessary directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 