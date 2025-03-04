# Federated Learning for Cervical Cancer Detection

This repository contains the implementation of a federated learning system for early detection of cervical cancer from colposcopy images using the Intel MobileODT Cervical Cancer Screening dataset.

## Project Overview

Cervical cancer is one of the most preventable cancers when detected early. This project aims to develop a privacy-preserving federated learning system that can analyze colposcopy images to detect early signs of cervical cancer. By using federated learning, we enable multiple healthcare institutions to collaboratively train a model without sharing sensitive patient data.

## Dataset

The project uses the Intel MobileODT Cervical Cancer Screening dataset, which contains colposcopy images classified into three types:

- Type 1: Normal cervix
- Type 2: Cervical intraepithelial neoplasia (CIN)
- Type 3: Invasive carcinoma

## Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architectures
│   ├── federated/      # Federated learning implementation
│   ├── utils/          # Utility functions
│   ├── visualization/  # Visualization tools
│   ├── evaluation/     # Model evaluation
│   └── web_app/        # Web application for demonstration
├── notebooks/          # Jupyter notebooks for exploration
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Features

- **Federated Learning**: Implementation using Flower framework
- **Deep Learning Models**: CNN architectures for image classification
- **Privacy Preservation**: Differential privacy mechanisms
- **Model Interpretability**: Grad-CAM visualizations
- **Web Application**: Demo interface for model testing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/RD-Bhowmik/fedarated_learning.git
cd federated-cervical-cancer-detection
```

```bash
sudo apt  install python3.11-venv
python3.11 -m venv <environment_name>
source <environment_name>/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda]

```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the dataset:

```bash
# Instructions for dataset setup
```

### Data Preprocessing

```bash
python src/data/preprocess.py
```

### Training

```bash
python src/federated/server.py
```

### Evaluation

```bash
python src/evaluation/evaluate.py
```

### Web Application

```bash
python src/web_app/app.py
```

## Contributing

- Ronodeep Bhowmik
- Anwesha Roy
- Shaptarshi Saha

## Acknowledgments

- Intel MobileODT for providing the dataset
