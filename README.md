# Cosmetics Product Category Classification

This project is a multi-modal classification model aimed at predicting product categories for cosmetics. It leverages both textual and image data to build a robust classifier using advanced feature extraction methods and CatBoost for handling highly imbalanced classes. A FastAPI server is provided for real-time model inference, and the entire project is dockerized for easy deployment based on the text-only model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Development](#local-development)
  - [Running with Docker](#running-with-docker)

## Overview

The goal of this project is to build a robust classifier that can accurately predict product categories in the cosmetics domain by:

- Extracting and fusing text features (e.g., product names/descriptions) and image features.
- Deploying the model using Docker
- Serving predictions via a FastAPI endpoint.

The initial model achieves approximately 74% accuracy. This README outlines the project’s components and suggests several avenues for further improvement.

## Features

- **Text Feature Extraction:** Utilizes techniques like TF-IDF (or transformer-based embeddings) for semantic representation, using Catboost.

- **Image Feature Extraction:** Uses a pre-trained ResNet50 (with the final classification layer replaced) to extract embeddings; dimensionality is further reduced via PCA.

- **CatBoost Classifier:** A robust gradient boosting model that natively handles categorical data.

- **FastAPI Deployment:** Provides real-time inference through a FastAPI server.

- **Dockerized Environment:** Complete Docker configuration for building and deploying the application.

# Project Structure

```

├── README.md          <- Contains the instructions to replicate code
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks, for testing and script development
│
├── pyproject.toml     <- Project configuration file with package metadata.
│         
├── reports            <- Contains the report of the challenge in pdf form
│         
├── app            <- Contains the FastAPI docker file to run.
│
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes a Python module
    │
    ├── consts.py               <- Store useful variables and configuration
    │
    ├── data_ingester.py        <- Script to download  data from API
    │
    ├── image_embeddings_generator.py             <- Generate Image embeddings
    │
    ├── train_mlflow_logging_no_image.py.py             <- Training script
    │               
```

Key thing to note is the data directory is not pushed to git and mlflow runs are local, not cloud based.


## Installation

### Prerequisites

- Python 3.11 (or compatible version)
- uv and pip package manager
- Docker (if deploying via Docker)

### Local Setup

This project uses uv as a package manager, read more about it here (https://github.com/astral-sh/uv)

1. **Clone the repository:**

```bash
git clone https://github.com/kameshkotwani/qogita.git
cd qogita
```

#### Docker Deployment

```bash
    # in the qogita directory, there is an app folder, switch to the directory
    # this will build the image along with the model and the requirements
    cd app
    docker build -t productsapi .

    # after the build is complete this command will expose the 5000 port to run the app
    docker docker run -d -p 5000:5000 productsapi    
```

After the deployment is complete, we can send a json request to (localhost:5000/classify) endpoint, using the curl, Postman or FASTAPI docs (localhost:5000/docs).

It accepts two features `name` and `brandName` in JSON format, and returns the prediction in JSON format

```JSON
{
  'product':'string',
  'prediction':'string'
}
```
