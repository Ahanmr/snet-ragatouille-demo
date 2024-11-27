<div align="center">
  <img src="logo.jpg" alt="RAGatouille Logo" width="200"/>

  # RAGatouille Demo
  ### Risk-Aware Synthetic Data Generation & Assessment

  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![SingularityNET](https://img.shields.io/badge/SingularityNET-Hackathon-blue)](https://singularitynet.io/)

  *A tool for generating and assessing synthetic prediction data with Photrek's risk assessment framework*

  [Getting Started](#installation) •
  [Documentation](#components) •
  [Demo](#streamlit-demo) •
  [API Reference](#api-endpoints)
</div>

---

# Overview
Submission to the SingularityNET's Photrek Risk Assessment Hackathon

## Environment Setup

Set the following environment variables:

```bash
export SNET_PRIVATE_KEY="your_private_key_here"
export ETH_RPC_ENDPOINT="https://mainnet.infura.io/v3/1c36b436da7645b6936ebf2e8156e6a7"
export PORT=5000  # optional, defaults to 5000
```

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Components

### 1. REST API Server
Start the REST API server:

```bash
python run.py
```

The server will start on http://localhost:5000 (or your configured PORT).

### 2. Streamlit Demo
Start the Streamlit demo application:

```bash
streamlit run app.py
```

The demo will be available at http://localhost:8501

The demo offers two main data generation modes:

#### A. General Synthetic Data
Configure synthetic data generation with:

- Number of classes (2-10)
- Number of samples (100-10000)
- Distribution type (Normal, Uniform, Beta, Gamma)
- Noise level (0-1): Controls prediction randomness
- Bias strength (0-1): Controls systematic bias
- Optional train-test split with stratification
- Optional class imbalance settings

#### B. Climate Scenarios
Specialized data generation for climate-related predictions:

1. **Temperature Anomalies**
   - Baseline temperature
   - Warming rate
   - Temperature variability
   - Classes: Cold, Normal, Hot

2. **Extreme Events**
   - Base event rate
   - Rate increase per year
   - Classes: Normal, Extreme

3. **Sea Level Rise**
   - Rise rate (mm/year)
   - Seasonal amplitude
   - Local variability
   - Classes: Low, Medium, High

### Visualization Features

The demo provides multiple visualization options:

1. **Class Distribution**: Shows the distribution of true classes
2. **Prediction Heatmap**: Visualizes probability distributions
3. **Probability Distribution**: Box plots of prediction probabilities
4. **Feature Correlations**: Correlation matrix of features
5. **Decision Boundary**: 2D visualization of class boundaries

### Data Export
- View raw data in tabular format
- Download generated datasets as CSV
- Run risk assessment directly from the interface

## Data Generator

The data generator uses a GAN-based approach to create synthetic prediction data. You can customize:

- Number of classes (2-10): The number of possible outcomes for each prediction
- Number of samples (100-10000): How many prediction instances to generate
- Noise level (0-1): Controls randomness in predictions
  - 0: Deterministic predictions
  - 1: Highly random predictions
- Bias strength (0-1): Controls systematic bias in predictions
  - 0: No systematic bias
  - 1: Strong systematic bias towards certain classes
- Class imbalance: Optional custom probabilities for each class
  - Default: Equal probability for all classes
  - Custom: Set specific probabilities (must sum to 1)

### Effects of Parameters

1. **Noise Level**
   - Low noise: More confident and consistent predictions
   - High noise: More uncertain and varied predictions
   - Affects the 'Decisiveness' metric

2. **Bias Strength**
   - Low bias: More balanced predictions across classes
   - High bias: Systematic preference for certain classes
   - Affects the 'Robustness' metric

3. **Class Imbalance**
   - Balanced: Equal representation of all classes
   - Imbalanced: Some classes appear more frequently
   - Affects both 'Robustness' and 'Accuracy' metrics

## API Endpoints

### Health Check
```
GET /health
```
Returns service health status.

### Assess Predictions
```
POST /assess
```
Submit predictions for assessment via CSV file upload.

Expected CSV format:
- Each row represents a prediction instance
- The last column contains the actual outcome (1-based index)
- Other columns contain predicted probabilities for each possible outcome

Example CSV row:
```
0.7,0.2,0.1,1
```
Where 0.7, 0.2, 0.1 are probabilities for each class, and 1 indicates the first class was the actual outcome.
