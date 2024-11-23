# snet-ragatouille-demo
Submission to the SingularityNET's Photrek Risk Assessment Hackathon

## Environment Setup

Set the following environment variables:

```bash
export SNET_PRIVATE_KEY="your_private_key_here"
export ETH_RPC_ENDPOINT="https://mainnet.infura.io/v3/09027f4a13e841d48dbfefc67e7685d5"
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
streamlit run streamlit_app.py
```

The demo will be available at http://localhost:8501

## Data Generator

The data generator uses a GAN-based approach to create synthetic prediction data. You can customize:

- Number of classes
- Number of samples
- Noise level (0-1)
- Bias strength (0-1)
- Class imbalance

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
