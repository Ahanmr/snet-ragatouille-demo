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
