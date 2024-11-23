# snet-ragatouille-demo
Submission to the SingularityNET's Photrek Risk Assessment Hackathon

## Environment Setup

Set the following environment variables:

```bash
export SNET_PRIVATE_KEY="your_private_key_here"
export ETH_RPC_ENDPOINT="your_ethereum_endpoint"
export PORT=5000  # optional, defaults to 5000
```
## Running the Service

Start the REST API server:

```bash
python run.py
```

The server will start on http://localhost:5000 (or your configured PORT).

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
