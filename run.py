from flask import Flask, request, jsonify
from snet import sdk
import pandas as pd
import numpy as np
import io
import base64
import os

app = Flask(__name__)

# Configure SingularityNET SDK
config = sdk.config.Config(
    private_key=os.getenv("SNET_PRIVATE_KEY"),
    eth_rpc_endpoint=os.getenv("ETH_RPC_ENDPOINT"),
    concurrency=False,
    force_update=False
)

snet_sdk = sdk.SnetSDK(config)

# Create service client
service_client = snet_sdk.create_service_client(
    org_id="Photrek",
    service_id="risk-aware-assessment",
    group_name="default_group"
)

def process_csv_data(csv_data):
    """Process CSV data into required format for the service"""
    df = pd.read_csv(io.StringIO(csv_data), header=None)
    num_rows, num_cols = df.shape
    
    # Convert dataframe to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)
    csv_string = csv_buffer.getvalue()
    
    # Encode as base64
    b64_data = base64.b64encode(csv_string.encode()).decode()
    
    # Format input string as expected by service
    input_string = f"{num_rows},{num_cols},{b64_data}"
    
    return input_string

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/assess", methods=["POST"])
def assess_predictions():
    """
    Endpoint to assess prediction quality using Risk-Aware Assessment
    
    Expected input format:
    CSV data where:
    - Each row represents a prediction instance
    - The last column contains the actual outcome (1-based index)
    - Other columns contain predicted probabilities for each possible outcome
    
    Returns:
    - accuracy: How well probabilities match actual outcomes
    - decisiveness: How well probabilities support decision making
    - robustness: How well forecasts handle low-probability cases
    - plot: Base64 encoded visualization of the assessment
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded. Please upload a CSV file."
            }), 400
            
        file = request.files['file']
        
        # Read and validate CSV data
        csv_data = file.read().decode('utf-8')
        
        # Process data into required format
        input_string = process_csv_data(csv_data)
        
        # Call the service
        result = service_client.call_rpc(
            rpc_name="adr",
            message_class="InputString",
            s=input_string
        )
        
        # Format response
        response = {
            "metrics": {
                "accuracy": float(result.a),
                "decisiveness": float(result.d),
                "robustness": float(result.r)
            },
            "data": {
                "num_predictions": int(result.numr),
                "num_classes": int(result.numc)
            },
            "visualization": result.img.decode() if result.img else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.getenv("PORT", 5000))
    
    app.run(host="0.0.0.0", port=port)
