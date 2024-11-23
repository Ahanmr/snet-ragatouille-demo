import streamlit as st
import pandas as pd
import requests
import io
from data_generator import DataGenerator
import os
import json

def generate_and_assess_data(
    num_classes,
    num_samples,
    noise_level,
    bias_strength,
    class_imbalance=None
):
    # Generate synthetic data
    generator = DataGenerator(num_classes=num_classes)
    df = generator.generate_dataset(
        num_samples=num_samples,
        noise_level=noise_level,
        bias_strength=bias_strength,
        class_imbalance=class_imbalance
    )
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)
    csv_string = csv_buffer.getvalue()
    
    # Call assessment service
    files = {'file': ('data.csv', csv_string)}
    response = requests.post(
        f"http://localhost:{os.getenv('PORT', '5000')}/assess",
        files=files
    )
    
    return response.json(), df

def main():
    st.title("Risk-Aware Assessment Demo")
    st.write("""
    Generate synthetic prediction data and assess it using the SingularityNET 
    Risk-Aware Assessment service.
    """)
    
    # Parameters
    with st.sidebar:
        st.header("Generation Parameters")
        num_classes = st.slider("Number of Classes", 2, 10, 3)
        num_samples = st.slider("Number of Samples", 100, 10000, 1000)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
        bias_strength = st.slider("Bias Strength", 0.0, 1.0, 0.3)
        
        # Optional class imbalance
        use_imbalance = st.checkbox("Use Class Imbalance")
        class_imbalance = None
        if use_imbalance:
            st.write("Class Probabilities (must sum to 1)")
            probs = []
            remaining = 1.0
            for i in range(num_classes):
                if i == num_classes - 1:
                    prob = remaining
                else:
                    prob = st.number_input(
                        f"Class {i+1}", 
                        0.0, 
                        remaining,
                        remaining/2 if remaining > 0 else 0.0,
                        key=f"class_{i}"
                    )
                    remaining -= prob
                probs.append(prob)
            class_imbalance = probs
    
    if st.button("Generate and Assess"):
        with st.spinner("Generating data and running assessment..."):
            try:
                result, df = generate_and_assess_data(
                    num_classes,
                    num_samples,
                    noise_level,
                    bias_strength,
                    class_imbalance
                )
                
                # Display results
                st.header("Assessment Results")
                
                metrics = result["metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                col2.metric("Decisiveness", f"{metrics['decisiveness']:.3f}")
                col3.metric("Robustness", f"{metrics['robustness']:.3f}")
                
                # Display visualization if available
                if result.get("visualization"):
                    st.image(result["visualization"])
                
                # Show sample of generated data
                st.header("Generated Data Sample")
                st.dataframe(df.head())
                
                # Download button for full dataset
                csv = df.to_csv(index=False, header=False)
                st.download_button(
                    "Download Full Dataset",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 