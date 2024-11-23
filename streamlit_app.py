import streamlit as st
import pandas as pd
import requests
import io
from data_generator import DataGenerator
import os
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Add custom styling
st.set_page_config(
    page_title="Risk-Aware Assessment Demo",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def generate_synthetic_data(num_classes, num_samples, noise_level, bias_strength, class_imbalance=None):
    """Generate and return synthetic data without making API call"""
    generator = DataGenerator(num_classes=num_classes)
    df = generator.generate_dataset(
        num_samples=num_samples,
        noise_level=noise_level,
        bias_strength=bias_strength,
        class_imbalance=class_imbalance
    )
    return df

def plot_class_distribution(df):
    # Assuming the last column is the true class
    true_class_col = df.columns[-1]
    fig = px.histogram(
        df, 
        x=true_class_col,
        title='Class Distribution',
        labels={true_class_col: 'True Class', 'count': 'Count'},
        template='plotly_white'
    )
    return fig

def plot_prediction_heatmap(df):
    # Assuming the last column is true class and second-to-last is predicted class
    true_class_col = df.columns[-1]
    pred_class_col = df.columns[-2]
    confusion_matrix = pd.crosstab(df[true_class_col], df[pred_class_col])
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        title="Prediction Heatmap",
        template='plotly_white'
    )
    return fig

def plot_probability_distribution(df, num_classes):
    """Plot distribution of prediction probabilities"""
    # Get probability columns (all except the last two columns which are predicted and true class)
    prob_cols = df.columns[:-2]
    if len(prob_cols) > 0:
        prob_data = df[prob_cols].melt()
        fig = px.box(
            prob_data,
            y='value',
            x='variable',
            title='Distribution of Prediction Probabilities',
            labels={'value': 'Probability', 'variable': 'Class'},
            template='plotly_white'
        )
        return fig
    return None

def main():
    st.title("🎯 Risk-Aware Assessment Demo")
    st.markdown("""
    Generate synthetic prediction data and assess it using the SingularityNET 
    Risk-Aware Assessment service. This demo allows you to experiment with different
    parameters and visualize the results.
    """)
    
    # Parameters
    with st.sidebar:
        st.header("📊 Generation Parameters")
        with st.expander("Basic Parameters", expanded=True):
            num_classes = st.slider("Number of Classes", 2, 10, 3)
            num_samples = st.slider("Number of Samples", 100, 10000, 1000)
            noise_level = st.slider(
                "Noise Level", 
                0.0, 1.0, 0.1, 
                help="Controls the randomness in predictions"
            )
            bias_strength = st.slider(
                "Bias Strength", 
                0.0, 1.0, 0.3,
                help="Controls the systematic bias in predictions"
            )
        
        # Optional class imbalance
        with st.expander("Advanced Parameters"):
            use_imbalance = st.checkbox("Use Class Imbalance")
            class_imbalance = None
            if use_imbalance:
                st.write("Class Probabilities (must sum to 1)")
                probs = []
                remaining = 1.0
                for i in range(num_classes):
                    if i == num_classes - 1:
                        prob = remaining
                        st.text(f"Class {i+1}: {prob:.3f}")
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

    # Generate synthetic data first
    df = generate_synthetic_data(
        num_classes,
        num_samples,
        noise_level,
        bias_strength,
        class_imbalance
    )

    # Display data visualizations
    st.header("📊 Data Overview")
    
    # Basic statistics
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", num_samples)
    col2.metric("Number of Classes", num_classes)
    
    # Get probability columns (all except the last two columns)
    prob_cols = df.columns[:-2]
    if len(prob_cols) > 0:
        avg_prob = df[prob_cols].mean().mean()
        col3.metric("Average Probability", f"{avg_prob:.3f}")
    else:
        col3.metric("Average Probability", "N/A")

    # Visualizations
    st.subheader("Data Visualizations")
    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Prediction Heatmap", "Probability Distribution"])
    
    with tab1:
        st.plotly_chart(plot_class_distribution(df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(plot_prediction_heatmap(df), use_container_width=True)
    
    with tab3:
        prob_dist_plot = plot_probability_distribution(df, num_classes)
        if prob_dist_plot:
            st.plotly_chart(prob_dist_plot, use_container_width=True)
        else:
            st.warning("Probability distribution plot not available - missing probability columns")

    # Data explorer
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df, use_container_width=True, height=300)
        
        # Download button for data
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Dataset",
            csv,
            "synthetic_predictions.csv",
            "text/csv",
            help="Download the complete dataset as CSV"
        )

    # Risk Assessment Service
    st.header("🎯 Risk Assessment")
    if st.button("Run Risk Assessment", type="primary"):
        with st.spinner("Calling Risk Assessment Service..."):
            try:
                # Prepare data for API
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, header=False)
                csv_string = csv_buffer.getvalue()
                
                # Call assessment service
                files = {'file': ('data.csv', csv_string)}
                response = requests.post(
                    f"http://localhost:{os.getenv('PORT', '5000')}/assess",
                    files=files
                )
                response.raise_for_status()
                result = response.json()
                
                # Display assessment results
                st.subheader("Assessment Results")
                metrics = result.get('metrics', {})
                
                cols = st.columns(3)
                cols[0].metric(
                    "Accuracy", 
                    f"{metrics.get('accuracy', 0.0):.3f}",
                    help="Model's prediction accuracy"
                )
                cols[1].metric(
                    "Decisiveness", 
                    f"{metrics.get('decisiveness', 0.0):.3f}",
                    help="Model's confidence in predictions"
                )
                cols[2].metric(
                    "Robustness", 
                    f"{metrics.get('robustness', 0.0):.3f}",
                    help="Model's stability across different conditions"
                )
                
                # Show service visualization if available
                if result.get("visualization"):
                    st.subheader("Service Visualization")
                    st.image(result["visualization"])
                
            except Exception as e:
                st.error(f"Error calling assessment service: {str(e)}")
                st.error("Please make sure the assessment service is running and accessible.")

if __name__ == "__main__":
    main() 