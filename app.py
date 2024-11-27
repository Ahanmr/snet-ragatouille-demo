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
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Optional, List, Tuple
from plotly.graph_objs._figure import Figure


st.set_page_config(
    page_title="RAGatouille: Scenario-based Synthetic Data Generation for Risk-Aware Assessment",
    page_icon="🐀",
    layout="wide"
)

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
    # assume the last column is the true class
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
    """Plot detailed prediction heatmap with 0.1 interval buckets"""
    prob_cols = df.columns[:-2]
    buckets = np.arange(0, 1.1, 0.1)
    bucket_labels = [f"{b:.1f}-{b+0.1:.1f}" for b in buckets[:-1]]
    heatmap_data = np.zeros((len(prob_cols), len(buckets)-1))
    for i, col in enumerate(prob_cols):
        hist, _ = np.histogram(df[col], bins=buckets)
        heatmap_data[i] = hist
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=prob_cols,
        columns=bucket_labels
    )
    fig = px.imshow(
        heatmap_df,
        labels=dict(
            x="Probability Range",
            y="Class",
            color="Count"
        ),
        title="Prediction Probability Distribution Heatmap",
        template='plotly_white',
        width=900,
        height=600,
        aspect='auto'
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar_thickness=20,
        coloraxis_colorbar_len=0.6,
        xaxis_tickangle=-45
    )
    fig.update_traces(
        text=heatmap_df.values,
        texttemplate="%{text}",
        textfont={"size": 12}
    )
    return fig

def plot_probability_distribution(df, num_classes):
    """Plot distribution of prediction probabilities"""
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

def plot_feature_correlations(df: pd.DataFrame) -> Figure:
    """Plot correlation matrix of feature probabilities"""
    prob_cols = df.columns[:-2]
    corr_matrix = df[prob_cols].corr()
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
        template='plotly_white'
    )
    return fig

def plot_decision_boundary(df: pd.DataFrame, num_classes: int) -> Optional[Figure]:
    """Plot 2D decision boundary using first two features"""
    if len(df.columns) < 4:
        return None
    features = df.iloc[:, :2] 
    true_class = df.iloc[:, -1]
    x_min, x_max = features.iloc[:, 0].min() - 1, features.iloc[:, 0].max() + 1
    y_min, y_max = features.iloc[:, 1].min() - 1, features.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    fig = px.scatter(
        x=features.iloc[:, 0],
        y=features.iloc[:, 1],
        color=true_class.astype(str),
        title="Decision Boundary (First 2 Features)",
        labels={'x': 'Feature 1', 'y': 'Feature 2', 'color': 'True Class'},
        template='plotly_white'
    )
    return fig

def generate_climate_data(scenario_type: str, num_samples: int, params: dict) -> pd.DataFrame:
    """Generate climate-specific synthetic data"""
    if scenario_type == "Temperature Anomalies":
        years = np.linspace(0, num_samples/12, num_samples)  # Monthly data points
        baseline_temp = params.get('baseline_temp', 15)
        warming_rate = params.get('warming_rate', 0.02)
        noise_level = params.get('noise_level', 0.5)
        
        temps = baseline_temp + warming_rate * years + np.random.normal(0, noise_level, size=years.shape)
        anomalies = temps - baseline_temp
        probs = np.zeros((len(temps), 3))  # Cold, Normal, Hot
        for i, temp in enumerate(temps):
            if temp < baseline_temp - 1:
                probs[i] = [0.7, 0.2, 0.1]
            elif temp > baseline_temp + 1:
                probs[i] = [0.1, 0.2, 0.7]
            else:
                probs[i] = [0.2, 0.6, 0.2]
        
        df = pd.DataFrame(probs, columns=['Cold', 'Normal', 'Hot'])
        df['predicted_class'] = np.argmax(probs, axis=1)
        df['true_class'] = np.digitize(temps, bins=[baseline_temp-1, baseline_temp+1]) - 1
        
        return df

    elif scenario_type == "Extreme Events":
        base_rate = params.get('base_rate', 0.1)
        increase_rate = params.get('increase_rate', 0.005)
        years = np.arange(num_samples)
        events = np.random.exponential(1/(base_rate + increase_rate * years))
        is_extreme = events < np.percentile(events, 90)
        probs = np.zeros((len(events), 2))  # Normal, Extreme
        for i, (event, extreme) in enumerate(zip(events, is_extreme)):
            if extreme:
                probs[i] = [0.3, 0.7]
            else:
                probs[i] = [0.8, 0.2]
        df = pd.DataFrame(probs, columns=['Normal', 'Extreme'])
        df['predicted_class'] = np.argmax(probs, axis=1)
        df['true_class'] = is_extreme.astype(int)
        
        return df

    elif scenario_type == "Sea Level Rise":
        years = np.linspace(0, num_samples/12, num_samples)  # monthly data
        beta = params.get('rise_rate', 3.3)  # mm/year
        seasonal_amp = params.get('seasonal_amplitude', 50)  # mm
        noise_level = params.get('noise_level', 10)  # mm
        trend = beta * years
        seasonal = seasonal_amp * np.sin(2 * np.pi * years)
        noise = np.random.normal(0, noise_level, size=years.shape)
        sea_level = trend + seasonal + noise
        probs = np.zeros((len(sea_level), 3))  # Low, Medium, High
        thresholds = np.percentile(sea_level, [33, 66])
        
        for i, level in enumerate(sea_level):
            if level < thresholds[0]:
                probs[i] = [0.6, 0.3, 0.1]
            elif level > thresholds[1]:
                probs[i] = [0.1, 0.3, 0.6]
            else:
                probs[i] = [0.2, 0.6, 0.2]
        df = pd.DataFrame(probs, columns=['Low', 'Medium', 'High'])
        df['predicted_class'] = np.argmax(probs, axis=1)
        df['true_class'] = np.digitize(sea_level, bins=thresholds)
        return df
    return None

def display_mathematical_formulas():
    st.markdown(r"""
    ### 📐 Mathematical Formulas & Concepts

    #### Why Synthetic Data Generation?
    Synthetic data generation is important for risk-aware assessment because it allows us to:
    1. **Control Data Properties**: We can simulate specific scenarios and edge cases
    2. **Balance Datasets**: Generate data for underrepresented classes
    3. **Privacy Preservation**: Train models without exposing sensitive real data
    4. **Validate Model Behavior**: Test model performance under various conditions
                
    #### Data Generation Process
    1. **Base Distribution**: For each class $c$, samples are drawn from a multivariate normal distribution:
       * $X_c \sim \mathcal{N}(\mu_c, \Sigma_c)$
       * where $\mu_c$ is the mean vector for class $c$
       * $\Sigma_c$ is the covariance matrix

    2. **Noise Addition**: 
       * $X_{noisy} = X + \epsilon$
       * where $\epsilon \sim \mathcal{N}(0, \sigma^2_{noise})$
       * $\sigma^2_{noise}$ controls uncertainty level

    3. **Bias Introduction**:
       * $P(Y=k|X) = \text{softmax}(f(X) + b_k)$
       * where $b_k$ is the bias term for class $k$
       * Simulates systematic measurement errors

    #### Climate Change Applications
    Synthetic data can model various climate-related distributions:

    1. **Temperature Anomalies**:
       * Normal distribution with shifting mean: $T(y) \sim \mathcal{N}(\mu + \alpha y, \sigma^2)$
       * where $y$ is years from baseline
       * $\alpha$ is warming rate (°C/year)
       * Example: $\alpha = 0.02$ °C/year represents global warming trend

    2. **Extreme Weather Events**:
       * Follows Extreme Value Distribution: $P(X \leq x) = \exp(-\exp(-(x-\mu)/\beta))$
       * Parameters adjust for increasing frequency
       * Used for modeling floods, heatwaves, storms

    3. **Sea Level Rise**:
       * Combination of linear trend and local variations
       * $S(t) = \beta t + A\sin(2\pi t) + \epsilon$
       * $\beta$: long-term rise rate
       * $A\sin(2\pi t)$: seasonal variations
       * $\epsilon$: local fluctuations

    #### Example Climate Scenarios
    ```python
    # Temperature Distribution Shift
    def generate_temp_data(years, baseline_temp=15, warming_rate=0.02):
        return baseline_temp + warming_rate * years + np.random.normal(0, 0.5, size=years.shape)

    # Extreme Event Frequency
    def generate_extreme_events(n_years, base_rate=0.1, increase_rate=0.005):
        return np.random.exponential(1/(base_rate + increase_rate * np.arange(n_years)))
    ```

    This synthetic data generation framework allows us to:
    * Test model performance under various climate scenarios
    * Evaluate prediction uncertainty in extreme conditions
    * Assess model reliability for different temporal scales
    * Validate adaptation strategies under different warming pathways
    """)

def main():
    st.title("🐀 RAGatouille: Scenario-based Synthetic Data Generation for Risk-Aware Assessment")
    
    # Add tabs for different sections
    tab_main, tab_math = st.tabs(["Demo", "Mathematical Documentation"])
    
    with tab_math:
        display_mathematical_formulas()
    
    with tab_main:
        st.markdown("""
        Generate synthetic prediction data and test it with the SingularityNET 
        Risk-Aware Assessment service. This demo allows you to experiment with different
        parameters and visualize the results.
        """)
        
        use_split = False
        num_classes = 3
        
        with st.sidebar:
            st.header("📊 Generation Parameters")
            
            # Data type selection
            data_type = st.selectbox(
                "Data Generation Type",
                ["General Synthetic", "Climate Scenarios"]
            )
            
            if data_type == "Climate Scenarios":
                scenario_type = st.selectbox(
                    "Climate Scenario",
                    ["Temperature Anomalies", "Extreme Events", "Sea Level Rise"]
                )
                
                # set num_classes based on scenario type
                if scenario_type == "Extreme Events":
                    num_classes = 2  # normal, extreme
                else:
                    num_classes = 3  # temperature: Cold, Normal, Hot or Sea Level: Low, Medium, High
                
                # Scenario-specific parameters
                with st.expander("Scenario Parameters", expanded=True):
                    if scenario_type == "Temperature Anomalies":
                        baseline_temp = st.slider("Baseline Temperature (°C)", 0.0, 30.0, 15.0)
                        warming_rate = st.slider("Warming Rate (°C/year)", 0.0, 0.1, 0.02)
                        noise_level = st.slider("Temperature Variability", 0.1, 2.0, 0.5)
                        scenario_params = {
                            'baseline_temp': baseline_temp,
                            'warming_rate': warming_rate,
                            'noise_level': noise_level
                        }
                    
                    elif scenario_type == "Extreme Events":
                        base_rate = st.slider("Base Event Rate", 0.01, 0.5, 0.1)
                        increase_rate = st.slider("Rate Increase per Year", 0.0, 0.02, 0.005)
                        scenario_params = {
                            'base_rate': base_rate,
                            'increase_rate': increase_rate
                        }
                    
                    elif scenario_type == "Sea Level Rise":
                        rise_rate = st.slider("Sea Level Rise Rate (mm/year)", 0.0, 10.0, 3.3)
                        seasonal_amp = st.slider("Seasonal Amplitude (mm)", 0.0, 100.0, 50.0)
                        noise_level = st.slider("Local Variability (mm)", 0.0, 50.0, 10.0)
                        scenario_params = {
                            'rise_rate': rise_rate,
                            'seasonal_amplitude': seasonal_amp,
                            'noise_level': noise_level
                        }
                
                num_samples = st.slider("Number of Samples", 100, 10000, 1000)
                
            else:
                with st.expander("Dataset Split", expanded=True):
                    use_split = st.checkbox("Create Train-Test Split")
                    if use_split:
                        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
                        stratify = st.checkbox("Stratified Split", True)
                with st.expander("Distribution Parameters", expanded=True):
                    distribution_type = st.selectbox(
                        "Data Distribution",
                        ["Normal", "Uniform", "Beta", "Gamma"]
                    )
                    if distribution_type == "Normal":
                        mean = st.slider("Mean", -5.0, 5.0, 0.0)
                        std = st.slider("Standard Deviation", 0.1, 5.0, 1.0)
                    elif distribution_type == "Beta":
                        alpha = st.slider("Alpha", 0.1, 10.0, 2.0)
                        beta = st.slider("Beta", 0.1, 10.0, 2.0)
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
        if data_type == "Climate Scenarios":
            df = generate_climate_data(scenario_type, num_samples, scenario_params)
        else:
            df = generate_synthetic_data(
                num_classes,
                num_samples,
                noise_level,
                bias_strength,
                class_imbalance
            )
        current_df = df
        
        if use_split and data_type != "Climate Scenarios":
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df.iloc[:, -1] if stratify else None
            )
            st.success(f"Data split into {len(train_df)} training and {len(test_df)} test samples")
            dataset_view = st.radio("Dataset View", ["Training", "Test"])
            current_df = train_df if dataset_view == "Training" else test_df
        st.header("📊 Data Overview")
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", num_samples)
        
        if data_type == "Climate Scenarios":
            scenario_classes = {
                "Temperature Anomalies": ["Cold", "Normal", "Hot"],
                "Extreme Events": ["Normal", "Extreme"],
                "Sea Level Rise": ["Low", "Medium", "High"]
            }
            col2.metric("Classes", ", ".join(scenario_classes[scenario_type]))
        else:
            col2.metric("Number of Classes", num_classes)
        prob_cols = current_df.columns[:-2]
        if len(prob_cols) > 0:
            avg_prob = current_df[prob_cols].mean().mean()
            col3.metric("Average Probability", f"{avg_prob:.3f}")
        else:
            col3.metric("Average Probability", "N/A")
        st.subheader("Data Visualizations")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Class Distribution",
            "Prediction Heatmap",
            "Probability Distribution",
            "Feature Correlations",
            "Decision Boundary"
        ])
        
        with tab1:
            st.plotly_chart(plot_class_distribution(current_df), use_container_width=True)
        
        with tab2:
            st.plotly_chart(plot_prediction_heatmap(current_df), use_container_width=True)
        
        with tab3:
            prob_dist_plot = plot_probability_distribution(current_df, num_classes)
            if prob_dist_plot:
                st.plotly_chart(prob_dist_plot, use_container_width=True)
        
        with tab4:
            st.plotly_chart(plot_feature_correlations(current_df), use_container_width=True)
        
        with tab5:
            decision_boundary = plot_decision_boundary(current_df, num_classes)
            if decision_boundary:
                st.plotly_chart(decision_boundary, use_container_width=True)
            else:
                st.warning("Decision boundary plot requires at least 2 features")
        with st.expander("🔍 View Raw Data"):
            st.dataframe(current_df, use_container_width=True, height=300)
            csv = current_df.to_csv(index=False)
            st.download_button(
                "📥 Download Dataset",
                csv,
                "synthetic_predictions.csv",
                "text/csv",
                help="Download the complete dataset as CSV"
            )
        st.header("🎯 Risk Assessment")
        if st.button("Run Risk Assessment", type="primary"):
            with st.spinner("Calling Risk Assessment Service..."):
                try:
                    csv_buffer = io.StringIO()
                    current_df.to_csv(csv_buffer, index=False, header=False)
                    csv_string = csv_buffer.getvalue()
                    files = {'file': ('data.csv', csv_string)}
                    response = requests.post(
                        f"http://localhost:{os.getenv('PORT', '5000')}/assess",
                        files=files
                    )
                    response.raise_for_status()
                    result = response.json()
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
                    if result.get("visualization"):
                        st.subheader("Service Visualization")
                        st.image(result["visualization"])
                    
                except Exception as e:
                    st.error(f"Error calling assessment service: {str(e)}")
                    st.error("Please make sure the assessment service is running and accessible.")

if __name__ == "__main__":
    main() 