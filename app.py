import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import time
import yfinance as yf
import io

from data_processing import preprocess_data, feature_engineering, split_data
from ml_models import train_model, evaluate_model, predict_with_model
from utils import load_lottie_url, get_witcher_color_palette, get_symbol_suggestions

# Page configuration
st.set_page_config(
    page_title="Witcher's Financial Analysis",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# Witcher-themed styling
witcher_colors = get_witcher_color_palette()

# Main header with Witcher theme
st.markdown(
    f"""
    <h1 style='text-align: center; color: {witcher_colors['primary']}; margin-bottom: 0'>
        The Witcher's Financial Analysis
    </h1>
    <h3 style='text-align: center; color: {witcher_colors['secondary']}; font-style: italic; margin-top: 0'>
        "Evil is evil. Lesser, greater, middling… Makes no difference. The degree is arbitrary. 
        The definition's blurred. If I'm to choose between one evil and another… I'd rather not choose at all."
    </h3>
    """, 
    unsafe_allow_html=True
)

# Load Witcher-themed animated GIF (finance related)
lottie_json = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_i9mtrven.json")
st.markdown(
    f"""
    <div style='display: flex; justify-content: center;'>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <lottie-player src="https://assets5.lottiefiles.com/packages/lf20_i9mtrven.json" background="transparent" speed="1" 
        style="width: 300px; height: 300px;" loop autoplay></lottie-player>
    </div>
    """,
    unsafe_allow_html=True
)

# Welcome message
st.markdown(
    f"""
    <div style='background-color: {witcher_colors['dark']}; padding: 15px; border-radius: 5px; margin: 10px 0;'>
        <p style='color: {witcher_colors['light']}; font-size: 16px;'>
            Welcome, traveler, to the Witcher's financial analysis portal. Here, you can harness the power of machine learning 
            to track market patterns and predict future movements - much like tracking a beast through the woods.
        </p>
        <p style='color: {witcher_colors['light']}; font-size: 16px;'>
            Upload your data, track market trends, and let the Signs of machine learning guide your financial hunt.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation and data loading
with st.sidebar:
    st.markdown(f"<h3 style='color: {witcher_colors['primary']}'>The Witcher's Tools</h3>", unsafe_allow_html=True)
    
    # Data source selection
    st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 1: Acquire the Contract</h4>", unsafe_allow_html=True)
    data_source = st.radio("Choose your data source:", ("Upload CSV", "Yahoo Finance API"))
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your financial dataset (CSV)", type=["csv"])
        if uploaded_file is not None and st.button("Load Data", key="load_csv"):
            try:
                data = pd.read_csv(uploaded_file)
                
                # Convert column names to lowercase for case-insensitive matching
                data.columns = [col.lower() for col in data.columns]
                
                # Handle common column name variations
                column_mapping = {
                    'date': 'date',
                    'price': 'price',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'vol.': 'vol',
                    'vol': 'vol',
                    'volume': 'vol',
                    'change%': 'change(%)',
                    'change(%)': 'change(%)',
                    'change %': 'change(%)'
                }
                
                # Rename columns based on mapping
                data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
                
                # Handle date column with variations
                if 'date' not in data.columns:
                    date_variations = ['time', 'datetime', 'day']
                    for var in date_variations:
                        if var in data.columns:
                            data = data.rename(columns={var: 'date'})
                            break
                
                # Add vol if it doesn't exist
                if 'vol' not in data.columns:
                    data['vol'] = np.zeros(len(data))
                    
                # Add change(%) if it doesn't exist
                if 'change(%)' not in data.columns and 'price' in data.columns:
                    data['change(%)'] = data['price'].pct_change() * 100
                
                # Required columns for the application
                required_columns = ['date', 'price', 'open', 'high', 'low']
                
                # Check for required columns
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
                else:
                    st.session_state.data = data
                    st.session_state.data_source = "CSV"
                    st.success("📜 Contract acquired! Data loaded successfully.")
                    st.session_state.current_step = 2
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                
    else:  # Yahoo Finance API
        ticker_suggestion = st.text_input("Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)")
        
        if ticker_suggestion:
            suggestions = get_symbol_suggestions(ticker_suggestion)
            if suggestions:
                selected_ticker = st.selectbox("Select a ticker:", suggestions)
                # Extract just the ticker symbol if it's in format "AAPL - Apple Inc."
                if "-" in selected_ticker:
                    selected_ticker = selected_ticker.split("-")[0].strip()
            else:
                selected_ticker = ticker_suggestion
                
        else:
            selected_ticker = ""
        
        period = st.selectbox("Select time period:", 
                            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], 
                            index=3)
        
        if selected_ticker and st.button("Fetch Data", key="fetch_yf"):
            try:
                with st.spinner("🔮 Casting Quen to protect against market volatility..."):
                    data = yf.download(selected_ticker, period=period)
                    if data.empty:
                        st.error(f"No data found for {selected_ticker}")
                    else:
                        # Rename columns to match expected format
                        data = data.reset_index()
                        data.columns = [x.lower() for x in data.columns]
                        data = data.rename(columns={
                            'adj close': 'price',
                            'volume': 'vol',
                            'date': 'date'
                        })
                        if 'change(%)' not in data.columns:
                            # Calculate daily percentage change
                            data['change(%)'] = data['price'].pct_change() * 100
                        
                        st.session_state.data = data
                        st.session_state.data_source = "Yahoo Finance"
                        st.success(f"📜 Contract acquired! Data for {selected_ticker} loaded successfully.")
                        st.session_state.current_step = 2
                        st.rerun()
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    # Only show further steps if data is loaded
    if st.session_state.data is not None:
        # Preprocessing step
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 2: Prepare the Hunt</h4>", unsafe_allow_html=True)
        if st.button("Preprocess Data", disabled=st.session_state.current_step < 2):
            if st.session_state.data is not None:
                with st.spinner("🔪 Preparing the hunting tools..."):
                    st.session_state.processed_data = preprocess_data(st.session_state.data)
                    st.success("🔪 Preparation complete! Data has been cleaned and processed.")
                    st.session_state.current_step = 3
                    st.rerun()
            else:
                st.error("No data loaded. Please load data first.")
        
        # Feature engineering step
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 3: Track the Beast</h4>", unsafe_allow_html=True)
        if st.button("Engineer Features", disabled=st.session_state.current_step < 3):
            if st.session_state.processed_data is not None:
                with st.spinner("👁️ Using Witcher senses to track market patterns..."):
                    st.session_state.features, st.session_state.target = feature_engineering(st.session_state.processed_data)
                    st.success("👁️ Tracks found! Features engineered successfully.")
                    st.session_state.current_step = 4
                    st.rerun()
            else:
                st.error("Data not preprocessed. Please preprocess first.")
        
        # Train/Test split step
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 4: Plan the Attack</h4>", unsafe_allow_html=True)
        if st.button("Split Data", disabled=st.session_state.current_step < 4):
            if st.session_state.features is not None and st.session_state.target is not None:
                with st.spinner("🗺️ Mapping the hunting grounds..."):
                    X_train, X_test, y_train, y_test = split_data(st.session_state.features, st.session_state.target)
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.success("🗺️ Battle plan ready! Data split into training and testing sets.")
                    st.session_state.current_step = 5
                    st.rerun()
            else:
                st.error("Features not engineered. Please engineer features first.")
        
        # Model selection and training
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 5: Choose Your Weapon</h4>", unsafe_allow_html=True)
        model_options = {
            "Linear Regression": "Like the steel sword - precise and effective for human-like patterns.",
            "Logistic Regression": "Like the silver sword - perfect for binary classification problems.",
            "K-Means Clustering": "Like the Aard sign - groups similar data points together."
        }
        
        model_type = st.selectbox(
            "Select your weapon (model):",
            options=list(model_options.keys()),
            disabled=st.session_state.current_step < 5
        )
        
        st.markdown(f"<p style='font-style: italic; font-size: 13px;'>{model_options[model_type]}</p>", unsafe_allow_html=True)
        
        # Model training
        if st.button("Train Model", disabled=st.session_state.current_step < 5):
            if all(x is not None for x in [st.session_state.X_train, st.session_state.y_train]):
                with st.spinner("⚔️ Forging your weapon..."):
                    model, feature_importance = train_model(
                        X_train=st.session_state.X_train, 
                        y_train=st.session_state.y_train,
                        model_type=model_type
                    )
                    st.session_state.model = model
                    st.session_state.feature_importance = feature_importance
                    st.success("⚔️ Weapon forged! Model trained successfully.")
                    st.session_state.current_step = 6
                    st.rerun()
            else:
                st.error("Data not split. Please split data first.")
        
        # Model evaluation
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 6: Test Your Mettle</h4>", unsafe_allow_html=True)
        if st.button("Evaluate Model", disabled=st.session_state.current_step < 6):
            if st.session_state.model is not None:
                with st.spinner("🧪 Testing the blade's sharpness..."):
                    evaluation_results, predictions = evaluate_model(
                        model=st.session_state.model,
                        X_test=st.session_state.X_test,
                        y_test=st.session_state.y_test,
                        model_type=model_type
                    )
                    st.session_state.evaluation = evaluation_results
                    st.session_state.predictions = predictions
                    st.success("🧪 Weapon tested! Model evaluation complete.")
                    st.session_state.current_step = 7
                    st.rerun()
            else:
                st.error("Model not trained. Please train model first.")
                
        # Results and visualization
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Step 7: Claim Your Reward</h4>", unsafe_allow_html=True)
        if st.button("Show Results", disabled=st.session_state.current_step < 7):
            st.session_state.current_step = 8
            st.rerun()
            
        # Reset application
        if st.button("Start New Analysis", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
# Main content area - Show different content based on the current step
if st.session_state.data is not None:
    # Show data preview
    st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>Data Exploration</h2>", unsafe_allow_html=True)
    
    # Basic data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", f"{len(st.session_state.data):,}")
    with col2:
        st.metric("Time Period", f"{st.session_state.data['date'].min()} to {st.session_state.data['date'].max()}")
    with col3:
        if st.session_state.data_source:
            st.metric("Data Source", st.session_state.data_source)
            
    # Data table with tabs
    tab1, tab2 = st.tabs(["📊 Data Preview", "📈 Price Chart"])
    
    with tab1:
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
    with tab2:
        try:
            df = st.session_state.data.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                fig = px.line(df, x='date', y=['open', 'high', 'low', 'price'], 
                          title='Price History',
                          color_discrete_map={
                              'open': witcher_colors['secondary'],
                              'high': witcher_colors['accent1'],
                              'low': witcher_colors['accent2'],
                              'price': witcher_colors['primary']
                          })
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor=witcher_colors['dark'],
                    paper_bgcolor=witcher_colors['dark'],
                    font=dict(color=witcher_colors['light']),
                    legend_title_text='Metric'
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")

    # Preprocessing results - Step 2 content
    if st.session_state.current_step >= 3 and st.session_state.processed_data is not None:
        st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>Data Preparation</h2>", unsafe_allow_html=True)
        
        # Data preprocessing summary
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values before and after
            st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Missing Values Treatment</h4>", unsafe_allow_html=True)
            missing_before = st.session_state.data.isnull().sum().sum()
            missing_after = st.session_state.processed_data.isnull().sum().sum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Before', 'After'],
                y=[missing_before, missing_after],
                marker_color=[witcher_colors['accent2'], witcher_colors['accent1']]
            ))
            fig.update_layout(
                title='Missing Values',
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Data distribution before/after
            st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Data Distribution</h4>", unsafe_allow_html=True)
            
            numeric_cols = st.session_state.processed_data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for distribution comparison:", numeric_cols)
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=st.session_state.data[selected_col],
                    name='Before',
                    marker_color=witcher_colors['accent2'],
                    opacity=0.7
                ))
                fig.add_trace(go.Histogram(
                    x=st.session_state.processed_data[selected_col],
                    name='After',
                    marker_color=witcher_colors['accent1'],
                    opacity=0.7
                ))
                fig.update_layout(
                    barmode='overlay',
                    title=f'{selected_col} Distribution',
                    template='plotly_dark',
                    plot_bgcolor=witcher_colors['dark'],
                    paper_bgcolor=witcher_colors['dark'],
                    font=dict(color=witcher_colors['light'])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature engineering results - Step 3 content
    if st.session_state.current_step >= 4 and st.session_state.features is not None:
        st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>Feature Engineering</h2>", unsafe_allow_html=True)
        
        # Show engineered features
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Engineered Features</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(st.session_state.features.head(5), use_container_width=True)
            st.markdown(f"**Features shape:** {st.session_state.features.shape}")
            
        with col2:
            # Correlation heatmap
            corr = st.session_state.features.corr()
            fig = px.imshow(
                corr,
                color_continuous_scale=[
                    witcher_colors['dark'],
                    witcher_colors['accent2'],
                    witcher_colors['accent1'],
                    witcher_colors['primary']
                ],
                title="Feature Correlation Matrix"
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Target variable info
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Target Variable</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Target:** Price prediction")
            st.markdown(f"**Target Range:** {st.session_state.target.min():.2f} to {st.session_state.target.max():.2f}")
            st.markdown(f"**Target Mean:** {st.session_state.target.mean():.2f}")
            
        with col2:
            # Target distribution
            fig = px.histogram(
                st.session_state.target,
                title="Target Distribution",
                color_discrete_sequence=[witcher_colors['primary']]
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Train/test split results - Step 4 content
    if st.session_state.current_step >= 5 and st.session_state.X_train is not None:
        st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>Data Split</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Train/test split ratio visualization
            train_size = len(st.session_state.X_train)
            test_size = len(st.session_state.X_test)
            total = train_size + test_size
            
            fig = go.Figure(data=[go.Pie(
                labels=['Training Set', 'Testing Set'],
                values=[train_size, test_size],
                hole=.4,
                marker_colors=[witcher_colors['accent1'], witcher_colors['accent2']]
            )])
            fig.update_layout(
                title=f'Data Split Ratio: {train_size/total:.0%} Train, {test_size/total:.0%} Test',
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Split info
            st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Split Information</h4>", unsafe_allow_html=True)
            st.markdown(f"**Training set shape:** {st.session_state.X_train.shape}")
            st.markdown(f"**Testing set shape:** {st.session_state.X_test.shape}")
            st.markdown(f"**Training target mean:** {st.session_state.y_train.mean():.2f}")
            st.markdown(f"**Testing target mean:** {st.session_state.y_test.mean():.2f}")
            
            # Data distribution similarity check
            st.markdown("##### Distribution Similarity Check")
            train_mean = st.session_state.X_train.mean()
            test_mean = st.session_state.X_test.mean()
            
            # Check if the distributions are similar
            # Calculate similarity score and cap it at 1.0 to avoid errors
            similarity = 1 - (abs(train_mean - test_mean) / train_mean).mean()
            # Ensure similarity is between 0 and 1 for the progress bar
            capped_similarity = min(max(float(similarity), 0.0), 1.0)
            
            st.progress(capped_similarity, text=f"Feature Distribution Similarity: {similarity:.2%}")
            
            if similarity > 0.9:
                st.success("✅ Train and test sets have similar distributions")
            elif similarity > 0.8:
                st.warning("⚠️ Minor distribution differences between sets")
            else:
                st.error("❌ Significant distribution differences between sets")
    
    # Model training results - Step 6 content
    if st.session_state.current_step >= 6 and st.session_state.model is not None:
        st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>Model Training</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model parameters
            st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Model Parameters</h4>", unsafe_allow_html=True)
            
            model_type = type(st.session_state.model).__name__
            st.markdown(f"**Model Type:** {model_type}")
            
            # Display model parameters differently based on model type
            if hasattr(st.session_state.model, 'get_params'):
                params = st.session_state.model.get_params()
                for param, value in params.items():
                    st.markdown(f"**{param}:** {value}")
            else:
                st.markdown("No parameters available for this model.")
            
        with col2:
            # Feature importance if available
            if st.session_state.feature_importance is not None:
                st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Feature Importance</h4>", unsafe_allow_html=True)
                
                importance_df = st.session_state.feature_importance.sort_values(ascending=False)
                
                fig = px.bar(
                    x=importance_df.values,
                    y=importance_df.index,
                    orientation='h',
                    title="Feature Importance",
                    color=importance_df.values,
                    color_continuous_scale=[witcher_colors['accent2'], witcher_colors['primary']]
                )
                fig.update_layout(
                    yaxis_title="Feature",
                    xaxis_title="Importance",
                    template='plotly_dark',
                    plot_bgcolor=witcher_colors['dark'],
                    paper_bgcolor=witcher_colors['dark'],
                    font=dict(color=witcher_colors['light'])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Feature Importance Not Available</h4>", unsafe_allow_html=True)
                st.markdown("This model type doesn't provide feature importance.")
    
    # Model evaluation results - Step 7 content
    if st.session_state.current_step >= 7 and st.session_state.evaluation is not None:
        st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>Model Evaluation</h2>", unsafe_allow_html=True)
        
        # Different metrics based on model type
        model_type = type(st.session_state.model).__name__
        
        if 'regression_metrics' in st.session_state.evaluation:
            # Regression metrics
            metrics = st.session_state.evaluation['regression_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("MSE", f"{metrics['mse']:.4f}")
            with col4:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            
            # Prediction vs Actual plot
            fig = px.scatter(
                x=st.session_state.y_test,
                y=st.session_state.predictions,
                title="Actual vs Predicted Values",
                labels={"x": "Actual", "y": "Predicted"}
            )
            
            # Add perfect prediction line
            min_val = min(st.session_state.y_test.min(), st.session_state.predictions.min())
            max_val = max(st.session_state.y_test.max(), st.session_state.predictions.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color=witcher_colors['accent1'], dash='dash')
                )
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            residuals = st.session_state.y_test - st.session_state.predictions
            
            fig = px.scatter(
                x=st.session_state.predictions,
                y=residuals,
                title="Residuals Plot",
                labels={"x": "Predicted", "y": "Residuals"}
            )
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=witcher_colors['accent1']
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif 'classification_metrics' in st.session_state.evaluation:
            # Classification metrics
            metrics = st.session_state.evaluation['classification_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1']:.4f}")
            
            # Confusion Matrix
            conf_matrix = metrics['confusion_matrix']
            
            fig = px.imshow(
                conf_matrix,
                labels=dict(x="Predicted", y="Actual"),
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                title="Confusion Matrix",
                color_continuous_scale=[
                    witcher_colors['dark'],
                    witcher_colors['accent2'],
                    witcher_colors['accent1'],
                    witcher_colors['primary']
                ]
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor=witcher_colors['dark'],
                paper_bgcolor=witcher_colors['dark'],
                font=dict(color=witcher_colors['light'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            if 'fpr' in metrics and 'tpr' in metrics:
                fig = px.line(
                    x=metrics['fpr'],
                    y=metrics['tpr'],
                    title=f"ROC Curve (AUC = {metrics['auc']:.4f})",
                    labels={"x": "False Positive Rate", "y": "True Positive Rate"}
                )
                
                # Add random guess line
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Random Guess',
                        line=dict(color='gray', dash='dash')
                    )
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor=witcher_colors['dark'],
                    paper_bgcolor=witcher_colors['dark'],
                    font=dict(color=witcher_colors['light'])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif 'clustering_metrics' in st.session_state.evaluation:
            # Clustering metrics
            metrics = st.session_state.evaluation['clustering_metrics']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
            with col2:
                st.metric("Inertia", f"{metrics['inertia']:.2f}")
            
            # Cluster visualization (2D projection using PCA if needed)
            if 'pca_result' in metrics and 'cluster_labels' in metrics:
                pca_result = metrics['pca_result']
                cluster_labels = metrics['cluster_labels']
                
                fig = px.scatter(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    color=cluster_labels,
                    title="Cluster Visualization (PCA 2D Projection)",
                    labels={"x": "Principal Component 1", "y": "Principal Component 2"},
                    color_continuous_scale=[
                        witcher_colors['dark'],
                        witcher_colors['accent2'],
                        witcher_colors['accent1'],
                        witcher_colors['primary']
                    ]
                )
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor=witcher_colors['dark'],
                    paper_bgcolor=witcher_colors['dark'],
                    font=dict(color=witcher_colors['light'])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Final results - Step 8 content
    if st.session_state.current_step >= 8:
        st.markdown(f"<h2 style='color: {witcher_colors['primary']}'>The Hunt's Conclusion</h2>", unsafe_allow_html=True)
        
        # Summary of the entire ML pipeline
        st.markdown(f"<h3 style='color: {witcher_colors['secondary']}'>The Witcher's Contract Complete</h3>", unsafe_allow_html=True)
        
        # Success message with Witcher quote
        st.success(
            """
            "Evil is evil. Lesser, greater, middling… Makes no difference. The degree is arbitrary. 
            The definition's blurred. If I'm to choose between one model and another… I'd rather not choose at all."
            
            But you chose wisely, and your hunt was successful!
            """
        )
        
        # Download options
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>Claim Your Reward (Download Results)</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download predictions
            if st.session_state.predictions is not None:
                predictions_df = pd.DataFrame({
                    'Actual': st.session_state.y_test,
                    'Predicted': st.session_state.predictions
                })
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="witcher_predictions.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Download model evaluation report
            if st.session_state.evaluation is not None:
                model_type = type(st.session_state.model).__name__
                
                report = f"# The Witcher's Financial Analysis Report\n\n"
                report += f"## Model Type: {model_type}\n\n"
                
                if 'regression_metrics' in st.session_state.evaluation:
                    metrics = st.session_state.evaluation['regression_metrics']
                    report += "## Regression Metrics\n\n"
                    report += f"- R² Score: {metrics['r2']:.4f}\n"
                    report += f"- Mean Absolute Error: {metrics['mae']:.4f}\n"
                    report += f"- Mean Squared Error: {metrics['mse']:.4f}\n"
                    report += f"- Root Mean Squared Error: {metrics['rmse']:.4f}\n"
                
                elif 'classification_metrics' in st.session_state.evaluation:
                    metrics = st.session_state.evaluation['classification_metrics']
                    report += "## Classification Metrics\n\n"
                    report += f"- Accuracy: {metrics['accuracy']:.4f}\n"
                    report += f"- Precision: {metrics['precision']:.4f}\n"
                    report += f"- Recall: {metrics['recall']:.4f}\n"
                    report += f"- F1 Score: {metrics['f1']:.4f}\n"
                    report += f"- AUC: {metrics.get('auc', 'N/A')}\n"
                
                elif 'clustering_metrics' in st.session_state.evaluation:
                    metrics = st.session_state.evaluation['clustering_metrics']
                    report += "## Clustering Metrics\n\n"
                    report += f"- Silhouette Score: {metrics['silhouette']:.4f}\n"
                    report += f"- Inertia: {metrics['inertia']:.2f}\n"
                    report += f"- Number of Clusters: {metrics['n_clusters']}\n"
                
                if st.session_state.feature_importance is not None:
                    report += "\n## Feature Importance\n\n"
                    for feature, importance in st.session_state.feature_importance.items():
                        report += f"- {feature}: {importance:.4f}\n"
                
                report += "\n## Witcher's Wisdom\n\n"
                report += "> People like to invent monsters and monstrosities. Then they seem less monstrous themselves.\n"
                report += "> When they get blind-drunk, cheat, steal, beat their wives, starve an old woman, when they kill a trapped fox with an axe or asphyxiate an ailing grandmother, they like to think that the Bane entering cottages at daybreak is more monstrous than they are.\n"
                
                st.download_button(
                    label="📜 Download Full Report",
                    data=report,
                    file_name="witcher_analysis_report.md",
                    mime="text/markdown"
                )
        
        # Final visualization of the entire pipeline
        st.markdown(f"<h4 style='color: {witcher_colors['secondary']}'>The Path You've Traveled</h4>", unsafe_allow_html=True)
        
        # Create a pipeline visualization
        pipeline_steps = [
            {"name": "Data Acquisition", "desc": "Loaded data from source", "complete": True},
            {"name": "Data Preprocessing", "desc": "Cleaned and prepared data", "complete": True},
            {"name": "Feature Engineering", "desc": "Created and selected features", "complete": True},
            {"name": "Data Split", "desc": "Split into training and testing sets", "complete": True},
            {"name": "Model Training", "desc": f"Trained {type(st.session_state.model).__name__}", "complete": True},
            {"name": "Model Evaluation", "desc": "Tested model performance", "complete": True},
            {"name": "Results", "desc": "Analyzed findings", "complete": True}
        ]
        
        # Create a horizontal timeline
        st.markdown(
            """
            <style>
            .pipeline-container {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
                position: relative;
            }
            .pipeline-step {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 14%;
                position: relative;
                z-index: 2;
            }
            .step-icon {
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .step-name {
                font-weight: bold;
                text-align: center;
                margin-bottom: 5px;
            }
            .step-desc {
                font-size: 12px;
                text-align: center;
            }
            .pipeline-line {
                position: absolute;
                top: 25px;
                left: 0;
                right: 0;
                height: 4px;
                z-index: 1;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Create the pipeline HTML
        pipeline_html = '<div class="pipeline-container">'
        
        # Add the connecting line
        pipeline_html += f'<div class="pipeline-line" style="background-color: {witcher_colors["primary"]};"></div>'
        
        # Add each step
        for i, step in enumerate(pipeline_steps):
            bg_color = witcher_colors["primary"] if step["complete"] else witcher_colors["dark"]
            text_color = witcher_colors["light"] if step["complete"] else witcher_colors["accent2"]
            
            pipeline_html += f'''
            <div class="pipeline-step">
                <div class="step-icon" style="background-color: {bg_color}; color: {text_color};">{i+1}</div>
                <div class="step-name" style="color: {witcher_colors['secondary']};">{step["name"]}</div>
                <div class="step-desc" style="color: {witcher_colors['light']};">{step["desc"]}</div>
            </div>
            '''
        
        pipeline_html += '</div>'
        
        st.markdown(pipeline_html, unsafe_allow_html=True)
        
        # Witcher sign-off
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 50px; padding: 20px; background-color: {witcher_colors['dark']}; border-radius: 5px;'>
                <p style='color: {witcher_colors['light']}; font-style: italic;'>
                    "If I'm to choose between one evil and another, I'd rather not choose at all."<br>
                    - Geralt of Rivia
                </p>
                <p style='color: {witcher_colors['accent1']}; margin-top: 15px;'>
                    Thank you for using The Witcher's Financial Analysis Tool
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# If no data is loaded, show instructions
else:
    st.markdown(
        f"""
        <div style='background-color: {witcher_colors['dark']}; padding: 20px; border-radius: 5px;'>
            <h3 style='color: {witcher_colors['secondary']}'>Instructions</h3>
            <p style='color: {witcher_colors['light']}; margin-bottom: 10px;'>
                To begin your financial analysis journey with the Witcher:
            </p>
            <ol style='color: {witcher_colors['light']}'>
                <li>Select a data source from the sidebar (Upload CSV or Yahoo Finance API)</li>
                <li>Load your financial data</li>
                <li>Follow the step-by-step guidance through the machine learning pipeline</li>
                <li>Analyze the results and download your findings</li>
            </ol>
            <p style='color: {witcher_colors['accent1']}; font-style: italic; margin-top: 15px;'>
                "Destiny helps people believe there's an order to this horseshit. There isn't."
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add a decorative Witcher symbol
    st.markdown(
        """
        <div style='display: flex; justify-content: center; margin-top: 30px;'>
            <svg width="200" height="200" viewBox="0 0 100 100">
                <path d="M50,5 L95,50 L50,95 L5,50 Z" fill="none" stroke="#b80e0e" stroke-width="2"/>
                <path d="M50,15 L85,50 L50,85 L15,50 Z" fill="none" stroke="#b80e0e" stroke-width="2"/>
                <path d="M50,25 L75,50 L50,75 L25,50 Z" fill="none" stroke="#b80e0e" stroke-width="2"/>
                <circle cx="50" cy="50" r="5" fill="#b80e0e"/>
                <path d="M50,15 L50,5" stroke="#b80e0e" stroke-width="2"/>
                <path d="M50,85 L50,95" stroke="#b80e0e" stroke-width="2"/>
                <path d="M15,50 L5,50" stroke="#b80e0e" stroke-width="2"/>
                <path d="M85,50 L95,50" stroke="#b80e0e" stroke-width="2"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True
    )
