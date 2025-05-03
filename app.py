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
import os

from data_processing import preprocess_data, feature_engineering, split_data
from ml_models import train_model, evaluate_model, predict_with_model
from utils import load_lottie_url, get_witcher_color_palette, get_symbol_suggestions, display_witcher_quote

# Page configuration
st.set_page_config(
    page_title="Witcher's Financial Analysis",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Loading screen with Witcher logo
st.markdown("""
<style>
.loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #1a1a1a;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    animation: fadeOut 2s ease-in-out forwards;
    animation-delay: 2s;
}

.witcher-logo {
    width: 200px;
    height: 200px;
    background: url('https://cdn-icons-png.flaticon.com/512/2504/2504123.png') center/contain no-repeat;
    animation: logoPulse 2s ease-in-out infinite;
    filter: drop-shadow(0 0 20px #ff1744);
}

.loading-text {
    position: absolute;
    bottom: 30%;
    color: #ffe600;
    font-family: 'Cinzel', serif;
    font-size: 24px;
    text-shadow: 0 0 10px #ff1744;
    animation: textGlow 2s ease-in-out infinite;
}

@keyframes logoPulse {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 20px #ff1744); }
    50% { transform: scale(1.1); filter: drop-shadow(0 0 30px #ffe600); }
}

@keyframes textGlow {
    0%, 100% { text-shadow: 0 0 10px #ff1744; }
    50% { text-shadow: 0 0 20px #ffe600; }
}

@keyframes fadeOut {
    0% { opacity: 1; }
    100% { opacity: 0; visibility: hidden; }
}

/* Add Google Font for Cinzel */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');
</style>

<div class="loading-screen">
    <div class="witcher-logo"></div>
    <div class="loading-text">Loading The Witcher's Financial Analysis...</div>
</div>
""", unsafe_allow_html=True)

# Add a small delay to show the loading screen
import time
time.sleep(2)

# Apply custom CSS for Witcher themed styling
def load_css():
    """Load custom CSS to make the app more visually appealing"""
    css_file = os.path.join("assets", "custom.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file doesn't exist
        st.markdown("""
        <style>
        /* Main app container */
        .stApp {
            position: relative;
            min-height: 100vh;
            background: #1a1a1a;
            overflow: hidden;
        }

        /* Main background with Witcher theme */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -2;
            background: url('https://wallpapercave.com/wp/wp5784740.jpg') center center/cover;
            animation: bgPulse 20s ease-in-out infinite;
            opacity: 0.7;
        }

        /* Animated overlay with Witcher medallion pattern */
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            background: 
                radial-gradient(circle at 50% 50%, rgba(255, 23, 68, 0.1) 0%, transparent 50%),
                repeating-linear-gradient(45deg, rgba(255, 230, 0, 0.05) 0px, rgba(255, 230, 0, 0.05) 1px, transparent 1px, transparent 10px);
            animation: overlayPulse 10s ease-in-out infinite;
        }

        /* Floating Witcher medallions */
        .witcher-medallion {
            position: fixed;
            width: 100px;
            height: 100px;
            background: url('https://cdn-icons-png.flaticon.com/512/2504/2504123.png') center/contain no-repeat;
            opacity: 0.1;
            z-index: -1;
            animation: floatMedallion 15s ease-in-out infinite;
        }

        .medallion-1 {
            top: 10%;
            left: 10%;
            animation-delay: 0s;
        }

        .medallion-2 {
            top: 20%;
            right: 15%;
            animation-delay: -5s;
        }

        .medallion-3 {
            bottom: 15%;
            left: 20%;
            animation-delay: -10s;
        }

        /* Animations */
        @keyframes bgPulse {
            0%, 100% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }

        @keyframes overlayPulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 0.7; }
        }

        @keyframes floatMedallion {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(10px, -10px) rotate(5deg); }
            50% { transform: translate(0, -20px) rotate(0deg); }
            75% { transform: translate(-10px, -10px) rotate(-5deg); }
        }

        /* Content styling */
        .main-content {
            position: relative;
            z-index: 1;
            background: rgba(26, 26, 26, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin: 20px;
            box-shadow: 0 0 20px rgba(255, 23, 68, 0.3);
        }

        /* Enhanced text styling */
        h1 {
            font-family: 'Cinzel', serif !important;
            text-shadow: 2px 2px 8px #ffe600, 0 0 8px #ff1744;
            border-bottom: 2px solid #ff1744;
            padding-bottom: 8px;
            color: #ff1744 !important;
            position: relative;
        }

        h1::after {
            content: "";
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, #ff1744, #ffe600, #ff1744);
            animation: borderGlow 2s linear infinite;
        }

        @keyframes borderGlow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Button styling */
        .stButton>button {
            background-color: #ff1744 !important;
            color: #ffe600 !important;
            border: 2px solid #ffe600 !important;
            border-radius: 5px !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px #ffe60033 !important;
            position: relative;
            overflow: hidden;
        }

        .stButton>button::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255, 230, 0, 0.3), transparent);
            transform: rotate(45deg);
            animation: buttonShine 3s linear infinite;
        }

        @keyframes buttonShine {
            0% { transform: translateX(-100%) rotate(45deg); }
            100% { transform: translateX(100%) rotate(45deg); }
        }

        .stButton>button:hover {
            background-color: #ffe600 !important;
            color: #ff1744 !important;
            border-color: #ff1744 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 16px #ff174433 !important;
        }

        /* Text styling */
        p, h2, h3, h4, h5, h6, label, .stMarkdown {
            color: #ffe600 !important;
            text-shadow: 0 0 4px #ff1744;
        }

        /* Input styling */
        .stSelectbox, .stTextInput {
            background-color: rgba(255, 241, 118, 0.1) !important;
            color: #ffe600 !important;
            border: 1px solid #ff1744 !important;
            border-radius: 5px !important;
            padding: 5px !important;
        }

        .stSelectbox:hover, .stTextInput:hover {
            border-color: #ffe600 !important;
            box-shadow: 0 0 10px rgba(255, 230, 0, 0.3) !important;
        }
        </style>

        <!-- Add floating medallions -->
        <div class="witcher-medallion medallion-1"></div>
        <div class="witcher-medallion medallion-2"></div>
        <div class="witcher-medallion medallion-3"></div>
        """, unsafe_allow_html=True)
    
    # Add Google Font for Cinzel
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

load_css()

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

# Main header with enhanced Witcher theme
st.markdown(f"""
<style>
.witcher-title {{
    font-family: 'Cinzel', serif !important;
    font-size: 3.5rem !important;
    text-align: center;
    background: linear-gradient(45deg, #ff1744, #ffe600, #ff1744);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s linear infinite;
    text-shadow: 0 0 15px rgba(255, 204, 0, 0.5);
    margin-bottom: 0.5rem;
    position: relative;
    letter-spacing: 2px;
}}

.medallion-large {{
    width: 80px;
    height: 80px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/2504/2504123.png');
    background-size: contain;
    background-repeat: no-repeat;
    position: absolute;
    left: 50%;
    transform: translateX(-50%) translateY(-100%);
    animation: medalPulse 2s ease-in-out infinite;
    filter: drop-shadow(0 0 20px #ff1744);
}}

@keyframes shine {{
    0% {{ background-position: 0% center; }}
    100% {{ background-position: 200% center; }}
}}

@keyframes medalPulse {{
    0% {{
        transform: translateX(-50%) translateY(-100%) scale(1);
        filter: drop-shadow(0 0 20px #ff1744);
    }}
    50% {{
        transform: translateX(-50%) translateY(-100%) scale(1.2);
        filter: drop-shadow(0 0 30px #ffe600);
    }}
    100% {{
        transform: translateX(-50%) translateY(-100%) scale(1);
        filter: drop-shadow(0 0 20px #ff1744);
    }}
}}

.quote-container {{
    text-align: center;
    font-style: italic;
    color: {witcher_colors['secondary']};
    margin-top: 0;
    margin-bottom: 5px;
    padding: 5px 10px;
    font-size: 1.2rem;
    position: relative;
    text-shadow: 0 0 10px #ff1744;
    animation: quoteGlow 2s ease-in-out infinite;
}}

@keyframes quoteGlow {{
    0%, 100% {{ text-shadow: 0 0 10px #ff1744; }}
    50% {{ text-shadow: 0 0 20px #ffe600; }}
}}
</style>

<div class="medallion-container">
    <div class="medallion-large"></div>
</div>
<h1 class="witcher-title">The Witcher's Financial Analysis</h1>
<div class="quote-container">
    {display_witcher_quote()}
</div>
""", unsafe_allow_html=True)

# Load Witcher-themed animated GIF (finance related) with reduced size
lottie_json = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_i9mtrven.json")
st.markdown(
    f"""
    <div style='display: flex; justify-content: center; margin-bottom: 5px; margin-top: 0;'>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <lottie-player src="https://assets5.lottiefiles.com/packages/lf20_i9mtrven.json" background="transparent" speed="1" 
        style="width: 200px; height: 200px;" loop autoplay></lottie-player>
    </div>
    """,
    unsafe_allow_html=True
)

# Welcome message with minimal spacing
st.markdown(
    f"""
    <div style='background-color: {witcher_colors['dark']}; padding: 15px; border-radius: 5px; margin-top: 0;'>
        <p style='color: {witcher_colors['light']}; font-size: 16px; margin-bottom: 10px;'>
            Welcome, traveler, to the Witcher's financial analysis portal. Here, you can harness the power of machine learning 
            to track market patterns and predict future movements - much like tracking a beast through the woods.
        </p>
        <p style='color: {witcher_colors['light']}; font-size: 16px; margin-top: 0; margin-bottom: 0;'>
            Upload your data, track market trends, and let the Signs of machine learning guide your financial hunt.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation and data loading
with st.sidebar:
    st.markdown(f"""
    <style>
    .sidebar-title {{
        font-family: 'Cinzel', serif !important;
        font-size: 1.8rem !important;
        background: linear-gradient(45deg, {witcher_colors['primary']}, {witcher_colors['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 5px rgba(255, 204, 0, 0.3);
        margin: 0 0 15px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid {witcher_colors['primary']};
        position: relative;
        animation: pulse 2s infinite alternate;
    }}
    
    @keyframes pulse {{
        0% {{text-shadow: 0 0 5px rgba(255, 204, 0, 0.3);}}
        100% {{text-shadow: 0 0 15px rgba(255, 204, 0, 0.6);}}
    }}
    
    .sidebar-icon {{
        width: 24px;
        height: 24px;
        display: inline-block;
        margin-right: 8px;
        vertical-align: middle;
        background-image: url('https://cdn-icons-png.flaticon.com/512/2504/2504123.png');
        background-size: contain;
        background-repeat: no-repeat;
    }}
    </style>
    
    <h3 class="sidebar-title"><span class="sidebar-icon"></span>The Witcher's Tools</h3>
    """, unsafe_allow_html=True)
    
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
                    # Download data with proper error handling
                    ticker = yf.Ticker(selected_ticker)
                    data = ticker.history(period=period)
                    
                    if data.empty:
                        st.error(f"No data found for {selected_ticker}")
                    else:
                        # Reset index to make date a column
                        data = data.reset_index()
                        
                        # Rename columns to match expected format
                        column_mapping = {
                            'Date': 'date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'price',
                            'Adj Close': 'price',
                            'Volume': 'vol'
                        }
                        
                        # Rename columns that exist in the data
                        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
                        
                        # Ensure all required columns exist
                        required_columns = ['date', 'price', 'open', 'high', 'low']
                        missing_cols = [col for col in required_columns if col not in data.columns]
                        
                        if missing_cols:
                            st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        else:
                            # Add volume if it doesn't exist
                            if 'vol' not in data.columns:
                                data['vol'] = 0
                            
                            # Calculate percentage change
                            data['change(%)'] = data['price'].pct_change() * 100
                            
                            # Convert date to datetime if it's not already
                            data['date'] = pd.to_datetime(data['date'])
                            
                            # Store the data
                            st.session_state.data = data
                            st.session_state.data_source = "Yahoo Finance"
                            st.success(f"📜 Contract acquired! Data for {selected_ticker} loaded successfully.")
                            st.session_state.current_step = 2
                            st.rerun()
            except Exception as e:
                error_message = str(e)
                if "too many requests" in error_message.lower():
                    st.error("Error fetching data from API due to too many requests. Please wait a moment and try again.")
                elif "symbol may be delisted" in error_message.lower():
                    st.error(f"Error: The symbol {selected_ticker} may be delisted or invalid. Please check the symbol and try again.")
                else:
                    st.error(f"Error fetching data: {error_message}")
    
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
            
    # Data table with tabs - using a more organized layout
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Price Chart", "📊 Data Statistics"])
    
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
                    plot_bgcolor='rgba(45, 45, 45, 0.7)',
                    paper_bgcolor='rgba(45, 45, 45, 0.4)',
                    font=dict(color=witcher_colors['light']),
                    legend_title_text='Metric',
                    title_font=dict(color=witcher_colors['secondary'], size=18),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            
    with tab3:
        # Display descriptive statistics
        st.markdown("<h4>Descriptive Statistics</h4>", unsafe_allow_html=True)
        st.dataframe(st.session_state.data.describe(), use_container_width=True)
        
        # Add a volume histogram if volume exists
        if 'vol' in st.session_state.data.columns:
            fig = px.histogram(
                st.session_state.data, 
                x='vol',
                title="Volume Distribution",
                template="plotly_dark",
                nbins=30
            )
            fig.update_layout(
                plot_bgcolor='rgba(45, 45, 45, 0.7)',
                paper_bgcolor='rgba(45, 45, 45, 0.4)',
                font=dict(color=witcher_colors['light']),
                title_font=dict(color=witcher_colors['secondary'], size=16),
                height=350
            )
            fig.update_traces(marker_color=witcher_colors['accent1'])
            st.plotly_chart(fig, use_container_width=True)

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
        
        # Show engineered features table first
        st.dataframe(st.session_state.features.head(5), use_container_width=True)
        st.markdown(f"**Features shape:** {st.session_state.features.shape}")
        
        # Enhanced correlation matrix below feature table with a small vertical spacing
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #c9a66b;'>Feature Correlation Matrix</h4>", unsafe_allow_html=True)
        
        # Generate a more readable heatmap
        corr = st.session_state.features.corr()
        fig = px.imshow(
            corr,
            color_continuous_scale=[
                witcher_colors['dark'],
                witcher_colors['accent2'],
                witcher_colors['accent1'],
                witcher_colors['primary']
            ],
            zmin=-1, zmax=1, # Fixed range for better color contrast
            height=500, # Taller heatmap for better visibility
            title=None # Remove title as we have an h4 heading now
        )
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(45, 45, 45, 0.7)',
            paper_bgcolor='rgba(45, 45, 45, 0.4)',
            font=dict(color=witcher_colors['light']),
            margin=dict(l=40, r=40, t=10, b=40) # Reduced top margin since we have a separate heading
        )
        # Make the text annotations more visible
        fig.update_traces(
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(color="white", size=10)
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
        
        # Display ML Pipeline journey visually in a clean way
        # Create a pipeline visualization with structured code rather than raw HTML
        pipeline_steps = [
            {"name": "Data Acquisition", "desc": "Loaded data from source", "icon": "1"},
            {"name": "Data Preprocessing", "desc": "Cleaned and prepared data", "icon": "2"},
            {"name": "Feature Engineering", "desc": "Created and selected features", "icon": "3"},
            {"name": "Data Split", "desc": "Split into training and testing sets", "icon": "4"},
            {"name": "Model Training", "desc": f"Trained model on data", "icon": "5"},
            {"name": "Model Evaluation", "desc": "Tested model performance", "icon": "6"},
            {"name": "Results", "desc": "Analyzed findings", "icon": "7"}
        ]
        
        st.markdown("<h4>The Path You've Traveled</h4>", unsafe_allow_html=True)
        
        # Create a clean container for the steps
        cols = st.columns(len(pipeline_steps))
        
        # Add each step in its own column
        for i, (col, step) in enumerate(zip(cols, pipeline_steps)):
            with col:
                st.markdown(
                    f"""
                    <div style="text-align: center; background-color: #1a1a1a; border-radius: 10px; padding: 15px; height: 100%;">
                        <div style="width: 50px; height: 50px; border-radius: 50%; background-color: #ff1744; color: #ffe600; 
                                display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto;
                                font-size: 20px; font-weight: bold;">
                            {step["icon"]}
                        </div>
                        <div style="font-weight: bold; color: #c9a66b; margin-bottom: 5px;">
                            {step["name"]}
                        </div>
                        <div style="font-size: 12px; color: #e8e8e8;">
                            {step["desc"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Success message with Witcher quote
        witcher_quote = display_witcher_quote()
        st.success(
            f"""
            "{witcher_quote}"
            
            Your financial hunt was successful! The patterns have been tracked, and the beast tamed.
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

# Custom CSS for enhanced UI elements
st.markdown("""
<style>
/* Text animations */
h1, h2, h3, h4, h5, h6 {
    animation: textGlow 2s ease-in-out infinite !important;
}

@keyframes textGlow {
    0%, 100% { text-shadow: 0 0 10px rgba(255, 23, 68, 0.5); }
    50% { text-shadow: 0 0 20px rgba(255, 230, 0, 0.7); }
}

/* Graph animations and styling */
[data-testid="stPlotlyChart"] {
    animation: graphFadeIn 1s ease-out !important;
    transition: all 0.3s ease !important;
}

[data-testid="stPlotlyChart"]:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 30px rgba(255, 230, 0, 0.5) !important;
}

@keyframes graphFadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Metric animations */
[data-testid="stMetricValue"] {
    animation: metricPulse 2s ease-in-out infinite !important;
}

[data-testid="stMetricLabel"] {
    animation: labelGlow 2s ease-in-out infinite !important;
}

@keyframes metricPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

@keyframes labelGlow {
    0%, 100% { text-shadow: 0 0 5px rgba(255, 23, 68, 0.3); }
    50% { text-shadow: 0 0 15px rgba(255, 230, 0, 0.5); }
}

/* Table animations */
[data-testid="stDataFrame"] {
    animation: tableFadeIn 1s ease-out !important;
    transition: all 0.3s ease !important;
}

[data-testid="stDataFrame"]:hover {
    box-shadow: 0 0 20px rgba(255, 23, 68, 0.3) !important;
}

@keyframes tableFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Tab animations */
[data-testid="stTabs"] {
    animation: tabsFadeIn 0.5s ease-out !important;
}

@keyframes tabsFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Success message animations */
.element-container:has(.stSuccess) {
    animation: successPulse 2s ease-in-out infinite !important;
}

@keyframes successPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

/* Error message animations */
.element-container:has(.stError) {
    animation: errorShake 0.5s ease-in-out !important;
}

@keyframes errorShake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

/* Chart tooltip animations */
.js-plotly-plot .plotly .hoverlayer .hover {
    animation: tooltipFade 0.3s ease-out !important;
}

@keyframes tooltipFade {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

/* Loading spinner animations */
.stSpinner {
    animation: spinnerPulse 1.5s ease-in-out infinite !important;
}

@keyframes spinnerPulse {
    0%, 100% { transform: scale(1); opacity: 0.7; }
    50% { transform: scale(1.1); opacity: 1; }
}

/* Progress bar animations */
.stProgress > div > div {
    background: linear-gradient(45deg, #ff1744, #ffe600) !important;
    background-size: 200% auto !important;
    animation: progressShine 2s linear infinite !important;
}

@keyframes progressShine {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* Quote container animations */
.quote-container {
    animation: quoteFloat 3s ease-in-out infinite !important;
}

@keyframes quoteFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

/* Keep existing button and navbar styles */
.stButton > button {
    background: linear-gradient(45deg, #ff1744, #ffe600);
    background-size: 200% auto;
    color: #1a1a1a !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px rgba(255, 23, 68, 0.5) !important;
    animation: buttonShine 3s linear infinite, buttonBlink 2s ease-in-out infinite !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 230, 0, 0.4),
        transparent
    );
    animation: buttonGlow 2s linear infinite;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 25px rgba(255, 230, 0, 0.7) !important;
    background-position: right center !important;
    animation: buttonShine 1.5s linear infinite, buttonBlink 1s ease-in-out infinite !important;
}

@keyframes buttonShine {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

@keyframes buttonBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

@keyframes buttonGlow {
    0% { left: -100%; }
    100% { left: 100%; }
}
</style>
""", unsafe_allow_html=True)
