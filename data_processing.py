import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st

def preprocess_data(data):
    """
    Preprocess financial data:
    - Handle missing values
    - Convert date to datetime
    - Convert string values to numeric
    - Add additional features if not present
    - Remove outliers
    - Normalize/standardize if needed
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The raw financial data
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Convert date to datetime if it's not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Convert string-based numeric values to actual numeric values
    # Handle cases like '32.40K', '1.5M', etc.
    def convert_to_numeric(val):
        if pd.isna(val):
            return val
        
        if isinstance(val, (int, float)):
            return val
            
        if isinstance(val, str):
            val = val.strip().upper()
            if val.endswith('K'):
                try:
                    return float(val[:-1]) * 1000
                except ValueError:
                    return np.nan
            elif val.endswith('M'):
                try:
                    return float(val[:-1]) * 1000000
                except ValueError:
                    return np.nan
            elif val.endswith('B'):
                try:
                    return float(val[:-1]) * 1000000000
                except ValueError:
                    return np.nan
            elif val.endswith('%'):
                try:
                    return float(val[:-1])
                except ValueError:
                    return np.nan
            else:
                try:
                    return float(val.replace(',', ''))
                except ValueError:
                    return np.nan
        return np.nan
    
    # Convert columns with potential string values to numeric
    for col in ['open', 'high', 'low', 'price', 'vol', 'change(%)']:
        if col in df.columns:
            df[col] = df[col].apply(convert_to_numeric)
    
    # Fill missing values in key columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        # Fill missing values with appropriate methods based on column type
        if col in ['open', 'high', 'low', 'price', 'vol']:
            # For price data, forward fill then backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        else:
            # For other numeric columns, use median
            df[col] = df[col].fillna(df[col].median())
    
    # Handle outliers using IQR method for price-related columns
    price_cols = ['open', 'high', 'low', 'price']
    for col in price_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers (using mild approach to avoid excessive trimming)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers instead of removing them
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Ensure 'change(%)' is calculated if not present
    if 'change(%)' not in df.columns and 'price' in df.columns:
        df['change(%)'] = df['price'].pct_change() * 100
    
    # Sort by date if available
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    return df

def feature_engineering(data):
    """
    Create features for financial prediction:
    - Moving averages
    - Technical indicators
    - Date-based features
    - Return trends
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed financial data
    
    Returns:
    --------
    tuple (X, y)
        X: pandas.DataFrame of features
        y: pandas.Series of target values (price)
    """
    df = data.copy()
    
    # Ensure date is datetime for time-based features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract date-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_month'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
    
    # Calculate moving averages if we have price data
    if 'price' in df.columns:
        # Short-term moving averages
        df['MA5'] = df['price'].rolling(window=5).mean()
        df['MA10'] = df['price'].rolling(window=10).mean()
        
        # Medium-term moving average
        df['MA20'] = df['price'].rolling(window=20).mean()
        
        # Long-term moving average
        df['MA50'] = df['price'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['price'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Price momentum (rate of change)
        df['price_momentum'] = df['price'].pct_change(periods=5) * 100
        
        # Volatility (standard deviation over a window)
        df['volatility'] = df['price'].rolling(window=10).std()
    
    # Trading volume features if volume data is available
    if 'vol' in df.columns:
        # Volume moving average
        df['volume_MA5'] = df['vol'].rolling(window=5).mean()
        
        # Volume rate of change
        df['volume_change'] = df['vol'].pct_change() * 100
    
    # Price range features if we have high/low data
    if all(col in df.columns for col in ['high', 'low']):
        # Daily trading range
        df['daily_range'] = df['high'] - df['low']
        
        # Normalized range (range as percentage of opening price)
        if 'open' in df.columns:
            df['normalized_range'] = (df['high'] - df['low']) / df['open'] * 100
    
    # Relative Strength Index (RSI) if we have price data
    if 'price' in df.columns:
        # Calculate price changes
        delta = df['price'].diff()
        
        # Separate gains and losses
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        
        # Calculate average gain and loss over 14 periods
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate relative strength and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands if we have price data
    if 'price' in df.columns:
        # Calculate 20-day moving average
        df['BB_middle'] = df['price'].rolling(window=20).mean()
        
        # Calculate standard deviation
        df['BB_std'] = df['price'].rolling(window=20).std()
        
        # Calculate upper and lower bands
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # Calculate width (indicates volatility)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # Replace infinite values with NaN then fill with appropriate values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Forward fill NaN values created by rolling calculations
    df = df.fillna(method='ffill')
    
    # For any remaining NaN values, fill with median of the column
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Drop the date column for modeling
    if 'date' in df.columns:
        df_features = df.drop('date', axis=1)
    else:
        df_features = df.copy()
    
    # Define target variable (next day's price)
    if 'price' in df.columns:
        y = df['price']
        X = df_features.drop('price', axis=1)
    else:
        # If no price column, use 'close' as target if available
        if 'close' in df.columns:
            y = df['close']
            X = df_features.drop('close', axis=1)
        else:
            # Default to first numeric column if no price/close column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            y = df[numeric_cols[0]]
            X = df_features.drop(numeric_cols[0], axis=1)
    
    # Drop any non-numeric columns that might remain
    X = X.select_dtypes(include=['float64', 'int64'])
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    test_size : float, default=0.2
        Proportion of the data to include in the test split
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
        Split data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
