import streamlit as st
import requests
import json
import pandas as pd
import yfinance as yf

def get_witcher_color_palette():
    """
    Returns a color palette inspired by The Witcher, now with bright red and yellow.
    """
    return {
        'primary': '#ff1744',      # Bright red
        'secondary': '#ffe600',    # Bright yellow
        'accent1': '#ff5252',      # Accent red
        'accent2': '#fff176',      # Accent yellow
        'dark': '#1a1a1a',         # Dark background
        'light': '#fffde7'         # Light text
    }

def load_lottie_url(url: str):
    """
    Loads a Lottie animation file from a URL.
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {e}")
        return None

def get_symbol_suggestions(query):
    """
    Get stock symbol suggestions based on a search query.
    Uses Yahoo Finance search API to find matching stocks.
    """
    try:
        # For a simple implementation, we'll return a list of common stocks
        # that match the query pattern
        common_stocks = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'BRK-B': 'Berkshire Hathaway Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Co.',
            'MA': 'Mastercard Inc.',
            'DIS': 'The Walt Disney Company',
            'HD': 'The Home Depot Inc.',
            'BAC': 'Bank of America Corporation',
            'INTC': 'Intel Corporation',
            'VZ': 'Verizon Communications Inc.',
            'CSCO': 'Cisco Systems Inc.',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'XOM': 'Exxon Mobil Corporation',
            'NKE': 'Nike Inc.',
            'CMCSA': 'Comcast Corporation',
            'PFE': 'Pfizer Inc.',
            'KO': 'The Coca-Cola Company',
            'COST': 'Costco Wholesale Corporation'
        }
        
        # Filter stocks that match the query (case insensitive)
        query = query.upper()
        matching_stocks = {}
        
        for symbol, name in common_stocks.items():
            if query in symbol or query in name.upper():
                matching_stocks[symbol] = name
        
        # Format as "SYMBOL - Company Name"
        return [f"{symbol} - {name}" for symbol, name in matching_stocks.items()]
    
    except Exception as e:
        st.error(f"Error getting symbol suggestions: {e}")
        return []

def display_witcher_quote():
    """
    Returns a random Witcher quote.
    """
    quotes = [
        "Evil is evil. Lesser, greater, middling… Makes no difference. The degree is arbitrary. The definition's blurred. If I'm to choose between one evil and another… I'd rather not choose at all.",
        "People like to invent monsters and monstrosities. Then they seem less monstrous themselves.",
        "If I'm to choose between one evil and another, I'd rather not choose at all.",
        "Destiny helps people believe there's an order to this horseshit. There isn't.",
        "Sometimes there's monsters, sometimes there's money. Rarely both.",
        "Time eats away at memories, distorts them. Sometimes we only remember the good... sometimes only the bad.",
        "The world doesn't need a hero. It needs a professional.",
        "There's never been a shortage of monsters in the world. And there's never been a shortage of money to be made killing them.",
        "When you know about something, it stops being a nightmare. When you know how to fight something, it stops being so threatening.",
        "Mistakes are also important to me. I don't cross them out of my life, or memory. And I never blame others for them."
    ]
    
    import random
    return random.choice(quotes)

def format_csv_data(data):
    """
    Ensures CSV data is in the correct format.
    """
    required_columns = ['date', 'price', 'open', 'high', 'low']
    
    # Check and add any missing required columns
    for col in required_columns:
        if col not in data.columns:
            if col == 'date':
                data['date'] = pd.date_range(start='2020-01-01', periods=len(data))
            else:
                # For numeric columns, if missing, use a similar column or zeros
                similar_cols = {
                    'price': ['close', 'adj close', 'closing_price'],
                    'open': ['opening', 'opening_price'],
                    'high': ['highest', 'max'],
                    'low': ['lowest', 'min']
                }
                
                found = False
                if col in similar_cols:
                    for alt_col in similar_cols[col]:
                        if alt_col in data.columns:
                            data[col] = data[alt_col]
                            found = True
                            break
                
                if not found:
                    data[col] = 0.0
    
    # Ensure date is in datetime format
    if 'date' in data.columns:
        try:
            data['date'] = pd.to_datetime(data['date'])
        except:
            data['date'] = pd.date_range(start='2020-01-01', periods=len(data))
    
    return data
