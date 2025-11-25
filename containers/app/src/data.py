import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

def get_current_price(symbol: str):
    """Get current/today's stock price."""
    try:
        ticker = yf.Ticker(symbol)
        today = datetime.now().date()
        # Try to get today's data
        hist = ticker.history(start=today, end=today + timedelta(days=1))
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        # Fallback to previous close
        return float(ticker.info.get('currentPrice', 0))
    except Exception as e:
        st.warning(f"Could not fetch current price for {symbol}: {str(e)}")
        return None

def load_data(ticker, period):
    """Load stock data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period)
        if data is None or data.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def get_actual_price_on_date(symbol: str, target_date):
    """Get actual price on a specific date."""
    try:
        ticker = yf.Ticker(symbol)
        next_day = target_date + timedelta(days=1)
        hist = ticker.history(start=target_date, end=next_day + timedelta(days=1))
        
        if not hist.empty:
            if target_date.strftime("%Y-%m-%d") in [d.strftime("%Y-%m-%d") for d in hist.index]:
                actual_price = float(hist.loc[hist.index.date == target_date, 'Close'].iloc[-1])
                return actual_price
        return None
    except Exception as e:
        st.warning(f"Could not fetch actual price for {target_date}: {str(e)}")
        return None
