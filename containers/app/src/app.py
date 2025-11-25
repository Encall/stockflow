import streamlit as st
from config import STOCKS, HORIZON_MAP, DEFAULT_STOCKS
from services import call_prediction_api
from data import load_data, get_actual_price_on_date
from views import (
    display_market_metrics, 
    display_prediction, 
    display_price_chart,
    display_deviation_analysis
)

# Page configuration
st.set_page_config(
    page_title="StockFlow - AI Price Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1f1f2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .prediction-up {
        color: #00ff41;
    }
    .prediction-down {
        color: #ff4136;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN PAGE - Stock Analysis with Prediction
# ============================================================================
st.markdown("# 📈 StockFlow - AI-Powered Stock Analysis")
st.markdown("Real-time stock analysis with AI price predictions")

# Input controls
col1, col2 = st.columns([2, 2])

with col1:
    ticker_input = st.selectbox(
        "📊 Select Stock Symbol",
        options=sorted(STOCKS),
        index=sorted(STOCKS).index("AAPL") if "AAPL" in STOCKS else 0,
        key="main_ticker"
    )

with col2:
    horizon = st.selectbox(
        "⏱️ Time Horizon",
        options=list(HORIZON_MAP.keys()),
        index=5,  # Default to 1 Year
        key="main_horizon"
    )

tickers = [ticker_input.upper()] if ticker_input else []

if not tickers:
    st.info("Pick a stock to analyze", icon=":material/info:")
    st.stop()

# Load stock data
try:
    data = load_data(ticker_input, HORIZON_MAP[horizon])
    if data is None:
        st.stop()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Create two columns: left for metrics, right for chart
metric_col, chart_col = st.columns([1, 2.5])

# ============================================================================
# LEFT COLUMN - Metrics and Predictions
# ============================================================================
with metric_col:
    st.markdown("### 📊 Market Metrics")
    
    # Display market metrics
    current_price = display_market_metrics(data)
    
    st.divider()
    
    # AI Prediction Section
    st.markdown("### 🤖 AI Prediction")
    
    # Automatically fetch prediction
    with st.spinner("Fetching prediction..."):
        pred_data = call_prediction_api(ticker_input)
        
        if pred_data and "error" not in pred_data:
            st.session_state.main_prediction = pred_data
        elif "error" in pred_data:
            st.error(f"❌ {pred_data['error']}")
    
    if "main_prediction" in st.session_state:
        pred = st.session_state.main_prediction
        predicted_price = pred.get("prediction", 0)
        display_prediction(predicted_price, current_price)

# ============================================================================
# RIGHT COLUMN - Chart
# ============================================================================
with chart_col:
    st.markdown("### 📊 Price Chart")
    
    predicted_price = None
    if "main_prediction" in st.session_state:
        predicted_price = st.session_state.main_prediction.get("prediction", 0)
    
    display_price_chart(data, ticker_input, predicted_price)

# ============================================================================
# DEVIATION ANALYSIS SECTION
# ============================================================================
st.divider()

st.header("📊 Deviation Analysis")
st.write("Compare predicted vs actual prices to analyze prediction deviations for a specific date.")
st.warning("⚠️ You can only check up to today's date. Future dates are not available.")

# Input section
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    accuracy_symbol = st.selectbox(
        "Select Stock Symbol",
        options=sorted(STOCKS),
        index=sorted(STOCKS).index("AAPL") if "AAPL" in STOCKS else 0,
        key="accuracy_symbol"
    )

with col2:
    from datetime import datetime, timedelta
    max_date = datetime.now().date()
    min_date = max_date - timedelta(days=365)
    
    check_date = st.date_input(
        "Select Date to Check",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="check_date"
    )

with col3:
    if st.button("🔍 Check Accuracy", key="accuracy_btn", use_container_width=True):
        st.session_state.accuracy_requested = True

if st.session_state.get("accuracy_requested", False):
    with st.spinner(f"Fetching accuracy data for {accuracy_symbol} on {check_date}..."):
        # Get prediction for the specified date
        date_str = check_date.strftime("%Y-%m-%d")
        accuracy_data = call_prediction_api(accuracy_symbol, end_date=date_str)
        
        if accuracy_data and "error" not in accuracy_data:
            st.session_state.last_accuracy = accuracy_data
            st.session_state.last_accuracy_symbol = accuracy_symbol
            st.session_state.last_accuracy_date = check_date
        elif "error" in accuracy_data:
            st.error(f"🔴 {accuracy_data['error']}")
    
    if "last_accuracy" in st.session_state:
        acc = st.session_state.last_accuracy
        symbol = st.session_state.last_accuracy_symbol
        check_date_val = st.session_state.last_accuracy_date
        
        # Get actual price on that date
        actual_price = get_actual_price_on_date(symbol, check_date_val)
        predicted_price = acc.get("prediction", 0)
        
        st.divider()
        
        # Display results
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                "🎯 Predicted Price",
                f"${predicted_price:.2f}",
            )
        
        with result_col2:
            st.metric(
                "📊 Actual Price",
                f"${actual_price:.2f}" if actual_price else "N/A",
            )
        
        if actual_price:
            with result_col3:
                pass  # Space reserved for deviation metric
            
            st.divider()
            display_deviation_analysis(predicted_price, actual_price)
        else:
            st.warning("Could not retrieve actual price data for the selected date. The market may have been closed on that day.")
        
        st.divider()
        
        # Model info
        with st.expander("📋 Model Details"):
            st.write(f"**Run ID:** `{acc.get('run_id', 'N/A')}`")
            st.write(f"**Model URI:** `{acc.get('model_uri', 'N/A')}`")
