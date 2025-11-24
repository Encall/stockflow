import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Stock peer analysis dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Stock peer analysis

Easily compare stocks against others in their peer group.
"""

""  # Add some space.

cols = st.columns([1, 3])
# Will declare right cell later to avoid showing it when no data.

STOCKS = [
    "AAPL",
    "ABBV",
    "ACN",
    "ADBE",
    "ADP",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "APD",
    "AVGO",
    "AXP",
    "BA",
    "BK",
    "BKNG",
    "BMY",
    "BRK.B",
    "BSX",
    "C",
    "CAT",
    "CI",
    "CL",
    "CMCSA",
    "COST",
    "CRM",
    "CSCO",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "ELV",
    "EOG",
    "EQR",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GOOG",
    "GOOGL",
    "HD",
    "HON",
    "HUM",
    "IBM",
    "ICE",
    "INTC",
    "ISRG",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDLZ",
    "META",
    "MMC",
    "MO",
    "MRK",
    "MSFT",
    "NEE",
    "NFLX",
    "NKE",
    "NOW",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PLD",
    "PM",
    "PSA",
    "REGN",
    "RTX",
    "SBUX",
    "SCHW",
    "SLB",
    "SO",
    "SPGI",
    "T",
    "TJX",
    "TMO",
    "TSLA",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "V",
    "VZ",
    "WFC",
    "WM",
    "WMT",
    "XOM",
]

DEFAULT_STOCKS = ["AAPL"]


def stocks_to_str(stocks):
    return ",".join(stocks)


if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", stocks_to_str(DEFAULT_STOCKS)
    ).split(",")


# Callback to update query param when input changes
def update_query_param():
    if st.session_state.tickers_input:
        st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)
    else:
        st.query_params.pop("stocks", None)


top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Selectbox for single stock ticker
    ticker_input = st.selectbox(
        "Stock ticker",
        options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
        index=0 if st.session_state.tickers_input else None,
        placeholder="Choose a stock. Example: NVDA",
    )
    tickers = [ticker_input] if ticker_input else []

# Time horizon selector
horizon_map = {
    "1 Months": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}

with top_left_cell:
    # Buttons for picking time horizon
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="6 Months",
    )

tickers = [t.upper() for t in tickers]

# Update query param when text input changes
if tickers:
    st.query_params["stocks"] = stocks_to_str(tickers)
else:
    # Clear the param if input is empty
    st.query_params.pop("stocks", None)

if not tickers:
    top_left_cell.info("Pick some stocks to compare", icon=":material/info:")
    st.stop()


right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

with right_cell:
    st.subheader("AI Price Forecast")
    
    # Placeholder prediction - replace with your ML model
    prediction = "UP"  # TODO: Replace with model.predict(ticker_input)
    change_percent = 4.25  # TODO: Replace with predicted percentage change
    confidence = 0.85  # TODO: Replace with model confidence score
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if prediction == "UP":
            st.metric(
                "Next Period Direction",
                "BULLISH ↑",
                delta=f"+{change_percent:.2f}%",
                delta_color="normal",
            )
        else:
            st.metric(
                "Next Period Direction",
                "BEARISH ↓",
                delta=f"-{change_percent:.2f}%",
                delta_color="inverse",
            )
    
    with col2:
        # Confidence gauge
        confidence_pct = confidence * 100
        if confidence_pct >= 80:
            confidence_label = "High"
            confidence_color = "🟢"
        elif confidence_pct >= 60:
            confidence_label = "Medium"
            confidence_color = "🟡"
        else:
            confidence_label = "Low"
            confidence_color = "🔴"
        
        st.metric(
            "Model Confidence",
            f"{confidence_color} {confidence_pct:.0f}%",
        )
    
    # Add model details
    with st.expander("Forecast Details"):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            st.metric("Predicted Change", f"{change_percent:.2f}%")
        with detail_col2:
            st.metric("Confidence Level", f"{confidence_pct:.0f}%")
        with detail_col3:
            st.metric("Time Horizon", "Next Period") 

@st.cache_resource(show_spinner=False, ttl="6h")
def load_data(tickers, period):
    tickers_obj = yf.Tickers(tickers)
    data = tickers_obj.history(period=period)
    if data is None:
        raise RuntimeError("YFinance returned no data.")
    return data["Close"]


# Load the data
try:
    data = load_data(tickers, horizon_map[horizon])
except yf.exceptions.YFRateLimitError as e:
    st.warning("YFinance is rate-limiting us :(\nTry again later.")
    load_data.clear()  # Remove the bad cache entry.
    st.stop()

empty_columns = data.columns[data.isna().all()].tolist()

if empty_columns:
    st.error(f"Error loading data for the tickers: {', '.join(empty_columns)}.")
    st.stop()

# Normalize prices (start at 1)
normalized = data.div(data.iloc[0])

latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
max_norm_value = max(latest_norm_values.items())
min_norm_value = min(latest_norm_values.items())

bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_left_cell:
    cols = st.columns(2)
    cols[0].metric(
        "Best stock",
        max_norm_value[1],
        delta=f"{round(max_norm_value[0] * 100)}%",
        width="content",
    )
    cols[1].metric(
        "Worst stock",
        min_norm_value[1],
        delta=f"{round(min_norm_value[0] * 100)}%",
        width="content",
    )


# Plot normalized prices (single stock)
with right_cell:
    if tickers:
        st.altair_chart(
            alt.Chart(
                normalized.reset_index().melt(
                    id_vars=["Date"], var_name="Stock", value_name="Normalized price"
                )
            )
            .mark_line()
            .encode(
                alt.X("Date:T"),
                alt.Y("Normalized price:Q").scale(zero=False),
                alt.Color("Stock:N"),
            )
            .properties(height=400)
        )

""
""

"""
## Raw data
"""

if tickers:
    data
