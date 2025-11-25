import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

def display_market_metrics(data):
    """Display current price metrics."""
    current_price = float(data['Close'].iloc[-1])
    previous_price = float(data['Close'].iloc[0])
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    st.metric(
        "Current Price",
        f"${current_price:.2f}",
        delta=f"${price_change:.2f} ({price_change_pct:+.2f}%)",
        delta_color="normal"
    )
    
    st.divider()
    
    # High/Low
    col1, col2 = st.columns(2)
    with col1:
        st.metric("High", f"${data['High'].max():.2f}")
    with col2:
        st.metric("Low", f"${data['Low'].min():.2f}")
    
    return current_price

def display_prediction(predicted_price, current_price):
    """Display prediction metrics and indicators."""
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    st.metric(
        "Tomorrow's Prediction",
        f"${predicted_price:.2f}",
        delta=f"{change_pct:+.2f}%",
        delta_color="normal"
    )
    
    # Direction indicator with emoji
    if change_pct > 0:
        st.markdown(
            f'<div style="color: #00ff41; font-size: 20px; text-align: center; font-weight: bold;">🟢 BULLISH ↑</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="color: #ff4136; font-size: 20px; text-align: center; font-weight: bold;">🔴 BEARISH ↓</div>',
            unsafe_allow_html=True
        )

def display_price_chart(data, ticker_input, predicted_price=None):
    """Display interactive price chart."""
    # Prepare data for chart
    chart_data = data[['Close']].reset_index()
    chart_data.columns = ['Date', 'Price']
    
    # Create interactive chart
    chart = alt.Chart(chart_data).mark_line(point=True, color='#1f77b4', size=2).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Price:Q', title=f'{ticker_input} Price ($)', scale=alt.Scale(zero=False)),
        tooltip=['Date:T', 'Price:Q']
    ).properties(
        height=450,
        width=600
    ).interactive()
    
    # Add prediction point if available
    if predicted_price is not None:
        tomorrow = datetime.now() + timedelta(days=1)
        
        pred_point = pd.DataFrame({
            'Date': [tomorrow],
            'Price': [predicted_price],
            'Type': ['Prediction']
        })
        
        pred_chart = alt.Chart(pred_point).mark_point(
            color='#ff7f0e',
            size=150,
            opacity=0.8
        ).encode(
            x='Date:T',
            y='Price:Q',
            tooltip=['Date:T', 'Price:Q']
        )
        
        chart = chart + pred_chart
    
    st.altair_chart(chart, use_container_width=True)

def display_deviation_analysis(predicted_price, actual_price):
    """Display deviation analysis results."""
    deviation_pct = abs((predicted_price - actual_price) / actual_price) * 100
    
    if deviation_pct <= 5:
        color = "🟢"
    elif deviation_pct <= 15:
        color = "🟡"
    else:
        color = "🔴"
    
    st.metric(
        f"{color} Deviation Percentage",
        f"{deviation_pct:.2f}%",
    )
    
    st.divider()
    
    # Detailed analysis
    st.subheader("📊 Deviation Details")
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    
    with analysis_col1:
        st.write(f"**Predicted:** ${predicted_price:.2f}")
    
    with analysis_col2:
        st.write(f"**Actual:** ${actual_price:.2f}")
    
    with analysis_col3:
        diff = abs(predicted_price - actual_price)
        st.write(f"**Absolute Deviation:** ${diff:.2f}")
    
    # Deviation direction
    st.subheader("🎯 Deviation Direction")
    if predicted_price > actual_price:
        deviation_type = "🔺 Positive Deviation"
    elif predicted_price < actual_price:
        deviation_type = "🔻 Negative Deviation"
    else:
        deviation_type = "✓ No deviation"
    
    st.info(f"**{deviation_type}**")
