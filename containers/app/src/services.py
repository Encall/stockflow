import requests
import streamlit as st
from config import SERVE_API_URL

@st.cache_data(ttl=3600)
def call_prediction_api(symbol: str, end_date: str | None = None):
    """Call the prediction API from serve container."""
    try:
        url = f"{SERVE_API_URL}/predict"
        payload = {"stock": symbol}
        if end_date:
            payload["end_date"] = end_date
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if response contains error
        if "detail" in data:
            return {"error": data['detail']}
            
        return data
    except requests.exceptions.ConnectionError as e:
        return {"error": f"Cannot connect to serve API at {SERVE_API_URL}. Make sure the backend is running."}
    except requests.exceptions.Timeout:
        return {"error": "Prediction API request timed out"}
    except Exception as e:
        return {"error": f"Failed to get prediction from API: {str(e)}"}
