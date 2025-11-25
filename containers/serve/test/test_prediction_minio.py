"""
Test script for the updated prediction endpoint that fetches data from MinIO.

This demonstrates how to make predictions using the new API that automatically
fetches the last N days of gold tier data from MinIO for a given stock.
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Check if the service is running and model is loaded."""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_metadata(stock: str = "DIG"):
    """Get metadata about the loaded model for a specific stock."""
    print("=" * 60)
    print(f"Testing Metadata Endpoint for {stock}")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/metadata", params={"stock": stock})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction_minio(stock: str = "DIG", n_days: int = 30, horizons: list = None):
    """
    Test prediction endpoint with MinIO data fetching.
    
    Args:
        stock: Stock symbol (e.g., 'AAPL', 'GOOGL')
        n_days: Number of historical days to fetch from MinIO (e.g., 30)
        horizons: List of prediction horizons in days (e.g., [1, 5, 10])
    """
    if horizons is None:
        horizons = [1, 5, 10]
    
    print("=" * 60)
    print(f"Testing Prediction Endpoint - MinIO Auto-Fetch")
    print("=" * 60)
    print(f"Stock: {stock}")
    print(f"N Days: {n_days}")
    print(f"Horizons: {horizons}")
    print("-" * 60)
    
    payload = {
        "stock": stock,
        "n_days": n_days,
        "horizons": horizons
    }
    
    print(f"Request Payload:")
    print(json.dumps(payload, indent=2))
    print()
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction Results:")
            print(f"Run ID: {result['run_id']}")
            print(f"Model URI: {result['model_uri']}")
            print(f"\nPredictions by Horizon:")
            
            for horizon_result in result['predictions']:
                horizon = horizon_result['horizon']
                predictions = horizon_result['predictions']
                print(f"\n  Horizon {horizon} days:")
                print(f"    Total predictions: {len(predictions)}")
                print(f"    First 5 predictions: {predictions[:5]}")
                if len(predictions) > 5:
                    print(f"    Last 5 predictions: {predictions[-5:]}")
        else:
            print(f"Error Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API. Is the server running?")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    
    print()

def test_reload(stock: str = None):
    """Test reloading the model."""
    print("=" * 60)
    print(f"Testing Reload Endpoint{' for ' + stock if stock else ''}")
    print("=" * 60)
    
    payload = {"stock": stock} if stock else {}
    response = requests.post(f"{API_BASE_URL}/reload", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("StockFlow Model Serving - MinIO Integration Test")
    print("=" * 60 + "\n")
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: Get metadata
    test_metadata(stock="AAPL")
    
    # Test 3: Prediction with MinIO auto-fetch
    # This will automatically fetch the last 30 days of gold tier data
    # from MinIO for stock AAPL and generate predictions for 1, 5, and 10 days ahead
    test_prediction_minio(
        stock="AAPL",
        n_days=30,
        horizons=[1, 5, 10]
    )
    
    # Test 4: Short-term prediction (last 7 days, predict next day only)
    test_prediction_minio(
        stock="AAPL",
        n_days=7,
        horizons=[1]
    )
    
    # Test 5: Long-term prediction (last 60 days, predict 1, 3, 7, 14, 30 days ahead)
    test_prediction_minio(
        stock="AAPL",
        n_days=60,
        horizons=[1, 3, 7, 14, 30]
    )
    
    print("=" * 60)
    print("Testing Complete!")
    print("=" * 60)
