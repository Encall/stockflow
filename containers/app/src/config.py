import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Configuration
SERVE_API_URL = os.getenv("SERVE_API_URL")

STOCKS = [
    "AAPL", "ABBV", "ACN", "ADBE", "ADP", "AMD", "AMGN", "AMT", "AMZN", "APD",
    "AVGO", "AXP", "BA", "BK", "BKNG", "BMY", "BRK.B", "BSX", "C", "CAT",
    "CI", "CL", "CMCSA", "COST", "CRM", "CSCO", "CVX", "DE", "DHR", "DIS",
    "DUK", "ELV", "EOG", "EQR", "FDX", "GD", "GE", "GILD", "GOOG", "GOOGL",
    "HD", "HON", "HUM", "IBM", "ICE", "INTC", "ISRG", "JNJ", "JPM", "KO",
    "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "META", "MMC", "MO",
    "MRK", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE",
    "PG", "PLD", "PM", "PSA", "REGN", "RTX", "SBUX", "SCHW", "SLB", "SO",
    "SPGI", "T", "TJX", "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "V",
    "VZ", "WFC", "WM", "WMT", "XOM",
]

HORIZON_MAP = {
    "1 Week": "1wk",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}

DEFAULT_STOCKS = ["AAPL"]
