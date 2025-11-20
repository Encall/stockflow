# 🥇 Gold Layer: Feature Engineering for Stock Prediction

This document details the features created in the **Gold Layer**. The goal is to transform raw time-series data (price and volume) into meaningful signals that a machine learning model can use to learn patterns.

---

## 1. Price-Derived Features

These features normalize the price and capture relative changes, which are often more stationary and useful for ML models than raw prices.

### `log_return`
- **What it is:** The natural logarithm of the ratio of the current day's closing price to the previous day's closing price.
- **Formula:** `ln(close_t / close_{t-1})`
- **Why it's important:** Log returns are time-additive and tend to be normally distributed, which are desirable properties for many financial models. It represents the continuous compounding rate of return.

### `gap_opening`
- **What it is:** The percentage difference between today's opening price and yesterday's closing price.
- **Formula:** `(open_today - close_yesterday) / close_yesterday`
- **Why it's important:** A significant gap can indicate overnight news or a change in market sentiment, providing a strong short-term signal.

---

## 2. Lag Features

These features provide the model with a view of the recent past, helping it identify trends and momentum.

### `return_lag_1`, `return_lag_3`, `return_lag_5`
- **What it is:** The `log_return` from 1, 3, and 5 days ago.
- **Why it's important:** Gives the model direct access to past performance, which is crucial for identifying auto-correlation and momentum patterns. For non-recurrent models (like Gradient Boosting), this is the primary way to provide historical context.

---

## 3. Window (Rolling) Statistics

These features capture trends and volatility over a specific period.

### `sma_14` (Simple Moving Average)
- **What it is:** The average closing price over the last 14 trading days.
- **Why it's important:** It's a classic trend-following indicator. It smooths out price data to help identify the direction of the underlying trend.

### `dist_from_sma` (Distance from SMA)
- **What it is:** The percentage distance of the current closing price from its 14-day SMA.
- **Formula:** `(close - sma_14) / sma_14`
- **Why it's important:** This normalizes the price relative to its recent trend. A large positive value might suggest the asset is overextended, while a large negative value might suggest it's oversold relative to its trend.

### `volatility_20`
- **What it is:** The standard deviation of `log_return` over the last 20 trading days.
- **Why it's important:** This is a direct measure of market risk and price fluctuation. High volatility indicates larger price swings and uncertainty.

---

## 4. Technical Indicators

These are well-established financial metrics designed to capture specific market dynamics like momentum, trend strength, and overbought/oversold conditions.

### `rsi` (Relative Strength Index)
- **What it is:** A momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100.
- **Why it's important:**
    - **Overbought/Oversold:** Traditionally, an RSI > 70 indicates an overbought condition (potential for a price drop), and an RSI < 30 indicates an oversold condition (potential for a price rise).
    - **Divergence:** A divergence between price and RSI can signal a potential trend reversal.

### `macd`, `macd_signal`, `macd_hist` (Moving Average Convergence Divergence)
- **What it is:** A trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a security’s price.
    - `macd`: The difference between the 12-period EMA and the 26-period EMA.
    - `macd_signal`: A 9-period EMA of the `macd` line.
    - `macd_hist`: The difference between the `macd` line and the `macd_signal` line.
- **Why it's important:**
    - **Trend Identification:** Crossovers of the MACD and signal lines are often used to signal buy (MACD crosses above signal) or sell (MACD crosses below signal) opportunities.
    - **Momentum:** The histogram (`macd_hist`) visualizes the distance between the two lines, indicating the strength of the momentum.

### `bb_upper`, `bb_lower`, `bb_middle` (Bollinger Bands)
- **What it is:** A set of bands plotted two standard deviations away from a simple moving average.
    - `bb_middle`: 20-day Simple Moving Average.
    - `bb_upper`: Middle Band + 2 * (20-day standard deviation of price).
    - `bb_lower`: Middle Band - 2 * (20-day standard deviation of price).
- **Why it's important:**
    - **Volatility:** The bands widen when volatility increases and narrow when it decreases.
    - **Relative Price:** The bands provide a relative definition of high and low. A price approaching the upper band is considered high; a price approaching the lower band is considered low.

---

## 5. Volume-Based Features

### `vol_change`
- **What it is:** The percentage change in trading volume from the previous day.
- **Formula:** `(volume_t - volume_{t-1}) / volume_{t-1}`
- **Why it's important:** A significant increase in volume can confirm the strength of a price trend (e.g., a price breakout accompanied by high volume is more significant).

---

## 6. Date-Time Features

These features capture cyclical patterns or seasonality in the market.

### `day_of_week`
- **What it is:** The day of the week, represented as an integer (Monday=0, Sunday=6).
- **Why it's important:** Captures weekly patterns, such as the "Monday effect" (tendency for prices to be lower on Mondays) or "Friday effect".

### `month`
- **What it is:** The month of the year, as an integer (1-12).
- **Why it's important:** Captures seasonal effects, such as the "January effect" or "Sell in May and go away" phenomenon.

---

## 7. Data Imputation Strategy: Preserving Data Integrity

When calculating features based on rolling windows (like moving averages or RSI), the initial data points in the time series will not have enough preceding data to compute a value. This naturally results in missing values (`NaN`).

Instead of discarding these valuable initial rows, we employ a two-step imputation strategy to ensure the dataset remains complete.

### Step 1: Fill with Zero
For features where a zero value is a logical starting point (e.g., `log_return` or `vol_change` on the very first day), we replace `NaN` with `0`.

### Step 2: Backfill (bfill)
For the remaining `NaN` values, which typically occur in rolling window features (`sma_14`, `rsi`, etc.), we use the **backfill** method. This method propagates the *first valid calculated value* backward to the start of the series.

**Example:** If the first valid `sma_14` is calculated on day 14, that value is used to fill the `NaN`s from day 1 to day 13.

### Why this approach?
This strategy ensures that **no rows are ever dropped** from the dataset. It preserves the original length of the time series while ensuring that the machine learning model receives a complete, non-null dataset for training.
