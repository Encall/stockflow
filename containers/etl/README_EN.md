# ETL Pipeline for Stock Data

## Overview
A pipeline for downloading (Bronze Layer), cleaning (Silver Layer), and creating features (Gold Layer) for stock data, following the **Medallion Architecture** pattern.

```
Bronze (Raw) → Silver (Cleaned) → Gold (Features)
     ↓              ↓                  ↓
  MinIO      ../../data/silver    ../../data/gold
```

---

## 📁 Project Structure

```
containers/etl/
├── etl.py                     # Entry point - runs the complete ETL pipeline
├── README_EN.md               # This document
├── GOLD_FEATURES.md           # Description of features in the Gold Layer
├── pyproject.toml             # uv dependencies
└── src/
    ├── minio.py              # Handles all MinIO up/down traffic
    ├── silver.py             # Cleans data from Bronze to create Silver
    └── gold.py               # Creates feature-engineered Gold data
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Using uv package manager (Rust-based, faster than pip)
cd containers/etl
uv sync
```

### 2. Run Pipeline
```bash
# Run from within the containers/etl folder
cd containers/etl
uv run python etl.py
```

---

## 📊 Pipeline Steps

### **Step 1: Download Data (Bronze)** (`src/minio.py`)
- Downloads `.parquet` files from the MinIO bucket `stockflow` (prefix: `raw/`).
- Saves to `data/bronze/`.

### **Step 2: Process Silver Layer** (`src/silver.py`)
- Reads raw data from `data/bronze/`.
- Cleans the data with the following rules:
    - Converts `date` to datetime and sorts.
    - Removes duplicates and rows with NaN values.
    - Removes prices ≤ 0 and volume < 0.
    - Validates `high >= low`.
- **Output**: Saves cleaned data to `data/silver/csv/` and `data/silver/parquet/`.

### **Step 3: Upload Silver Data** (`src/minio.py`)
- Uploads all files from `data/silver/csv/` and `data/silver/parquet/`.
- Saves to MinIO prefixes `stockflow/silver/csv/` and `stockflow/silver/parquet/`.

### **Step 4: Create Gold Layer** (`src/gold.py`)
- Reads cleaned `.csv` data from `data/silver/csv/`.
- Creates technical and financial features for the ML model.
- See `GOLD_FEATURES.md` for a full description of all features.
- **Output**: Saves feature-engineered data to `data/gold/csv/` and `data/gold/parquet/`.

### **Step 5: Upload Gold Data** (`src/minio.py`)
- Uploads all files from `data/gold/csv/` and `data/gold/parquet/`.
- Saves to MinIO prefixes `stockflow/gold/csv/` and `stockflow/gold/parquet/`.

---

## 📈 Data Structure

### Input Data (Bronze - Raw)
```
MinIO: stockflow/raw/*.parquet
Local: data/bronze/*.parquet
```
- **Columns**: `date`, `act_symbol`, `open`, `high`, `low`, `close`, `volume`

### Intermediate Data (Silver - Cleaned)
```
MinIO: stockflow/silver/csv/*.csv
       stockflow/silver/parquet/*.parquet
Local: data/silver/csv/*.csv
       data/silver/parquet/*.parquet
```
- **Format**: CSV and Parquet
- **Data**: Cleaned data, ready for feature creation.

### Output Data (Gold - Features)
```
MinIO: stockflow/gold/csv/*.csv
       stockflow/gold/parquet/*.parquet
Local: data/gold/csv/*.csv
       data/gold/parquet/*.parquet
```
- **Format**: CSV and Parquet
- **Data**: Silver data enriched with feature columns, such as:
    - `log_return`, `gap_opening`
    - `return_lag_1`, `return_lag_3`, `return_lag_5`
    - `sma_14`, `dist_from_sma`, `volatility_20`
    - `rsi`, `macd`, `macd_signal`, `macd_hist`
    - `bb_upper`, `bb_lower`, `bb_middle`
    - `vol_change`, `day_of_week`, `month`

---

## 📝 Sample Output

```
============================================================
🚀 Starting ETL Pipeline
Data directory: /path/to/stockflow/data
============================================================

📥 Step 1: Downloading data from MinIO...
✅ Downloaded 20 files

🧹 Step 2: Processing Silver Layer...
✅ Silver Layer processed for 20 files.

📤 Step 3: Uploading Silver data to MinIO...
✅ Uploaded 40 files to MinIO (silver layer).

✨ Step 4: Creating Gold Layer...
Found 20 files in silver directory.
✅ Gold Layer created for 20 files.

📤 Step 5: Uploading Gold data to MinIO...
✅ Uploaded 40 files to MinIO (gold layer).

============================================================
📊 Pipeline Summary
============================================================
📥 Downloaded: 20 files
🧹 Silver Layer Processed: 20 files
📤 Silver Layer Uploaded: 40 files
✨ Gold Layer Created: 20 files
📤 Gold Layer Uploaded: 40 files
--------------------
❌ Failed Silver Processing: 0 files
❌ Failed Gold Creation: 0 files

🎉 ETL Pipeline completed!
```
---

## 📚 Dependencies

```toml
[project]
dependencies = [
    "minio>=7.2.14",
    "pandas>=2.2.3",
    "pyarrow>=18.1.0",
    "python-dotenv>=1.0.1"
]
```

---

## 🏗️ Architecture Pattern

**Medallion Architecture**:
- 🥉 **Bronze** (Raw): Raw data from MinIO.
- 🥈 **Silver** (Cleaned): Cleaned and CSV-formatted data.
- 🥇 **Gold** (Features): Feature-engineered data ready for model training. ← **Implemented**
```

---

## 📁 Project Structure

```
containers/etl/
├── main.py                    # Entry point - runs complete ETL pipeline
├── README.md                  # This document
├── pyproject.toml            # uv dependencies (minio, pandas, pyarrow)
└── src/
    ├── download_data.py      # Download data from MinIO
    ├── clean_data.py         # Clean and validate data
    ├── fill_missing_dates.py # Check for missing dates (no filling)
    └── upload_data.py        # Upload back to MinIO
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Using uv package manager (Rust-based, faster than pip)
uv sync
```

### 2. Run Pipeline
```bash
uv run python main.py
```

---

## 📊 Pipeline Steps

### **Step 1: Download Data** (`download_data.py`)
- Downloads parquet files from MinIO bucket `stockflow`
- Saves to `../../data/`
- **Output**: List of downloaded filenames (20 files)

```python
# MinIO Configuration
ENDPOINT = "api.minio.encall.space"
BUCKET = "stockflow"
```

### **Step 2: Clean Data** (`clean_data.py`)
Cleans data with the following rules:

✅ **Processing**:
- Convert `date` to datetime
- Sort by date
- Remove duplicates
- Remove rows with NaN values
- Remove prices ≤ 0
- Remove negative volume

❌ **Validation Rules**:
- `high >= low` (remove if violated)

📁 **Output**: Saves to `../../data/cleaned/`

### **Step 3: Check Missing Dates** (`fill_missing_dates.py`)
- **Does NOT fill market holidays** (MLK Day, Christmas, etc.)
- Only reports gaps > 3 days (potential missing data)
- Keeps only actual trading days

```
Example:
⚠️  XLP_data.parquet: Found data gaps > 3 days (5 occurrences)
   2020-03-11 -> 2020-03-15 (4 days)  <- Normal weekend
```

### **Step 4: Upload Data** (`upload_data.py`)
- Uploads cleaned data back to MinIO
- Prefix: `stockflow/cleaned/`
- **Output**: Number of files uploaded

---

## 📈 Data Structure

### Input Data (Bronze - Raw)
```
MinIO: stockflow/*.parquet
Local: ../../data/*.parquet

Columns:
- date         (datetime)
- act_symbol   (string)    # Stock symbol (XLP, XLI, FUND, ...)
- open         (float)     # Opening price
- high         (float)     # Highest price
- low          (float)     # Lowest price
- close        (float)     # Closing price
- volume       (int)       # Trading volume

Initial rows: ~3,740 rows/file
```

### Output Data (Silver - Cleaned)
```
MinIO: stockflow/cleaned/*.parquet
Local: ../../data/cleaned/*.parquet

Same columns, but:
✅ No duplicates
✅ No NaN values
✅ No invalid prices (high >= low)
✅ No negative volumes
✅ Only actual trading days (no market holidays)

Final rows: ~3,740 rows/file (same as input)
```

---

## 🔧 Configuration

### MinIO Credentials
```python
# In src/download_data.py and src/upload_data.py
ENDPOINT = "api.minio.encall.space"
ACCESS_KEY = "your-access-key"
SECRET_KEY = "your-secret-key"
BUCKET = "stockflow"
```

### Data Paths
```python
# Relative paths from containers/etl/
RAW_DATA = "../../data/"              # Bronze layer
CLEANED_DATA = "../../data/cleaned/"  # Silver layer
```

---

## 📝 Sample Output

```
============================================================
🚀 Starting ETL Pipeline
============================================================

📥 Step 1: Downloading data from MinIO...
✅ Downloaded 20 files

🧹 Step 2: Cleaning data...
✅ Successfully cleaned: 20 files

📊 Row counts per file:
  • XLP_data.parquet: 3,740 rows
  • XLI_data.parquet: 3,740 rows
  ...

📅 Step 3: Checking for missing dates...
✅ Checked 20 files for missing dates

📤 Step 4: Uploading cleaned data to MinIO...
✅ Uploaded 20 files to MinIO

============================================================
📊 Pipeline Summary
============================================================
📥 Downloaded: 20 files
🧹 Cleaned: 20 files
📅 Checked for missing dates: 20 files
📤 Uploaded: 20 files
❌ Failed: 0 files

🎉 ETL Pipeline completed!
```

---

## 🎯 Next Steps (Gold Layer)

Not implemented yet - Feature Engineering:

- [ ] Technical Indicators (MA, RSI, MACD, Bollinger Bands)
- [ ] Rolling statistics (7-day, 30-day volatility)
- [ ] Price momentum features
- [ ] Volume analysis
- [ ] Multi-timeframe features

---

## 🛠️ Development

### Package Manager
Using **uv** (Rust-based Python package manager):
```bash
# Add package
uv add package-name

# Run Python script
uv run python script.py

# Sync dependencies
uv sync
```

### VS Code Setup
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/containers/etl/.venv/bin/python"
}
```

---

## 📌 Notes

1. **No market holidays filling**: Pipeline keeps only actual trading days to avoid data distortion when calculating indicators
2. **Loose validation**: Only checks `high >= low` because open/close prices can be outside this range in special cases
3. **MinIO Prefix**: Cleaned data is stored at `stockflow/cleaned/` prefix, not a separate bucket

---

## 📚 Dependencies

```toml
[project]
dependencies = [
    "minio>=7.2.14",
    "pandas>=2.2.3",
    "pyarrow>=18.1.0",
]
```

---

## 🏗️ Architecture Pattern

**Medallion Architecture**:
- 🥉 **Bronze** (Raw): Raw data from MinIO
- 🥈 **Silver** (Cleaned): Cleaned and validated data ← **Current position**
- 🥇 **Gold** (Features): Feature-engineered data ready for model training ← **Not implemented yet**
