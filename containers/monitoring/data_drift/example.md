## How to Use `DriftDetector`

```python
from monitoring.data_drift import DriftDetector
import pandas as pd

# Load reference (training) and current (production) data
reference_df = pd.read_parquet("data/train.parquet")
current_df   = pd.read_parquet("data/today.parquet")

# Create detector (optional: specify features manually)
detector = DriftDetector(
    features=["age", "income", "transaction_amount"],  # or None for auto-select
    save_dir="reports/data_drift",
    file_prefix="daily_drift",
    save_html=True,
    save_json=True,
)

# Run drift check
result = detector.check(reference_df, current_df)

# Use the result in pipelines (e.g., alert/retrain)
print(result["metrics"])        # drift metrics
print(result["data_drift"])     # overall drift status
print("Report saved to:", detector.save_dir)
