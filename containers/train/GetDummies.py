import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta


def get_dummy(spec: dict, n_rows: int = 10, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"id": np.arange(n_rows)})

    for col, typ in spec.items():
        t = str(typ).lower()

        if t in ("int", "integer"):
            df[col] = rng.integers(0, 1000, size=n_rows)

        elif t in ("float", "double"):
            df[col] = rng.random(n_rows) * 100.0

        elif t in ("string", "str"):
            vals = rng.integers(0, 1000, size=n_rows)
            df[col] = [f"{col}_{v}" for v in vals]

        elif t in ("bool", "boolean"):
            df[col] = (rng.random(n_rows) > 0.5)

        elif t == "date":
            start = date.today()
            offsets = rng.integers(0, 30, size=n_rows)
            df[col] = [start - timedelta(days=int(d)) for d in offsets]

        elif t in ("timestamp", "datetime"):
            start = datetime.now()
            # random seconds within last 24 hours
            offsets = rng.integers(0, 24 * 3600, size=n_rows)
            df[col] = [start - timedelta(seconds=int(s)) for s in offsets]

        else:
            raise ValueError(f"Unsupported type: {typ}")

    # remove internal id if user didn't explicitly request it
    if "id" not in spec:
        df = df.drop(columns=["id"])

    return df
