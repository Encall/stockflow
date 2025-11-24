from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List
import os

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DriftDetector:
    def __init__(
        self,
        features: Optional[Iterable[str]] = None,
        save_dir: str = "reports/data_drift",
        file_prefix: str = "drift_report",
        save_html: bool = True,
        save_json: bool = True,
    ) -> None:
        self.features = list(features) if features is not None else None
        self.save_dir = Path(save_dir)
        self.file_prefix = file_prefix
        self.save_html = save_html
        self.save_json = save_json

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _select_features(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> List[str]:
        if self.features is not None:
            common = sorted(
                set(self.features)
                & set(reference_df.columns)
                & set(current_df.columns)
            )
        else:
            common = sorted(
                set(reference_df.columns) & set(current_df.columns)
            )

        if not common:
            raise ValueError(
                "No common features found between reference and current data."
            )

        return common

    def check(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        features = self._select_features(reference_df, current_df)

        ref = reference_df[features].copy()
        cur = current_df[features].copy()

        report = Report(metrics=[DataDriftPreset()])

        report.run(
            reference_data=ref,
            current_data=cur,
        )

        result_dict = report.as_dict()

        base_path = os.path.join(self.save_dir.as_posix(), self.file_prefix)

        if self.save_html:
            report.save_html(base_path + ".html")

        if self.save_json:
            report.save_json(base_path + ".json")

        return result_dict
